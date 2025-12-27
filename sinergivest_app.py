# sinergivest_botves_stateful.py
"""
BotVes - perbaikan percakapan:
- Stateful conversation: simpan nama, horizon, jenis instrumen, budget, risk_pref
- Intent handling yang lebih deterministik & to-the-point
- Parse budget (format '1,2 juta', '1200000', '1.2m', '1 juta', '500rb')
- Jika budget ada => rekomendasi saham sesuai preferensi (aggressive/conservative/balanced)
- Mendukung upload file (.csv / .xls / .xlsx) & 'analisis file' command
- Responses edukatif + safety disclaimers
"""
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from scipy.stats import norm, skew, kurtosis

# optional yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

st.set_page_config(page_title="SinergiVest — BotVes (stateful)", layout="wide")

# --- minimal style
st.markdown("""
<style>
.user { background:#6b6b6b; color:white; padding:10px; border-radius:8px; margin:8px 0; text-align:right; }
.bot { background:#eef7ff; color:#000; padding:10px; border-radius:8px; margin:8px 0; text-align:left; }
.small { font-size:0.85rem; color:#666; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Session state init (conversation slots)
# -------------------------
if "page" not in st.session_state: st.session_state.page = "home"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "uploaded_df" not in st.session_state: st.session_state.uploaded_df = None
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "name": None,
        "age": None,
        "horizon_years": None,
        "instruments": [],     # e.g. ['saham','reksa dana']
        "risk_pref": None,     # 'conservative'|'balanced'|'aggressive'
        "budget": None,        # in IDR integer
        "expecting_budget": False,
        "expecting_horizon": False
    }
# cache for fetched series
if "last_stock_cache" not in st.session_state: st.session_state.last_stock_cache = {}

# sample tickers
TICKERS = ["BBCA.JK","BBRI.JK","TLKM.JK","ASII.JK","BMRI.JK","GOTO.JK","UNVR.JK","ICBP.JK","AMRT.JK","BBNI.JK"]

# -------------------------
# Helpers
# -------------------------
def parse_budget(text: str):
    """Extract an integer budget in IDR from free text (Indonesian forms).
    Recognizes: '1,2 juta', '1.2 juta', '1200000', '1 juta', '500rb', '500 ribu'
    Returns integer or None.
    """
    if not text:
        return None
    t = text.lower()
    # normalize separators
    t = t.replace('.', '').replace(' ', '')
    # search pattern like 1,2juta or 1200000
    m = re.search(r'([\d,]+)(juta|jt|ribu|rb|k|m)?', t)
    if not m:
        return None
    num = m.group(1).replace(',','')
    unit = m.group(2)
    try:
        val = float(num)
    except:
        return None
    if unit in ('juta','jt'):
        return int(val * 1_000_000)
    if unit in ('ribu','rb','k'):
        return int(val * 1_000)
    if unit == 'm':  # treat m as million
        return int(val * 1_000_000)
    # no unit: interpret heuristically
    if val < 1000:  # likely user typed '1' meaning 1 juta
        return int(val * 1_000_000)
    return int(val)

@st.cache_data(ttl=3600)
def fetch_close_series(ticker: str, period="1mo", interval="1d"):
    """Return close price series indexed by date (datetime.date). fallback to cached or synthetic."""
    t = ticker.upper()
    if HAS_YFINANCE:
        try:
            df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=True)
            if df is not None and not df.empty and "Close" in df.columns:
                s = df["Close"].copy()
                s.index = pd.to_datetime(s.index).date
                st.session_state.last_stock_cache[t] = s
                return s
        except Exception:
            pass
    # fallback cached
    if t in st.session_state.last_stock_cache:
        return st.session_state.last_stock_cache[t]
    # synthetic realistic
    dates = pd.bdate_range(end=datetime.now().date(), periods=30)
    base = 1000 + np.cumsum(np.random.normal(0, 10, len(dates)))
    s = pd.Series(base, index=dates.date)
    st.session_state.last_stock_cache[t] = s
    return s

def compute_metrics(series: pd.Series):
    if series is None or len(series) < 2:
        return {}
    ln = np.log(series / series.shift(1)).dropna()
    mu = ln.mean()
    sigma = ln.std(ddof=0)
    S = skew(ln)
    K_excess = kurtosis(ln, fisher=True)
    z = norm.ppf(0.95)
    z_cf = z + (1/6)*(z**2 - 1)*S + (1/24)*(z**3 - 3*z)*K_excess - (1/36)*(2*z**3 - 5*z)*(S**2)
    var95 = float(-(mu + sigma * z))
    var95_cf = float(-(mu + sigma * z_cf))
    return {
        "last": float(series.iloc[-1]),
        "period_return": float(np.exp(ln.sum()) - 1),
        "mu": float(mu), "sigma": float(sigma),
        "skew": float(S), "kurtosis_excess": float(K_excess),
        "VaR95_log": var95, "VaR95_CF_log": var95_cf
    }

def recommend_stocks_for_budget(tickers, budget:int, approach="balanced", top_n=3):
    cand = []
    for t in tickers:
        s = fetch_close_series(t)
        if s is None or len(s) < 6: continue
        last = float(s.iloc[-1])
        # allow recommending even if last > budget: propose fractional/alternatives? for now require last <= budget
        if last > budget:
            continue
        m = compute_metrics(s)
        if not m: continue
        # score definitions
        if approach == "aggressive":
            score = m["period_return"]  # favor high return
        elif approach == "conservative":
            score = -m["sigma"]         # favor low volatility
        else:
            score = m["period_return"] / (m["sigma"] + 1e-9)  # balanced
        cand.append({
            "ticker": t, "last": last, "return": m["period_return"], "sigma": m["sigma"], "skew": m["skew"], "kurtosis_excess": m["kurtosis_excess"], "score": score
        })
    if not cand:
        return []
    df = pd.DataFrame(cand).sort_values("score", ascending=False)
    return df.head(top_n).to_dict(orient="records")

def analyze_uploaded_df(df: pd.DataFrame):
    # same logic as earlier: look for ticker & qty
    summary = {}
    cols = [c.lower() for c in df.columns]
    def col_like(sub):
        for i,c in enumerate(cols):
            if sub in c:
                return df.columns[i]
        return None
    if col_like('ticker') and (col_like('qty') or col_like('quantity')):
        ticker_col = col_like('ticker')
        qty_col = col_like('qty') or col_like('quantity')
        price_col = col_like('price') or col_like('close') or col_like('last')
        if price_col:
            df['value'] = df[qty_col] * df[price_col]
        else:
            prices = []
            for t in df[ticker_col].astype(str):
                s = fetch_close_series(t.strip()+('.JK' if not t.strip().upper().endswith('.JK') else ''))
                prices.append(float(s.iloc[-1]) if s is not None and len(s)>0 else np.nan)
            df['price_fetched'] = prices
            df['value'] = df[qty_col] * df['price_fetched']
        total = df['value'].sum()
        df['alloc_pct'] = df['value'] / total * 100
        summary['table'] = df
        summary['total'] = float(total)
        return summary
    else:
        numeric = df.select_dtypes(include='number')
        summary['numeric_summary'] = numeric.describe().to_dict()
        return summary

# -------------------------
# Conversational handler (stateful)
# -------------------------
def handle_user_message(msg: str):
    msg = msg.strip()
    if not msg:
        return "Silakan ketik pertanyaan atau perintah (mis. 'saham apa', 'analisis file')."

    lower = msg.lower()
    profile = st.session_state.user_profile

    # 1) name detection
    m_name = re.search(r'\b(?:nama saya|nama ku|namaku|saya bernama|saya adalah)\s+([A-Za-z0-9\-_ ]{2,30})', lower)
    if m_name:
        name = m_name.group(1).strip().title()
        profile['name'] = name
        return f"Salam, {name}! Senang bertemu. Sebutkan tujuan investasi Anda (mis. dana pendidikan/jangka panjang) atau modal yang ingin diinvestasikan."

    # 2) direct short "Nama ku Okta" or "aku Okta"
    m_name2 = re.search(r'\b(?:aku|saya)\s+([A-Z][a-z]{1,20})\b', msg)
    if m_name2 and not profile.get('name'):
        # be conservative: ensure it's likely a name (start uppercase) - if not, ignore
        candidate = m_name2.group(1).strip().title()
        profile['name'] = candidate
        return f"Senang bertemu, {candidate}! Untuk membantu, sebutkan tujuan & horizon atau modal investasi Anda."

    # 3) horizon detection (jangka panjang / pendek)
    if any(k in lower for k in ["jangka panjang","persiapan pendidikan","untuk pendidikan","long term","long-term","lama"]):
        profile['horizon_years'] = profile.get('horizon_years') or 5
        profile['instruments'] = list(set(profile.get('instruments',[]) + ['saham']))
        # confirm
        return "Terima kasih. Karena tujuan jangka panjang (pendidikan), saham dan reksa dana saham cocok; apakah Anda ingin rekomendasi saham konkret atau contoh alokasi?"

    if any(k in lower for k in ["jangka pendek","short term","untuk liburan","sebulan","beberapa bulan"]):
        profile['horizon_years'] = profile.get('horizon_years') or 1
        return "Baik, horizon pendek. Rekomendasi sebaiknya konservatif (rekasa dana pasar uang / obligasi). Mau saya contohkan alokasi konservatif?"

    # 4) instrument interest
    if any(k in lower for k in ["saham","reksa dana","obligasi","sukuk"]):
        # record which instruments mentioned
        inst = []
        if "saham" in lower: inst.append("saham")
        if "reksa" in lower or "reksa dana" in lower: inst.append("reksa dana")
        if "obligasi" in lower or "sukuk" in lower: inst.append("obligasi")
        profile['instruments'] = list(set(profile.get('instruments',[]) + inst))
        # if user only answered "saham" as reply to question, continue flow: ask budget or risk pref
        if lower.strip() in ("saham","reksa dana","obligasi","saham saja"):
            # prompt for budget
            profile['expecting_budget'] = True
            return "Baik, Anda tertarik pada saham. Berapa modal yang ingin Anda mulai (mis. '1,2 juta' atau '1200000')? Atau ketik 'tanpa modal' jika ingin simulasi saja."
        # else acknowledge and continue
        return f"Terima kasih. Saya catat minat Anda pada: {', '.join(profile['instruments'])}. Apakah Anda punya modal yang ingin dipakai sekarang?"

    # 5) budget parsing (either user asked directly or in sentence)
    budget_val = parse_budget(msg)
    if budget_val:
        profile['budget'] = budget_val
        profile['expecting_budget'] = False
        # if user previously asked for recommendations, handle immediately
        # detect if they earlier asked "saham apa" etc by checking last messages
        last_user_asked_rec = any("saham apa" in m.get('content','').lower() or "saham yang cocok" in m.get('content','').lower() for m in st.session_state.chat_history if m['role']=='user')
        # determine risk pref if mentioned
        if any(w in lower for w in ["aman","konservatif"]):
            profile['risk_pref'] = "conservative"
        elif any(w in lower for w in ["agresif","tinggi","besar"]):
            profile['risk_pref'] = "aggressive"
        else:
            profile['risk_pref'] = profile.get('risk_pref') or "balanced"
        # provide immediate recommendation if user asked earlier about saham
        if last_user_asked_rec or 'saham' in profile.get('instruments',[]):
            # choose approach
            approach = profile['risk_pref'] or "balanced"
            recs = recommend_stocks_for_budget(TICKERS, profile['budget'], approach=approach, top_n=3)
            if recs:
                lines = [f"Dengan modal Rp {profile['budget']:,} dan preferensi '{approach}', rekomendasi (edukatif):"]
                for r in recs:
                    lines.append(f"- {r['ticker']}: harga ~Rp {int(r['last']):,}, estimasi return (periode): {r['return']*100:.2f}% , volatilitas: {r['sigma']:.4f}")
                lines.append("Catatan: ini panduan edukatif. Untuk return lebih tinggi, fokus pada saham growth/volume kecil (risiko lebih tinggi).")
                return "\n".join(lines)
            else:
                return f"Maaf, tidak menemukan saham top dengan harga <= Rp {profile['budget']:,}. Pertimbangkan reksa dana indeks atau strategi DCA (beli bertahap)."
        # otherwise just confirm
        return f"Catat: modal Anda Rp {profile['budget']:,}. Anda dapat menanyakan 'saham apa yang cocok' sekarang."

    # 6) direct request: "saham apa" or "saham apa yang cocok"
    if any(phrase in lower for phrase in ["saham apa", "saham yang cocok", "apa saham"]):
        # if budget known, recommend immediately
        if profile.get('budget'):
            approach = "balanced"
            if any(w in lower for w in ["return tinggi","high return","ingin return tinggi","agresif","besar keuntungan"]):
                approach = "aggressive"
            elif any(w in lower for w in ["aman","konservatif","amanlah"]):
                approach = "conservative"
            recs = recommend_stocks_for_budget(TICKERS, profile['budget'], approach=approach, top_n=3)
            if recs:
                lines = [f"Dengan modal Rp {profile['budget']:,} dan preferensi '{approach}', saya rekomendasikan:"]
                for r in recs:
                    lines.append(f"- {r['ticker']}: harga ~Rp {int(r['last']):,}, return est.: {r['return']*100:.2f}% , vol: {r['sigma']:.4f}")
                lines.append("Ingat: rekomendasi ini edukatif. Untuk return tinggi, alokasikan porsi kecil ke saham berisiko; sisanya simpan di instrumen konservatif.")
                return "\n".join(lines)
            else:
                return f"Saya tidak menemukan saham top yang harganya <= Rp {profile['budget']:,}. Anda ingin saya sarankan reksa dana yang cocok?"
        # ask for budget
        profile['expecting_budget'] = True
        return "Berapa modal Anda untuk investasi (mis. '1,2 juta')? Dengan modal itu saya akan rekomendasikan saham yang cocok."

    # 7) user asks about specific ticker (BBCA, TLKM etc)
    m_t = re.search(r'\b([A-Za-z]{2,5}(?:\.JK)?)\b', msg)
    if m_t:
        tok = m_t.group(1).upper()
        if not tok.endswith('.JK'):
            tok = tok + '.JK'
        s = fetch_close_series(tok)
        metrics = compute_metrics(s)
        if not metrics:
            return f"Data untuk {tok} tidak cukup untuk analisis."
        else:
            return (f"Ringkasan {tok}: Harga terakhir ~Rp {metrics['last']:,.0f}. "
                    f"Perkiraan return periode: {metrics['period_return']*100:.2f}% , volatilitas (sigma): {metrics['sigma']:.4f}, "
                    f"VaR95(CF,log): {metrics['VaR95_CF_log']:.4%}. (Ini informasi edukatif, bukan rekomendasi beli/jual.)")

    # 8) analyze uploaded file request
    if any(k in lower for k in ["analisis file","analisis portofolio","analyze file","analyze portfolio"]):
        if st.session_state.uploaded_df is None:
            return "Saya tidak menemukan file yang diunggah. Silakan unggah file (.csv/.xls/.xlsx) di menu 'Upload & Analyze' atau uploader di halaman Chat."
        summ = analyze_uploaded_df(st.session_state.uploaded_df)
        if 'table' in summ:
            total = summ['total']
            return f"Saya menemukan file portofolio. Estimasi total nilai: Rp {total:,.0f}. Saya menampilkan ringkasan di halaman Upload & Analyze."
        else:
            return "File tidak berisi kolom Ticker/Qty. Saya sudah menampilkan ringkasan statistik numerik di halaman Upload & Analyze."

    # 9) small talk / help
    if any(w in lower for w in ["halo","hai","selamat","pagi","siang","sore"]):
        return "Halo! Saya BotVes. Sebutkan tujuan investasi atau tanya 'saham apa yang cocok dengan modal X' atau unggah file portofolio dan ketik 'analisis file'."

    if any(w in lower for w in ["cara mulai","bagaimana mulai","mulai investasi"]):
        return ("Langkah singkat: 1) Tentukan tujuan & horizon; 2) Pilih profil risiko; 3) Mulai kecil & rutin (DCA); 4) Gunakan alokasi konservatif bila pemula. "
                "Mau saya buat contoh alokasi untuk tujuan pendidikan jangka panjang? (ketik 'contoh alokasi')")

    # 10) user gave modal words like 'modal 1,2 juta' embedded earlier but parse_budget may have failed; handle numbers like '1,2 juta' with comma removal earlier.
    # Already handled by parse_budget at top.

    # 11) fallback
    return ("Maaf, saya belum paham. Coba: 'saham apa yang cocok dengan modal 1 juta', "
            "'analisis file', atau tanyakan tentang ticker (mis. 'BBCA').")

# -------------------------
# UI layout & usage
# -------------------------
st.title("SinergiVest — BotVes (improved & stateful)")
st.write("Catatan: ini alat edukatif. Jangan gunakan sebagai nasihat investasi formal.")

# simple uploader
uploaded = st.file_uploader("Unggah file portofolio (.csv/.xls/.xlsx) — opsional (Upload & Analyze)", type=['csv','xls','xlsx'])
if uploaded:
    try:
        if uploaded.name.lower().endswith('.csv'):
            dfu = pd.read_csv(uploaded)
        else:
            dfu = pd.read_excel(uploaded)
        st.session_state.uploaded_df = dfu
        st.success(f"File {uploaded.name} tersimpan di sesi.")
        st.dataframe(dfu.head(10))
    except Exception as e:
        st.error("Gagal membaca file: " + str(e))

st.markdown("---")
# render chat history
for m in st.session_state.chat_history:
    if m['role']=='user':
        st.markdown(f"<div class='user'>{m['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot'>{m['content']}</div>", unsafe_allow_html=True)

# initial greeting if empty
if not st.session_state.chat_history:
    greeting = ("Halo! Perkenalkan nama saya BotVes — asisten edukasi investasi. "
                "Saya akan membantu merekomendasikan opsi investasi edukatif. "
                "Untuk memulai, Anda bisa berkata mis. 'Nama ku Okta', 'Tujuan: dana pendidikan jangka panjang', atau 'saham apa yang cocok'.")
    st.session_state.chat_history.append({'role':'assistant','content':greeting})
    st.experimental_rerun()

# input
user_text = st.text_input("Ketik pesan Anda...", key="chat_input")
if st.button("Kirim"):
    if not user_text or user_text.strip()=="":
        st.warning("Tolong ketik pesan terlebih dahulu.")
    else:
        # append user message
        st.session_state.chat_history.append({'role':'user','content': user_text})
        # handle
        reply = handle_user_message(user_text)
        st.session_state.chat_history.append({'role':'assistant','content': reply})
        # immediate render (rerun to reflow)
        st.experimental_rerun()

# helper info panel
st.markdown("---")
st.markdown("**Tips cepat:** ketik `saham apa yang cocok dengan modal 1,2 juta` — BotVes akan langsung menjawab jika modal terdeteksi. Ketik `analisis file` setelah upload file untuk ringkasan portofolio.")
