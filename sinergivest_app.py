# sinergivest_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go

# optional yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    HAS_YFINANCE = False

# page config
st.set_page_config(page_title="SinergiVest", layout="wide")

# -----------------------------
# CSS styling (header, oval buttons, chat bubbles)
# -----------------------------
st.markdown("""
<style>
/* Header center */
.header { text-align:center; margin-bottom: 20px; }
.header h1 { font-size: 50px; margin: 0; color: #ffffff; }
.header p { margin: 0 0 30px 0; color: #cccccc; }

/* Carousel boxes */
.carousel-box {
  background: #111217;
  border-radius: 12px;
  padding: 18px;
  min-height: 260px;
  color: #fff;
  box-shadow: 0 6px 18px rgba(0,0,0,0.5);
}

/* Oval buttons */
.btn-oval > button {
  border-radius: 50px !important;
  height: 56px !important;
  font-size: 16px !important;
  font-weight: 600 !important;
  padding: 0 26px !important;
}
.btn-green > button { background: #2ecc71 !important; color: #fff !important; border: none !important; }
.btn-blue > button { background: #0077be !important; color: #fff !important; border: none !important; }
.btn-white > button { background: #ffffff !important; color: #000000 !important; border: 1px solid #cccccc !important; }

/* Tombol khusus */
button.start-bot { background-color: #2ecc71 !important; color: white !important; border-radius: 50px !important; }
button.login { background-color: #0077be !important; color: white !important; border-radius: 50px !important; }
button.nav { background-color: white !important; color: black !important; border: 1px solid gray !important; border-radius: 50px !important; }

/* Override for all buttons to ensure oval shape */
button {
  border-radius: 50px !important;
}

/* Chat bubbles */
.user { background:#6b6b6b; color:white; padding:10px; border-radius:10px; margin:8px 0; text-align:right; }
.bot  { background:#eef7ff; color:#000; padding:10px; border-radius:10px; margin:8px 0; text-align:left; }

/* Responsive tweaks */
@media (max-width: 800px) {
  .header h1 { font-size: 36px; }
  .carousel-box { min-height: 220px; padding:12px; }
  .user, .bot { padding: 8px; font-size: 14px; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session state defaults
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"   # home / chat / login
if "stock_idx" not in st.session_state:
    st.session_state.stock_idx = 0
if "edu_idx" not in st.session_state:
    st.session_state.edu_idx = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "name": None, "horizon_years": None, "instruments": [], "risk_pref": None, "budget": None, "expecting_budget": False
    }
# simple cache for fetched series
if "stock_cache" not in st.session_state:
    st.session_state.stock_cache = {}

# sample tickers & OJK videos (YouTube links)
TICKERS = [
    "BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "BMRI.JK", "UNTR.JK", "GOTO.JK", "INDF.JK", "ICBP.JK", "AMRT.JK",
    "BBNI.JK", "ANTM.JK", "CPIN.JK", "JSMR.JK", "KLBF.JK", "MDKA.JK", "PGAS.JK", "PTBA.JK", "SMGR.JK", "UNVR.JK"
]
TICKER_NAMES = {
    "BBCA.JK": "Bank Central Asia",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "TLKM.JK": "Telkom Indonesia",
    "ASII.JK": "Astra International",
    "BMRI.JK": "Bank Mandiri",
    "UNTR.JK": "United Tractors",
    "GOTO.JK": "GoTo Gojek Tokopedia",
    "INDF.JK": "Indofood Sukses Makmur",
    "ICBP.JK": "Indofood CBP Sukses Makmur",
    "AMRT.JK": "Sumber Alfaria Trijaya",
    "BBNI.JK": "Bank Negara Indonesia",
    "ANTM.JK": "Aneka Tambang",
    "CPIN.JK": "Charoen Pokphand Indonesia",
    "JSMR.JK": "Jasa Marga",
    "KLBF.JK": "Kalbe Farma",
    "MDKA.JK": "Merck Sharp & Dohme Indonesia",
    "PGAS.JK": "Perusahaan Gas Negara",
    "PTBA.JK": "Bukit Asam",
    "SMGR.JK": "Semen Indonesia",
    "UNVR.JK": "Unilever Indonesia"
}
EDU_VIDEOS = [
    {"title":"Digital Finance Literacy - OJK","url":"https://www.youtube.com/watch?v=GFgbxJyCSmE"},
    {"title":"Cerdas Berinvestasi - OJK","url":"https://www.youtube.com/watch?v=Is3BfJN3bp0"},
    {"title":"Bijak Berinvestasi - OJK","url":"https://www.youtube.com/watch?v=FcMG-ZMQP1g"}
]

# List of Top 20 Blue Chip Stocks
stocks = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "BMRI.JK", "ASII.JK", "UNTR.JK", "GOTO.JK", "UNVR.JK", "ICBP.JK", "ADRO.JK", "BBNI.JK", "PGAS.JK", "CPIN.JK", "KLBF.JK", "ANTM.JK", "PTBA.JK", "SMGR.JK", "INDF.JK", "AMRT.JK", "BRIS.JK"]

# Dictionary mapping ticker to company name
stock_names = {
    "BBCA.JK": "Bank Central Asia",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "TLKM.JK": "Telkom Indonesia",
    "BMRI.JK": "Bank Mandiri",
    "ASII.JK": "Astra International",
    "UNTR.JK": "United Tractors",
    "GOTO.JK": "Gojek Tokopedia",
    "UNVR.JK": "Unilever Indonesia",
    "ICBP.JK": "Indofood CBP",
    "ADRO.JK": "Adaro Energy",
    "BBNI.JK": "Bank Negara Indonesia",
    "PGAS.JK": "Perusahaan Gas Negara",
    "CPIN.JK": "Charoen Pokphand Indonesia",
    "KLBF.JK": "Kalbe Farma",
    "ANTM.JK": "Aneka Tambang",
    "PTBA.JK": "Bukit Asam",
    "SMGR.JK": "Semen Indonesia",
    "INDF.JK": "Indofood Sukses Makmur",
    "AMRT.JK": "Alfamart",
    "BRIS.JK": "Bank Syariah Indonesia"
}

# -----------------------------
# Utility: get stock series (1 month, daily) with fallback
# -----------------------------
@st.cache_data(ttl=3600)
def get_stock_series(ticker, period="1mo", interval="1d"):
    t = ticker.upper()
    # try yfinance
    if HAS_YFINANCE:
        try:
            df = yf.download(t, period=period, interval=interval, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                # Handle MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    # For single ticker, access the first level
                    close_col = ("Close", t) if ("Close", t) in df.columns else "Close"
                    if close_col in df.columns:
                        s = df[close_col].copy()
                    else:
                        # Fallback to first available Close
                        s = df.xs("Close", axis=1, level=0).iloc[:, 0].copy()
                else:
                    if "Close" in df.columns:
                        s = df["Close"].copy()
                    else:
                        raise ValueError("No 'Close' column found")
                # Ensure float32 for accuracy
                s = s.astype(np.float32)
                # keep full datetime index for better plotly x-axis
                s.index = pd.to_datetime(s.index)
                return s
        except Exception as e:
            st.write(f"Error fetching {t}: {e}")
    # fallback: synthetic business days series
    rng = pd.bdate_range(end=pd.Timestamp.now(), periods=30)
    base = 1000 + np.cumsum(np.random.normal(0, 10, len(rng)))
    s = pd.Series(base, index=rng, dtype=np.float32)
    return s

# -----------------------------
# Small analytics helpers (log-return & Cornish-Fisher VaR)
# -----------------------------
from scipy.stats import norm, skew, kurtosis

def compute_stats(series):
    if series is None or len(series) < 3:
        return None
    # Handle if series is DataFrame or 2D array
    if isinstance(series, pd.DataFrame):
        series = series.squeeze()  # Convert to Series if single column
    if hasattr(series, 'shape') and len(series.shape) > 1:
        series = series.flatten()
    s = pd.Series(series).astype(float)
    ln = np.log(s / s.shift(1)).dropna()
    if ln.empty:
        return None
    mu = ln.mean()
    sigma = ln.std(ddof=0)
    S = skew(ln)
    K = kurtosis(ln, fisher=True)
    z = norm.ppf(0.95)
    z_cf = z + (1/6)*(z**2 - 1)*S + (1/24)*(z**3 - 3*z)*K - (1/36)*(2*z**3 - 5*z)*(S**2)
    # VaR on log-return scale; convert to price approximate (1-day)
    var95 = -(mu + sigma * z)
    var95_cf = -(mu + sigma * z_cf)
    return {"mu":mu, "sigma":sigma, "skew":S, "kurtosis_excess":K, "VaR95":var95, "VaR95_CF":var95_cf, "last": float(s.iloc[-1])}

# -----------------------------
# Conversational handler (simple, to-the-point)
# -----------------------------
def parse_budget(text):
    if not text: return None
    t = text.lower().replace(' ', '')
    # Regex to capture number with optional decimal comma
    m = re.search(r'([\d,]+)(juta|jt|ribu|rb|k|m|biliar)?', t)
    if not m: return None
    num_str = m.group(1)
    # Replace comma with dot for decimal (Indonesian locale: comma is decimal separator)
    num_str = num_str.replace(',', '.')
    try:
        val = float(num_str)
    except:
        return None
    unit = m.group(2)
    if unit in ('juta','jt','m'): return int(val * 1_000_000)
    if unit in ('ribu','rb','k'): return int(val * 1_000)
    if unit in ('biliar',): return int(val * 1_000_000_000)
    # If no unit and val < 1000, assume juta
    if val < 1000: return int(val * 1_000_000)
    return int(val)

def parse_risk_preference(text):
    ml = text.lower()
    if any(word in ml for word in ["untung tinggi", "agresif", "risiko tinggi", "return tinggi"]):
        return "aggressive"
    elif any(word in ml for word in ["aman", "konservatif", "rendah risiko", "stabil"]):
        return "conservative"
    else:
        return "balanced"

def recommend_for_budget(budget, approach="balanced", top_n=3):
    cand = []
    for t in TICKERS:
        s = get_stock_series(t)
        if s is None or len(s)<5: continue
        last = float(s.iloc[-1])
        # Assume lot size 100 shares, skip if lot price > budget (strict budgeting)
        lot_price = last * 100
        if lot_price > budget:
            continue
        stats = compute_stats(s)
        if stats is None: continue
        if approach=="aggressive":
            score = stats["mu"]
        elif approach=="conservative":
            score = -stats["sigma"]
        else:
            score = stats["mu"] / (stats["sigma"] + 1e-9)
        cand.append({"ticker":t, "last":last, "lot_price":lot_price, "return":stats["mu"], "sigma":stats["sigma"], "score":score})
    if not cand: return []
    df = pd.DataFrame(cand).sort_values("score", ascending=False)
    return df.head(top_n).to_dict(orient="records")

def handle_message(msg):
    if not msg or not msg.strip():
        return "Tolong ketik pesan atau perintah (mis. 'saham apa yang cocok', '1,2 juta', atau 'BBCA')."
    m = msg.strip()
    ml = m.lower()
    # name declaration
    if any(p in ml for p in ["nama saya","namaku","aku adalah","saya nama","nama ku"]):
        # ekstrak nama sederhana
        name = re.sub(r'.*(nama saya|namaku|aku adalah|saya nama|nama ku)\s*','',ml,flags=re.I).strip().title()
        if name:
            st.session_state.user_profile["name"] = name
            return f"Salam {name}! Sebutkan tujuan atau modal Anda untuk mulai."
    # budget parse direct
    b = parse_budget(m)
    if b:
        st.session_state.user_profile["budget"] = b
        approach = parse_risk_preference(m)  # Extract from message
        st.session_state.user_profile["risk_pref"] = approach
        recs = recommend_for_budget(b, approach=approach, top_n=3)
        if recs:
            # Greeting & Confirmation
            greetings = [
                "Halo! Menarik sekali minat investasinya.",
                "Hai! Bagus, kita mulai dengan modal yang jelas.",
                "Salam! Siap membantu dengan dana tersebut.",
                "Wow, semangat investasinya luar biasa!"
            ]
            greeting = np.random.choice(greetings)
            response = f"{greeting} Dengan modal Rp {b:,} Anda, kita bisa eksplorasi opsi investasi yang sesuai. "
            
            # Strategy Analysis
            if approach == "aggressive":
                response += "Karena Anda mengejar return tinggi, kita fokus pada saham dengan potensi pertumbuhan besar, meski risiko lebih tinggi. "
                target_desc = "return tinggi (di atas rata-rata)"
            elif approach == "conservative":
                response += "Untuk pendekatan aman, kita pilih saham dengan volatilitas rendah agar modal lebih stabil. "
                target_desc = "stabilitas dan risiko rendah"
            else:
                response += "Dengan pendekatan seimbang, kita imbangi antara return dan risiko untuk hasil optimal. "
                target_desc = "keseimbangan return dan risiko"
            response += f"Strategi ini cocok untuk target {target_desc} berdasarkan analisis historis. "
            
            # Stock Recommendations
            response += "Berikut rekomendasi saham yang bisa dijangkau:\n"
            for r in recs:
                company = stock_names.get(r['ticker'], r['ticker'])
                per_share = int(r['last'])
                per_lot = int(r['lot_price'])
                if approach == "aggressive":
                    reason = f"karena memiliki return historis tinggi ({r['return']:.4f}) meski volatilitasnya ({r['sigma']:.4f}) cukup tinggi, cocok untuk target pertumbuhan cepat."
                elif approach == "conservative":
                    reason = f"karena volatilitasnya rendah ({r['sigma']:.4f}), sehingga risiko fluktuasi harga lebih kecil, ideal untuk investasi jangka panjang yang stabil."
                else:
                    reason = f"karena rasio return terhadap risiko ({r['score']:.4f}) optimal, memberikan keseimbangan yang baik antara pertumbuhan dan keamanan."
                response += f"- {r['ticker']} ({company}): harga per lembar Rp {per_share:,}, per lot (100 lembar) Rp {per_lot:,}. {reason}\n"
            
            # Educational Disclaimer
            response += "\nIni simulasi edukasi berdasarkan data historis, bukan nasihat keuangan. Selalu konsultasikan dengan ahli sebelum berinvestasi."
            return response
        return f"Baik, dengan modal Rp {b:,}, pertimbangkan reksa dana atau DCA untuk investasi jangka panjang. Saham satu lot mungkin terlalu mahal, tapi Anda bisa mulai bertahap."
    # direct ask 'saham apa'
    if "saham apa" in ml or "saham yang cocok" in ml:
        if st.session_state.user_profile.get("budget"):
            # call handle_message on budget number to reuse logic
            return handle_message(str(st.session_state.user_profile["budget"]))
        else:
            st.session_state.user_profile["expecting_budget"] = True
            return "Berapa modal Anda (mis. '1,2 juta') supaya saya rekomendasikan saham yang cocok?"
    # ticker explicit
    tmatch = re.search(r'\b([A-Za-z]{2,5}(?:\.JK)?)\b', m)
    if tmatch:
        tok = tmatch.group(1).upper()
        if not tok.endswith('.JK'):
            tok = tok + '.JK'
        s = get_stock_series(tok)
        stats = compute_stats(s)
        if stats is None:
            return f"Maaf, data untuk {tok} tidak cukup untuk analisis saat ini. Coba saham lain atau periksa koneksi internet."
        company = stock_names.get(tok, tok)
        response = f"Untuk {company} ({tok}), harga terakhir adalah Rp {stats['last']:,.0f}. "
        response += f"Volatilitas harian sekitar {stats['sigma']:.4f}, yang menunjukkan tingkat fluktuasi harga. "
        response += f"VaR95 (Cornish-Fisher) adalah {stats['VaR95_CF']:.4f}, sebagai ukuran risiko potensial. "
        response += "Ini adalah analisis edukatif berdasarkan data historis. Untuk investasi nyata, pertimbangkan faktor fundamental perusahaan dan kondisi pasar."
        return response
    # basic help
    if any(x in ml for x in ["halo","hai","selamat","cara mulai","bagaimana mulai"]):
        return ("Halo! Contoh perintah: 'Nama panggilan ...', 'Tujuan: pendidikan jangka panjang', 'saham apa yang cocok', "
                "'1,2 juta' (untuk modal), atau sebutkan ticker mis. 'BBCA'.")
    # fallback
    return "Maaf saya belum paham. Coba: 'saham apa yang cocok dengan modal 1 juta' atau ketik ticker (mis. BBCA)."

# -----------------------------
# PAGES
# -----------------------------
def render_home():
    # header center
    st.markdown("<div class='header'><h1>SinergiVest</h1><p>Pendamping Investasi Inklusif</p></div>", unsafe_allow_html=True)

    # two carousels side-by-side
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("<div class='carousel-box'>", unsafe_allow_html=True)
        stock_idx = st.session_state.stock_idx if "stock_idx" in st.session_state else 0
        ticker = TICKERS[stock_idx]
        company_name = TICKER_NAMES.get(ticker, ticker)
        st.subheader(f"📈 Tren Harga {company_name} ({ticker})")
        series = get_stock_series(ticker)
        try:
            fig = go.Figure(data=[go.Scatter(x=series.index, y=series.values, mode="lines", name=ticker)])
            fig.update_layout(template="plotly_dark", height=300, margin=dict(t=30,l=10,r=10,b=10))
            fig.update_xaxes(type='date', tickformat="%Y-%m-%d")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write("Gagal menampilkan grafik:", e)
        # simple stat
        stats = compute_stats(series)
        if stats:
            st.write(f"Harga terakhir: Rp {stats['last']:,.0f} • Volatilitas: {stats['sigma']:.4f}")
            # Saran berdasarkan perbandingan dengan rata-rata 30 hari
            if len(series) >= 30:
                mean_30 = series.tail(30).mean()
            else:
                mean_30 = series.mean()
            if stats['last'] < mean_30:
                saran = "Buy"
            elif stats['last'] > mean_30 * 1.05:  # Jika lebih dari 5% di atas rata-rata
                saran = "Wait"
            else:
                saran = "Hold"
            st.write(f"Saran: {saran}")
            st.write("*Catatan: Ini analisis sederhana berdasarkan data historis. Bukan nasihat investasi formal. Konsultasikan ahli keuangan.*")
        # navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="btn-oval btn-white">', unsafe_allow_html=True)
            if st.button("⟨ Prev Saham"):
                st.session_state.stock_idx = (st.session_state.stock_idx - 1) % len(TICKERS)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="btn-oval btn-white">', unsafe_allow_html=True)
            if st.button("Next Saham ⟩"):
                st.session_state.stock_idx = (st.session_state.stock_idx + 1) % len(TICKERS)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='carousel-box'>", unsafe_allow_html=True)
        st.subheader("🎥 Literasi & Edukasi (OJK)")
        vid_idx = st.session_state.edu_idx if "edu_idx" in st.session_state else 0
        video = EDU_VIDEOS[vid_idx]
        # use try/except because some embed may not work in offline env
        try:
            st.video(video["url"])
        except:
            st.write("Video tidak dapat dimuat. Silakan buka:", video["url"])
        st.write(f"Topik: {video['title']}")
        st.write("Pelajari literasi keuangan dan investasi yang bijak dari OJK untuk pendidikan inklusif.")
        col1, col2 = st.columns(2)
        if col1.button("⟨ Prev Edu"):
            st.session_state.edu_idx = (st.session_state.edu_idx - 1) % len(EDU_VIDEOS)
        if col2.button("Next Edu ⟩"):
            st.session_state.edu_idx = (st.session_state.edu_idx + 1) % len(EDU_VIDEOS)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # bottom oval buttons centered
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        b1, b2 = st.columns(2)
        with b1:
            st.markdown('<div class="btn-oval btn-green">', unsafe_allow_html=True)
            if st.button("Mulai dengan BotVes"):
                st.session_state.page = "chat"
            st.markdown('</div>', unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="btn-oval btn-blue">', unsafe_allow_html=True)
            if st.button("Masuk / Daftar"):
                st.session_state.page = "login"
            st.markdown('</div>', unsafe_allow_html=True)

def render_login():
    st.markdown("<div style='text-align:center;'><h2>Masuk / Daftar</h2></div>", unsafe_allow_html=True)
    with st.form("login_form"):
        user = st.text_input("Email atau username")
        pw = st.text_input("Password", type="password")
        cap = st.text_input("Captcha (ketik 1234 untuk demo)")
        submitted = st.form_submit_button("Masuk")
        if submitted:
            if not user or not pw:
                st.error("Lengkapi username & password.")
            elif cap.strip() != "1234":
                st.error("Captcha salah (demo: 1234).")
            else:
                st.success("Login demo berhasil.")
                st.session_state.user_profile["name"] = user
                st.session_state.page = "chat"

    if st.button("⟵ Kembali ke Beranda"):
        st.session_state.page = "home"

def render_chat():
    st.markdown("<div style='text-align:center;'><h2>🤖 BotVes — Pendamping Investasi</h2></div>", unsafe_allow_html=True)
    st.write("Catatan: ini fitur edukatif. Jangan gunakan sebagai nasihat formal.")
    # show uploaded file area inside chat
    uploaded = st.file_uploader("Unggah file portofolio (.csv/.xlsx) — opsional", type=['csv','xls','xlsx'])
    if uploaded:
        try:
            if uploaded.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.uploaded_df = df
            st.success(f"File {uploaded.name} disimpan di sesi.")
            st.dataframe(df.head(8))
        except Exception as e:
            st.error("Gagal baca file: " + str(e))

    # initial greeting once
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({"role":"assistant","content":"Halo! Saya BotVes. Sebutkan tujuan atau modal Anda untuk mulai."})

    # display chat bubbles
    for m in st.session_state.chat_history:
        if m["role"] == "user":
            st.markdown(f"<div class='user'>{m['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot'>{m['content']}</div>", unsafe_allow_html=True)

    # chat input: prefer chat_input if available, otherwise use text_input + button
    if hasattr(st, "chat_input"):
        user_msg = st.chat_input("Ketik pesan...")
        if user_msg:
            st.session_state.chat_history.append({"role":"user","content":user_msg})
            reply = handle_message(user_msg)
            st.session_state.chat_history.append({"role":"assistant","content":reply})
    else:
        col_a, col_b = st.columns([8,1])
        with col_a:
            txt = st.text_input("Ketik pesan...", key="chat_text")
        with col_b:
            if st.button("Kirim"):
                if txt and txt.strip():
                    st.session_state.chat_history.append({"role":"user","content":txt})
                    reply = handle_message(txt)
                    st.session_state.chat_history.append({"role":"assistant","content":reply})
                    # reset input
                    st.session_state["chat_text"] = ""

    if st.button("⟵ Kembali ke Beranda"):
        st.session_state.page = "home"

# main render
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "login":
    render_login()
elif st.session_state.page == "chat":
    render_chat()
else:
    st.session_state.page = "home"
    render_home()