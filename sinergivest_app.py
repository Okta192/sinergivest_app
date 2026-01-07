# sinergivest_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

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
.header { 
  text-align: center; 
  margin-bottom: 20px; 
  background: linear-gradient(135deg, #001f3f, #003366); /* Navy Blue gradient for contrast */
  padding: 30px; 
  border-radius: 15px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.3); /* Subtle shadow for depth */
}
.header h1 { 
  font-size: 50px; 
  margin: 0; 
  color: #ffffff; 
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Modern font */
  font-weight: bold;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.5); /* Text shadow for readability */
}
.header p { 
  margin: 10px 0 0 0; 
  color: #e0e0e0; 
  font-size: 18px;
}

/* Carousel boxes */
.carousel-box {
  background: #111217;
  border-radius: 15px; /* Softer rounded corners */
  padding: 20px;
  min-height: 260px;
  color: #fff;
  box-shadow: 0 8px 25px rgba(0,0,0,0.6); /* Enhanced shadow for 'timbul' effect */
  border: 1px solid #333; /* Subtle border */
}

/* Oval buttons */
.btn-oval > button {
  border-radius: 50px !important;
  height: 56px !important;
  font-size: 16px !important;
  font-weight: 600 !important;
  padding: 0 26px !important;
  box-shadow: 0 4px 10px rgba(0,0,0,0.2); /* Soft shadow for all buttons */
  transition: all 0.3s ease; /* Smooth transitions */
}
.btn-green > button { background: #00E676 !important; color: #fff !important; border: none !important; }
.btn-green > button:hover { background: #00C853 !important; box-shadow: 0 6px 15px rgba(0,0,0,0.3); }
.btn-blue > button { background: #ADD8E6 !important; color: #fff !important; border: none !important; }
.btn-blue > button:hover { background: #87CEEB !important; box-shadow: 0 6px 15px rgba(0,0,0,0.3); }
.btn-white > button { background: #ffffff !important; color: #000000 !important; border: 1px solid #cccccc !important; }
.btn-white > button:hover { background: #f0f0f0 !important; box-shadow: 0 6px 15px rgba(0,0,0,0.3); }

/* Tombol khusus */
button.start-bot { 
  background-color: #00E676 !important; /* Traffic Light Green */
  color: white !important; 
  border-radius: 50px !important; 
  border: none !important;
  font-weight: bold !important;
  box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
button.start-bot:hover { 
  background-color: #00C853 !important; /* Darker green on hover */
  box-shadow: 0 6px 15px rgba(0,0,0,0.3);
}

button.login { 
  background-color: #ADD8E6 !important; /* Light Blue */
  color: white !important; 
  border: none !important; 
  border-radius: 50px !important;
  font-weight: bold !important;
  box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
button.login:hover { 
  background-color: #87CEEB !important; /* Darker blue on hover */
  box-shadow: 0 6px 15px rgba(0,0,0,0.3);
}

button.nav { 
  background-color: #a9a9a9 !important; /* Neutral modern gray */
  color: white !important; 
  border: 1px solid #a9a9a9 !important; 
  border-radius: 50px !important;
  font-weight: 600 !important;
  box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}
button.nav:hover { 
  background-color: #808080 !important; /* Darker gray on hover */
  box-shadow: 0 6px 15px rgba(0,0,0,0.3);
}

/* Override for all buttons to ensure oval shape and shadow */
button {
  border-radius: 50px !important;
  box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
}
button:hover {
  box-shadow: 0 6px 15px rgba(0,0,0,0.3) !important;
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
    st.session_state.page = "home"   # home / chat / login / portfolio
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
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []  # list of {"ticker": str, "shares": int, "buy_price": float}
# simple cache for fetched series
if "stock_cache" not in st.session_state:
    st.session_state.stock_cache = {}

# Optimasi memori: Batasi ukuran session state untuk mencegah pembengkakan memori
if len(st.session_state.chat_history) > 50:  # Batasi chat history maksimal 50 pesan
    st.session_state.chat_history = st.session_state.chat_history[-50:]
if len(st.session_state.portfolio) > 20:  # Batasi portofolio maksimal 20 saham
    st.session_state.portfolio = st.session_state.portfolio[-20:]

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
    "BRIS.JK": "Bank Syariah Indonesia",
    # PERTAMBANGAN (Mining)
    "ADRO.JK": "Adaro Energy",
    "ITMG.JK": "Indo Tambangraya Megah",
    "PTBA.JK": "Bukit Asam",
    "ANTM.JK": "Aneka Tambang",
    "TINS.JK": "Timah",
    # OTOMOTIF (Automotive)
    "ASII.JK": "Astra International",
    "AUTO.JK": "Astra Otoparts",
    "IMAS.JK": "Indomobil Sukses Internasional",
    "SMSM.JK": "Selamat Sempurna",
    # ASURANSI (Insurance)
    "ASJT.JK": "Asuransi Jiwa Taspen",
    "AMAG.JK": "Asuransi Multi Artha Guna",
    "ASBI.JK": "Asuransi Bintang",
    "PNIN.JK": "Panin Financial",
    # PERKEBUNAN (Plantation)
    "AALI.JK": "Astra Agro Lestari",
    "LSIP.JK": "PP London Sumatra Indonesia",
    "BWPT.JK": "BW Plantation",
    "DSNG.JK": "Dharma Satya Nusantara",
    # TRANSPORTASI (Transportation)
    "ASSA.JK": "Adi Sarana Armada",
    "BIRD.JK": "Blue Bird",
    "SMDR.JK": "Samudera Indonesia",
    "TMAS.JK": "Temas"
}

# Dictionary sektor ke list ticker untuk rekomendasi berdasarkan sektor
SEKTOR_STOCKS = {
    "pertambangan": ["ADRO.JK", "ITMG.JK", "PTBA.JK", "ANTM.JK", "TINS.JK"],
    "otomotif": ["ASII.JK", "AUTO.JK", "IMAS.JK", "SMSM.JK"],
    "asuransi": ["ASJT.JK", "AMAG.JK", "ASBI.JK", "PNIN.JK"],
    "perkebunan": ["AALI.JK", "LSIP.JK", "BWPT.JK", "DSNG.JK"],
    "transportasi": ["ASSA.JK", "BIRD.JK", "SMDR.JK", "TMAS.JK"],
    "bank": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK"],  # Tambahan untuk konsistensi
    "telekomunikasi": ["TLKM.JK"],
    "konsumer": ["UNTR.JK", "GOTO.JK", "UNVR.JK", "ICBP.JK", "INDF.JK", "AMRT.JK"],
    "farmasi": ["KLBF.JK", "MDKA.JK"],
    "energi": ["PGAS.JK"],
    "infrastruktur": ["CPIN.JK", "JSMR.JK"],
    "semiconductor": ["SMGR.JK"]
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

@st.cache_data  # Optimasi memori: Cache perhitungan statistik untuk menghindari ulang perhitungan
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
# Generate Plotly figure for stock series
# -----------------------------
@st.cache_data  # Optimasi memori: Cache grafik Plotly untuk menghindari pembuatan ulang setiap render
def generate_stock_figure(ticker, series):
    fig = go.Figure(data=[go.Scatter(x=series.index, y=series.values, mode="lines", name=ticker)])
    fig.update_layout(template="plotly_dark", height=300, margin=dict(t=30,l=10,r=10,b=10))
    fig.update_xaxes(type='date', tickformat="%Y-%m-%d")
    return fig
def generate_analysis_text(series, stats, saran):
    if series is None or stats is None:
        return "Data tidak cukup untuk analisis."
    
    # Analisis Tren
    first_price = series.iloc[0]
    last_price = series.iloc[-1]
    change = (last_price - first_price) / first_price * 100
    if change > 5:
        trend = "uptrend"
        trend_desc = f"Harga saham menunjukkan tren naik dengan peningkatan sebesar {change:.2f}% selama periode observasi."
    elif change < -5:
        trend = "downtrend"
        trend_desc = f"Harga saham menunjukkan tren turun dengan penurunan sebesar {abs(change):.2f}% selama periode observasi."
    else:
        trend = "sideways"
        trend_desc = f"Harga saham cenderung stabil atau sideways dengan perubahan sebesar {change:.2f}% selama periode observasi."
    
    # Penjelasan Volatilitas
    sigma = stats['sigma']
    if sigma < 0.02:
        vol_desc = f"Volatilitas harian sebesar {sigma:.4f} tergolong rendah. Ini menunjukkan bahwa harga saham relatif stabil dan risiko fluktuasi harga dalam jangka pendek cukup kecil."
    elif sigma < 0.05:
        vol_desc = f"Volatilitas harian sebesar {sigma:.4f} tergolong sedang. Ini menunjukkan bahwa harga saham memiliki fluktuasi yang moderat, yang umum untuk saham di pasar modal."
    else:
        vol_desc = f"Volatilitas harian sebesar {sigma:.4f} tergolong tinggi. Ini menunjukkan bahwa harga saham sangat fluktuatif, yang dapat menimbulkan risiko investasi yang lebih besar."
    
    # Interpretasi Titik Tertinggi/Terendah
    max_price = series.max()
    min_price = series.min()
    max_desc = f"Harga tertinggi dalam periode ini adalah Rp {max_price:,.0f}, yang mungkin menunjukkan momentum positif atau reaksi pasar terhadap berita baik."
    min_desc = f"Harga terendah adalah Rp {min_price:,.0f}, yang bisa menjadi area support potensial atau indikasi tekanan jual yang kuat."
    
    # Rationale untuk Saran
    if saran == "Buy":
        rationale = "Saran 'Buy' diberikan karena harga saat ini berada di bawah rata-rata 30 hari terakhir, menunjukkan potensi undervalued. Dengan volatilitas yang stabil, ini bisa menjadi kesempatan untuk entry dengan risiko yang terkendali."
    elif saran == "Hold":
        rationale = "Saran 'Hold' diberikan karena harga saat ini berada dalam kisaran normal relatif terhadap rata-rata 30 hari, tanpa indikasi overvalued yang signifikan. Ini cocok untuk investor yang sudah memiliki posisi."
    elif saran == "Wait":
        rationale = "Saran 'Wait' diberikan karena harga saat ini lebih dari 5% di atas rata-rata 30 hari, menunjukkan potensi overvalued. Disarankan menunggu koreksi harga sebelum mempertimbangkan pembelian."
    else:
        rationale = "Saran tidak dapat ditentukan dengan data yang tersedia."
    
    # Gabungkan narasi
    analysis = f"""
    **Analisis Tren:** {trend_desc}
    
    **Penjelasan Volatilitas:** {vol_desc}
    
    **Interpretasi Titik Ekstrem:** {max_desc} {min_desc} Implikasinya bagi investor adalah penting untuk memantau level-level ini sebagai indikator sentimen pasar.
    
    **Rationale untuk Saran '{saran}':** {rationale}
    
    *Analisis ini berdasarkan data historis dan metode statistik dasar seperti log-return dan standar deviasi. Bukan nasihat investasi profesional; gunakan sebagai referensi pendidikan.*
    """
    return analysis

# -----------------------------
# Fungsi untuk Deep Personalization: Korelasi Dana Kesehatan vs Tabungan Investasi
# -----------------------------
def health_investment_correlation(risk_pref, budget):
    # Saran dana darurat kesehatan berdasarkan profil risiko
    if risk_pref == "conservative":
        health_ratio = 0.3  # 30% dari budget untuk dana kesehatan
        advice = "Dengan pendekatan konservatif, prioritaskan dana kesehatan yang lebih besar untuk mengurangi risiko kesehatan yang dapat mengganggu investasi."
    elif risk_pref == "aggressive":
        health_ratio = 0.1  # 10% dari budget
        advice = "Untuk pendekatan agresif, alokasikan dana kesehatan minimal, tapi pastikan ada buffer untuk risiko kesehatan yang tidak terduga."
    else:
        health_ratio = 0.2  # 20% dari budget
        advice = "Pendekatan seimbang: Alokasikan dana kesehatan yang cukup untuk keseimbangan antara kesehatan dan investasi."
    
    health_fund = int(budget * health_ratio)
    investment_fund = budget - health_fund
    correlation_note = "Korelasi antara dana kesehatan dan tabungan investasi: Semakin besar dana kesehatan, semakin stabil investasi karena risiko kesehatan berkurang, memungkinkan fokus pada pertumbuhan aset."
    
    return f"Dana kesehatan yang disarankan: Rp {health_fund:,} ({health_ratio*100:.0f}% dari modal). Dana investasi tersisa: Rp {investment_fund:,}. {advice} {correlation_note}"

# -----------------------------
# Fungsi untuk Simulasi Portofolio
# -----------------------------
@st.cache_data  # Optimasi memori: Cache simulasi portofolio berdasarkan parameter untuk menghindari perhitungan ulang
def simulate_portfolio_growth(portfolio, target_return, years):
    # target_return: annual return, e.g., 0.1 for 10%
    total_value = 0
    growth_data = []
    for item in portfolio:
        ticker = item["ticker"]
        shares = item["shares"]
        buy_price = item["buy_price"]
        current_price = get_stock_series(ticker).iloc[-1] if get_stock_series(ticker) is not None else buy_price
        current_value = shares * current_price
        total_value += current_value
        # Simple growth simulation: compound annually
        future_value = current_value * (1 + target_return) ** years
        growth_data.append({
            "ticker": ticker,
            "current_value": current_value,
            "future_value": future_value,
            "growth": (future_value - current_value) / current_value * 100
        })
    
    return total_value, growth_data

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

def parse_sektor(text):
    ml = text.lower()
    sektor_keywords = {
        "pertambangan": ["tambang", "mining", "batubara", "emas", "nikel"],
        "otomotif": ["otomotif", "mobil", "motor", "auto"],
        "asuransi": ["asuransi", "insurance"],
        "perkebunan": ["perkebunan", "sawit", "kelapa", "plantation"],
        "transportasi": ["transportasi", "logistik", "kapal", "transport"],
        "bank": ["bank", "banking"],
        "telekomunikasi": ["telekom", "telco", "internet"],
        "konsumer": ["konsumer", "consumer", "makanan", "minuman"],
        "farmasi": ["farmasi", "obat", "pharma"],
        "energi": ["energi", "gas", "minyak"],
        "infrastruktur": ["infrastruktur", "jalan", "tol"],
        "semiconductor": ["semikonduktor", "chip"]
    }
    for sektor, keywords in sektor_keywords.items():
        if any(kw in ml for kw in keywords):
            return sektor
    return None

@st.cache_data  # Optimasi memori: Cache rekomendasi berdasarkan budget dan approach untuk menghindari perhitungan ulang
def recommend_for_budget(budget, approach="balanced", top_n=3, sektor=None):
    # Jika sektor ditentukan, gunakan ticker dari sektor tersebut, else gunakan TICKERS
    tickers_to_use = SEKTOR_STOCKS.get(sektor.lower(), TICKERS) if sektor else TICKERS
    cand = []
    for t in tickers_to_use:
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
        sektor = parse_sektor(m)  # Extract sektor from message
        st.session_state.user_profile["risk_pref"] = approach
        recs = recommend_for_budget(b, approach=approach, top_n=3, sektor=sektor)
        if recs:
            # Greeting & Confirmation
            greetings = [
                "Halo! Menarik sekali minat investasinya.",
                "Hai! Bagus, kita mulai dengan modal yang jelas.",
                "Salam! Siap membantu dengan dana tersebut.",
                "Wow, semangat investasinya luar biasa!"
            ]
            greeting = np.random.choice(greetings)
            # Validasi eksplisit: konfirmasi nominal dalam format Rp lengkap
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
            
            # Rekomendasi saham
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
            
            # Tambahkan saran kesehatan jika relevan
            if "kesehatan" in ml or "dana darurat" in ml:
                health_advice = health_investment_correlation(approach, b)
                response += f"\n{health_advice}"
            
            # Educational Disclaimer
            response += "\nIni simulasi edukasi berdasarkan data historis, bukan nasihat keuangan. Selalu konsultasikan dengan ahli sebelum berinvestasi."
            return response
        return f"Baik, dengan modal Rp {b:,}, pertimbangkan reksa dana atau DCA untuk investasi jangka panjang. Saham satu lot mungkin terlalu mahal, tapi Anda bisa mulai bertahap."
    # direct ask 'saham apa'
    if "saham apa" in ml or "saham yang cocok" in ml:
        if st.session_state.user_profile.get("budget"):
            sektor = parse_sektor(m)
            approach = st.session_state.user_profile.get("risk_pref", "balanced")
            recs = recommend_for_budget(st.session_state.user_profile["budget"], approach=approach, top_n=3, sektor=sektor)
            if recs:
                response = "Berikut rekomendasi saham yang bisa dijangkau:\n"
                for r in recs:
                    company = stock_names.get(r['ticker'], r['ticker'])
                    per_share = int(r['last'])
                    per_lot = int(r['lot_price'])
                    response += f"- {r['ticker']} ({company}): harga per lembar Rp {per_share:,}, per lot (100 lembar) Rp {per_lot:,}.\n"
                response += "\nIni simulasi edukasi berdasarkan data historis, bukan nasihat keuangan."
                return response
            else:
                return "Tidak ada saham yang cocok dengan modal Anda di sektor tersebut."
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
    # Deep Personalization: kesehatan
    if "kesehatan" in ml or "dana darurat" in ml:
        if st.session_state.user_profile.get("budget") and st.session_state.user_profile.get("risk_pref"):
            b = st.session_state.user_profile["budget"]
            approach = st.session_state.user_profile["risk_pref"]
            return health_investment_correlation(approach, b)
        else:
            return "Untuk saran dana kesehatan, sebutkan modal dan preferensi risiko Anda terlebih dahulu (mis. '1 juta konservatif')."
    # basic help
    if any(x in ml for x in ["halo","hai","selamat","cara mulai","bagaimana mulai"]):
        return ("Halo! Contoh perintah: 'Nama panggilan ...', 'Tujuan: pendidikan jangka panjang', 'saham apa yang cocok', '1,2 juta' (untuk modal), atau sebutkan ticker mis. 'BBCA'. Untuk kesehatan: 'dana kesehatan'.")
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
        st.subheader(f" Tren Harga {company_name} ({ticker})")
        series = get_stock_series(ticker)
        try:
            fig = generate_stock_figure(ticker, series)
            # Optimasi memori: Nonaktifkan modebar dan fitur interaktif lainnya untuk menghemat memori browser
            config = {"displayModeBar": False, "displaylogo": False, "modeBarButtonsToRemove": ["pan2d", "select2d", "lasso2d", "resetScale2d"]}
            st.plotly_chart(fig, use_container_width=True, config=config)
        except Exception as e:
            st.write("Gagal menampilkan grafik:", e)
        # simple stat
        stats = compute_stats(series)
        if stats:
            st.write(f"Harga terakhir: Rp {stats['last']:,.0f}  Volatilitas: {stats['sigma']:.4f}")
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
            # Analisis Deskriptif & Statistik
            with st.expander("Lihat Analisis Deskriptif & Statistik Mendalam"):
                analysis = generate_analysis_text(series, stats, saran)
                st.write(analysis)
        # navigation buttons
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="btn-oval btn-white">', unsafe_allow_html=True)
            if st.button(" Prev Saham"):
                st.session_state.stock_idx = (st.session_state.stock_idx - 1) % len(TICKERS)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="btn-oval btn-white">', unsafe_allow_html=True)
            if st.button("Next Saham "):
                st.session_state.stock_idx = (st.session_state.stock_idx + 1) % len(TICKERS)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='carousel-box'>", unsafe_allow_html=True)
        st.subheader(" Literasi & Edukasi (OJK)")
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
        if col1.button(" Prev Edu"):
            st.session_state.edu_idx = (st.session_state.edu_idx - 1) % len(EDU_VIDEOS)
        if col2.button("Next Edu "):
            st.session_state.edu_idx = (st.session_state.edu_idx + 1) % len(EDU_VIDEOS)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # bottom oval buttons centered
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        b1, b2, b3 = st.columns(3)
        with b1:
            with st.container():
                st.markdown('<div class="btn-green">', unsafe_allow_html=True)
                if st.button("Mulai dengan BotVes"):
                    st.session_state.page = "chat"
                st.markdown('</div>', unsafe_allow_html=True)
        with b2:
            st.markdown('<div class="btn-oval btn-blue">', unsafe_allow_html=True)
            if st.button("Masuk / Daftar"):
                st.session_state.page = "login"
            st.markdown('</div>', unsafe_allow_html=True)
        with b3:
            st.markdown('<div class="btn-oval btn-white">', unsafe_allow_html=True)
            if st.button("Simulasi Portofolio"):
                st.session_state.page = "portfolio"
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

    if st.button(" Kembali ke Beranda"):
        st.session_state.page = "home"

def render_portfolio():
    st.markdown("<div style='text-align:center;'><h2>Simulasi Portofolio</h2></div>", unsafe_allow_html=True)
    st.write("Simulasi pertumbuhan nilai aset berdasarkan target return tahunan.")
    
    # Tambah saham ke portofolio
    with st.form("add_stock"):
        ticker = st.selectbox("Pilih Saham", options=list(stock_names.keys()), format_func=lambda x: f"{x} - {stock_names[x]}")
        shares = st.number_input("Jumlah Saham", min_value=1, value=100)
        buy_price = st.number_input("Harga Beli per Saham (Rp)", min_value=1.0, value=1000.0)
        submitted = st.form_submit_button("Tambah ke Portofolio")
        if submitted:
            st.session_state.portfolio.append({"ticker": ticker, "shares": shares, "buy_price": buy_price})
            st.success(f"Ditambahkan: {ticker} ({shares} saham @ Rp {buy_price:,.0f})")
    
    # Tampilkan portofolio
    if st.session_state.portfolio:
        st.subheader("Portofolio Anda")
        df_port = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(df_port)
        
        # Simulasi
        target_return = st.slider("Target Return Tahunan (%)", 0, 50, 10) / 100
        years = st.slider("Jangka Waktu (Tahun)", 1, 10, 5)
        if st.button("Simulasi Pertumbuhan"):
            total_value, growth_data = simulate_portfolio_growth(st.session_state.portfolio, target_return, years)
            st.write(f"Total Nilai Saat Ini: Rp {total_value:,.0f}")
            df_growth = pd.DataFrame(growth_data)
            st.dataframe(df_growth)
            
            # Grafik interaktif dengan Plotly
            fig = px.bar(df_growth, x="ticker", y="future_value", title=f"Estimasi Nilai Portofolio dalam {years} Tahun", labels={"future_value": "Nilai Masa Depan (Rp)", "ticker": "Saham"})
            fig.update_traces(hovertemplate="Saham: %{x}<br>Nilai: Rp %{y:,.0f}")
            # Optimasi memori: Nonaktifkan modebar untuk menghemat memori browser
            config = {"displayModeBar": False, "displaylogo": False}
            st.plotly_chart(fig, use_container_width=True, config=config)
    else:
        st.write("Portofolio kosong. Tambahkan saham terlebih dahulu.")
    
    if st.button(" Kembali ke Beranda"):
        st.session_state.page = "home"

def render_chat():
    st.markdown("<div style='text-align:center;'><h2> BotVes  Pendamping Investasi</h2></div>", unsafe_allow_html=True)
    st.write("Catatan: ini fitur edukatif. Jangan gunakan sebagai nasihat formal.")
    # show uploaded file area inside chat
    uploaded = st.file_uploader("Unggah file portofolio (.csv/.xlsx)  opsional", type=['csv','xls','xlsx'])
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

    if st.button(" Kembali ke Beranda"):
        st.session_state.page = "home"

# main render
if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "login":
    render_login()
elif st.session_state.page == "portfolio":
    render_portfolio()
elif st.session_state.page == "chat":
    render_chat()
else:
    st.session_state.page = "home"
    render_home()
