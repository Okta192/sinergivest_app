from __future__ import annotations

import math
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf

    HAS_YFINANCE = True
except Exception:
    yf = None
    HAS_YFINANCE = False

try:
    import google.generativeai as genai

    HAS_GEMINI = True
except Exception:
    genai = None
    HAS_GEMINI = False

try:
    from scipy.stats import kurtosis, norm, skew

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


st.set_page_config(
    page_title="SinergiVest",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@dataclass(frozen=True)
class GreenMSME:
    name: str
    sector: str
    city: str
    province: str
    funding_need: int
    target_yield: float
    tenor_months: int
    environmental_score: int
    social_score: int
    governance_score: int
    impact: str
    use_of_funds: str

    @property
    def location(self) -> str:
        return f"{self.city}, {self.province}"

    @property
    def esg_score(self) -> int:
        return round(
            self.environmental_score * 0.4
            + self.social_score * 0.3
            + self.governance_score * 0.3
        )

    @property
    def risk_label(self) -> str:
        if self.esg_score >= 85:
            return "Rendah"
        if self.esg_score >= 76:
            return "Moderat"
        return "Perlu Pendampingan"


UMKM_PARTNERS: list[GreenMSME] = [
    GreenMSME(
        name="Kopi Lestari Arunika",
        sector="Agrikultur Regeneratif",
        city="Temanggung",
        province="Jawa Tengah",
        funding_need=180_000_000,
        target_yield=11.5,
        tenor_months=18,
        environmental_score=89,
        social_score=84,
        governance_score=78,
        impact="Mengolah limbah kulit kopi menjadi kompos dan briket biomassa untuk menekan residu panen.",
        use_of_funds="Mesin pengering hemat energi, perluasan rumah kompos, dan digitalisasi pembukuan.",
    ),
    GreenMSME(
        name="Batik Tirta Warna",
        sector="Fesyen Sirkular",
        city="Pekalongan",
        province="Jawa Tengah",
        funding_need=125_000_000,
        target_yield=10.2,
        tenor_months=12,
        environmental_score=82,
        social_score=88,
        governance_score=80,
        impact="Menggunakan pewarna alami, filtrasi air mikro, dan pelatihan perajin perempuan.",
        use_of_funds="IPAL mikro, stok pewarna nabati, dan sertifikasi produksi bersih.",
    ),
    GreenMSME(
        name="Surya Desa Mandiri",
        sector="Energi Terbarukan",
        city="Lombok Timur",
        province="NTB",
        funding_need=260_000_000,
        target_yield=12.8,
        tenor_months=24,
        environmental_score=93,
        social_score=81,
        governance_score=86,
        impact="Panel surya komunal untuk cold storage nelayan dan warung desa.",
        use_of_funds="Panel surya, baterai, inverter, dan sistem monitoring energi.",
    ),
    GreenMSME(
        name="DaurPack Nusantara",
        sector="Kemasan Daur Ulang",
        city="Bandung",
        province="Jawa Barat",
        funding_need=95_000_000,
        target_yield=9.6,
        tenor_months=10,
        environmental_score=77,
        social_score=79,
        governance_score=74,
        impact="Produksi kemasan dari kertas pascakonsumsi untuk pelaku kuliner lokal.",
        use_of_funds="Cetakan baru, bahan baku daur ulang, dan sistem akuntansi sederhana.",
    ),
]

SEKTOR_SAHAM: dict[str, list[str]] = {
    "1_Keuangan": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK"],
    "2_Pertambangan_Energi": ["ADRO.JK", "PTBA.JK", "ANTM.JK", "PGAS.JK", "ITMG.JK", "TINS.JK"],
    "3_Kesehatan": ["KLBF.JK", "MDKA.JK"],
    "4_Infrastruktur": ["JSMR.JK", "TLKM.JK"],
    "5_Barang_Baku": ["SMGR.JK", "TINS.JK"],
    "6_Konsumen_Primer": ["INDF.JK", "ICBP.JK", "AMRT.JK"],
    "7_Konsumen_Non_Primer": ["ASII.JK", "GOTO.JK", "AUTO.JK"],
    "8_Properti": ["BSDE.JK", "PWON.JK"],
    "9_Teknologi": ["EMTK.JK", "MPPA.JK"],
    "10_Perindustrian": ["UNTR.JK", "CPIN.JK"],
    "11_Transportasi_Logistik": ["BIRD.JK", "ASSA.JK", "SMDR.JK", "TMAS.JK"],
}

STOCK_NAMES: dict[str, str] = {
    "ADRO.JK": "Adaro Energy Indonesia",
    "AMRT.JK": "Sumber Alfaria Trijaya",
    "ANTM.JK": "Aneka Tambang",
    "ASII.JK": "Astra International",
    "ASSA.JK": "Adi Sarana Armada",
    "AUTO.JK": "Astra Otoparts",
    "BBCA.JK": "Bank Central Asia",
    "BBNI.JK": "Bank Negara Indonesia",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "BIRD.JK": "Blue Bird",
    "BMRI.JK": "Bank Mandiri",
    "BRIS.JK": "Bank Syariah Indonesia",
    "BSDE.JK": "Bumi Serpong Damai",
    "CPIN.JK": "Charoen Pokphand Indonesia",
    "EMTK.JK": "Elang Mahkota Teknologi",
    "GOTO.JK": "GoTo Gojek Tokopedia",
    "ICBP.JK": "Indofood CBP Sukses Makmur",
    "INDF.JK": "Indofood Sukses Makmur",
    "ITMG.JK": "Indo Tambangraya Megah",
    "JSMR.JK": "Jasa Marga",
    "KLBF.JK": "Kalbe Farma",
    "MDKA.JK": "Merdeka Copper Gold",
    "MPPA.JK": "Matahari Putra Prima",
    "PGAS.JK": "Perusahaan Gas Negara",
    "PTBA.JK": "Bukit Asam",
    "PWON.JK": "Pakuwon Jati",
    "SMDR.JK": "Samudera Indonesia",
    "SMGR.JK": "Semen Indonesia",
    "TLKM.JK": "Telkom Indonesia",
    "TINS.JK": "Timah",
    "TMAS.JK": "Temas",
    "UNTR.JK": "United Tractors",
}

ALL_TICKERS: list[str] = list(dict.fromkeys(ticker for tickers in SEKTOR_SAHAM.values() for ticker in tickers))

OJK_VIDEOS: list[dict[str, str]] = [
    {
        "title": "Literasi Keuangan Digital untuk Masyarakat",
        "url": "https://www.youtube.com/watch?v=GFgbxJyCSmE",
    },
    {
        "title": "Cerdas Mengelola Keuangan dan Investasi",
        "url": "https://www.youtube.com/watch?v=Is3BfJN3bp0",
    },
    {
        "title": "Edukasi Investasi dan Perlindungan Konsumen",
        "url": "https://www.youtube.com/watch?v=FcMG-ZMQP1g",
    },
]


def rupiah(value: float | int) -> str:
    return f"Rp {value:,.0f}".replace(",", ".")


def percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def default_chat_history() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                "Halo, saya BotVes. Ceritakan modal, sektor yang diminati, dan profil risiko Anda. "
                "Contoh: 'modal 2 juta konservatif sektor bank' atau 'bagaimana alokasi dana kesehatan?'"
            ),
        }
    ]


def init_state() -> None:
    defaults: dict[str, Any] = {
        "page": "home",
        "startup_done": False,
        "stock_idx": 0,
        "edu_idx": 0,
        "umkm_idx": 0,
        "chat_history": default_chat_history(),
        "uploaded_df": None,
        "portfolio": [],
        "user_profile": {
            "name": None,
            "budget": None,
            "risk_pref": "balanced",
            "sector": None,
        },
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    st.session_state.chat_history = st.session_state.chat_history[-30:]
    st.session_state.portfolio = st.session_state.portfolio[-30:]


def startup_loader() -> None:
    if not st.session_state.startup_done:
        with st.spinner("⏳ Menyinkronkan Data Saham & Geolocation UMKM..."):
            time.sleep(1.35)
        st.session_state.startup_done = True


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --sv-bg: #071324;
            --sv-panel: #0d1d33;
            --sv-panel-2: #10243f;
            --sv-line: rgba(151, 177, 214, 0.24);
            --sv-text: #eef6ff;
            --sv-muted: #9eb2cc;
            --sv-green: #1edc8d;
            --sv-cyan: #38bdf8;
            --sv-gold: #f5c451;
            --sv-rose: #fb7185;
            --sv-shadow: 0 18px 42px rgba(0, 0, 0, 0.36), inset 0 1px 0 rgba(255, 255, 255, 0.04);
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: var(--sv-bg) !important;
            color: var(--sv-text) !important;
        }

        [data-testid="stSidebar"] {
            background: #08182b !important;
            border-right: 1px solid var(--sv-line);
        }

        .main .block-container {
            max-width: 1500px;
            padding: 1.05rem 2rem 1.4rem;
        }

        div[data-testid="stVerticalBlock"] {
            gap: 0.78rem;
        }

        div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
            gap: 0.62rem;
        }

        h1, h2, h3, h4, p, span, label, div {
            letter-spacing: 0;
        }

        h1, h2, h3 {
            color: var(--sv-text) !important;
        }

        .sv-hero, .sv-card, .sv-metric, .sv-chat-user, .sv-chat-bot {
            border: 1px solid var(--sv-line);
            background:
                linear-gradient(145deg, rgba(255,255,255,0.035), rgba(255,255,255,0.008)),
                var(--sv-panel);
            box-shadow: var(--sv-shadow);
            border-radius: 8px;
        }

        .sv-hero {
            padding: 1.25rem 1.35rem;
            position: relative;
            overflow: hidden;
        }

        .sv-hero h1 {
            margin: 0.25rem 0 0.35rem;
            font-size: clamp(2rem, 3.8vw, 3.6rem);
            line-height: 1.02;
        }

        .sv-hero p {
            color: var(--sv-muted);
            font-size: 1.02rem;
            margin: 0;
            max-width: 980px;
        }

        .sv-badge, .sv-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid rgba(56, 189, 248, 0.28);
            background: rgba(56, 189, 248, 0.1);
            border-radius: 999px;
            color: #c8ecff;
            font-size: 0.86rem;
            font-weight: 700;
            padding: 0.24rem 0.72rem;
        }

        .sv-card {
            padding: 0.92rem 1rem;
            min-height: 100%;
        }

        .sv-card h3, .sv-card h4 {
            margin-top: 0.2rem;
            margin-bottom: 0.35rem;
        }

        .sv-muted {
            color: var(--sv-muted);
        }

        .sv-grid-note {
            color: var(--sv-muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .sv-metric {
            padding: 0.88rem 0.95rem;
            min-height: 116px;
        }

        .sv-metric small {
            color: var(--sv-muted);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.72rem;
        }

        .sv-metric strong {
            display: block;
            margin-top: 0.28rem;
            color: var(--sv-text);
            font-size: clamp(1.45rem, 2.2vw, 2.05rem);
            line-height: 1.05;
        }

        .sv-metric span {
            display: block;
            margin-top: 0.32rem;
            color: var(--sv-muted);
            font-size: 0.86rem;
        }

        .sv-chat-user, .sv-chat-bot {
            padding: 0.75rem 0.9rem;
            margin: 0.48rem 0;
            line-height: 1.5;
        }

        .sv-chat-user {
            margin-left: 18%;
            background: rgba(56, 189, 248, 0.16);
        }

        .sv-chat-bot {
            margin-right: 18%;
            background: rgba(30, 220, 141, 0.11);
        }

        div[data-testid="stMetric"], div[data-testid="stExpander"], div[data-testid="stDataFrame"],
        div[data-testid="stFileUploader"], div[data-testid="stForm"] {
            border-radius: 8px;
            box-shadow: var(--sv-shadow);
        }

        div[data-testid="stMetric"] {
            background: var(--sv-panel);
            border: 1px solid var(--sv-line);
            padding: 0.75rem;
        }

        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            border-radius: 50px !important;
            min-height: 2.55rem;
            border: 1px solid rgba(151,177,214,0.35);
            background: linear-gradient(135deg, #14375e, #0e2746);
            color: var(--sv-text);
            font-weight: 750;
            box-shadow: 0 8px 20px rgba(0,0,0,0.28);
            transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
        }

        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            transform: translateY(-1px);
            border-color: rgba(30,220,141,0.72);
            box-shadow: 0 14px 26px rgba(0,0,0,0.36);
            color: #ffffff;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--sv-line);
            color: var(--sv-text);
        }

        input, textarea, [data-baseweb="select"] {
            border-radius: 8px !important;
        }

        hr {
            border-color: var(--sv-line);
            margin: 0.6rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_heading(text: str) -> None:
    st.markdown(f"### {text}")


@st.cache_data(show_spinner=False)
def detect_user_location() -> dict[str, str]:
    fallback = {
        "status": "fallback",
        "city": "Semarang",
        "regionName": "Jawa Tengah",
        "country": "Indonesia",
        "query": "-",
    }
    try:
        with urllib.request.urlopen("http://ip-api.com/json/?fields=status,country,regionName,city,query", timeout=2.8) as response:
            data = response.read().decode("utf-8")
        parsed = json.loads(data)
        if parsed.get("status") == "success":
            return {
                "status": "success",
                "city": parsed.get("city") or fallback["city"],
                "regionName": parsed.get("regionName") or fallback["regionName"],
                "country": parsed.get("country") or fallback["country"],
                "query": parsed.get("query") or "-",
            }
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        pass
    return fallback


def apply_geolocation_priority() -> None:
    if st.session_state.get("geo_priority_applied"):
        return
    location = detect_user_location()
    province = (location.get("regionName") or "").lower()
    if province == "jawa tengah":
        st.session_state.umkm_idx = 0
    else:
        location = {
            "status": "fallback",
            "city": "Semarang",
            "regionName": "Jawa Tengah",
            "country": "Indonesia",
            "query": location.get("query", "-"),
        }
        st.session_state.umkm_idx = 0
    st.session_state.geo_location = location
    st.session_state.geo_priority_applied = True


def ordered_umkm_partners() -> list[GreenMSME]:
    location = st.session_state.get("geo_location", {"regionName": "Jawa Tengah"})
    if (location.get("regionName") or "").lower() == "jawa tengah":
        priority = {"Kopi Lestari Arunika", "Batik Tirta Warna"}
        return sorted(UMKM_PARTNERS, key=lambda item: (item.name not in priority, item.name))
    return UMKM_PARTNERS


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_stock_close(ticker: str, period: str = "1y") -> pd.Series:
    ticker = ticker.upper().strip()
    if HAS_YFINANCE and yf is not None:
        try:
            frame = yf.download(
                ticker,
                period=period,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            close = extract_close_series(frame, ticker)
            if close is not None and len(close) >= 5:
                return close.astype(float)
        except Exception:
            pass
    return synthetic_stock_series(ticker)


def extract_close_series(frame: pd.DataFrame | None, ticker: str) -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    if isinstance(frame.columns, pd.MultiIndex):
        candidates = [
            ("Close", ticker),
            ("Adj Close", ticker),
            (ticker, "Close"),
            (ticker, "Adj Close"),
        ]
        for column in candidates:
            if column in frame.columns:
                series = frame[column]
                break
        else:
            try:
                close_frame = frame.xs("Close", axis=1, level=0)
                series = close_frame.iloc[:, 0]
            except Exception:
                try:
                    close_frame = frame.xs("Close", axis=1, level=-1)
                    series = close_frame.iloc[:, 0]
                except Exception:
                    return None
    elif "Close" in frame.columns:
        series = frame["Close"]
    elif "Adj Close" in frame.columns:
        series = frame["Adj Close"]
    else:
        return None
    series = pd.Series(series).dropna()
    series.index = pd.to_datetime(series.index)
    return series


def synthetic_stock_series(ticker: str, days: int = 252) -> pd.Series:
    seed = abs(hash(ticker)) % 2**32
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    base = 900 + (seed % 38) * 115
    drift = 0.00025 + ((seed % 7) - 3) * 0.00005
    shocks = rng.normal(drift, 0.018, len(dates))
    prices = base * np.exp(np.cumsum(shocks))
    return pd.Series(prices, index=dates, name="Close")


@st.cache_data(show_spinner=False)
def compute_stock_stats(values: tuple[float, ...]) -> dict[str, float] | None:
    series = pd.Series(values, dtype=float).dropna()
    if len(series) < 5:
        return None
    log_returns = np.log(series / series.shift(1)).dropna()
    if log_returns.empty:
        return None

    mu = float(log_returns.mean())
    sigma = float(log_returns.std(ddof=0))
    if HAS_SCIPY:
        skewness = float(skew(log_returns))
        excess_kurt = float(kurtosis(log_returns, fisher=True))
        z95 = float(norm.ppf(0.95))
    else:
        skewness = float(log_returns.skew())
        excess_kurt = float(log_returns.kurt())
        z95 = 1.6448536269514722

    z_cf = (
        z95
        + (1 / 6) * (z95**2 - 1) * skewness
        + (1 / 24) * (z95**3 - 3 * z95) * excess_kurt
        - (1 / 36) * (2 * z95**3 - 5 * z95) * (skewness**2)
    )
    var95_cf = max(0.0, -(mu - sigma * z_cf))
    ma30 = float(series.tail(min(30, len(series))).mean())
    last = float(series.iloc[-1])
    first = float(series.iloc[0])
    total_return = (last / first) - 1 if first else 0.0
    advice = stock_advice(last, ma30)
    return {
        "last": last,
        "first": first,
        "ma30": ma30,
        "mu": mu,
        "sigma": sigma,
        "skewness": skewness,
        "excess_kurtosis": excess_kurt,
        "var95_cf": var95_cf,
        "total_return": total_return,
        "advice": advice,
    }


def stock_advice(last_price: float, ma30: float) -> str:
    if ma30 <= 0:
        return "Hold"
    if last_price < ma30 * 0.98:
        return "Buy"
    if last_price > ma30 * 1.05:
        return "Wait"
    return "Hold"


def stock_advice_narrative(ticker: str, stats: dict[str, float]) -> str:
    company = STOCK_NAMES.get(ticker, ticker)
    advice = str(stats["advice"])
    if advice == "Buy":
        rationale = "harga terakhir berada di bawah MA30, sehingga layak masuk watchlist akumulasi bertahap."
    elif advice == "Wait":
        rationale = "harga terakhir sudah cukup jauh di atas MA30, sehingga lebih bijak menunggu koreksi atau konfirmasi tren."
    else:
        rationale = "harga masih dekat dengan MA30, sehingga posisi yang sudah ada dapat dipantau sambil menunggu sinyal lebih jelas."
    return (
        f"{company} ({ticker}) mencatat return 1 tahun sekitar {percent(stats['total_return'])}, "
        f"volatilitas harian {percent(stats['sigma'])}, dan Cornish-Fisher VaR 95% harian sekitar "
        f"{percent(stats['var95_cf'])}. Saran edukatif: **{advice}**, karena {rationale}"
    )


def stock_line_chart(ticker: str, series: pd.Series) -> go.Figure:
    stats = compute_stock_stats(tuple(series.astype(float).round(6).tolist()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=ticker,
            line=dict(color="#38bdf8", width=2.5),
            hovertemplate="%{x|%d %b %Y}<br>Close: Rp %{y:,.0f}<extra></extra>",
        )
    )
    if stats:
        ma = series.rolling(30, min_periods=5).mean()
        fig.add_trace(
            go.Scatter(
                x=ma.index,
                y=ma.values,
                mode="lines",
                name="MA30",
                line=dict(color="#1edc8d", width=2, dash="dot"),
                hovertemplate="%{x|%d %b %Y}<br>MA30: Rp %{y:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(t=18, b=18, l=8, r=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,12,24,0.55)",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(title=None, gridcolor="rgba(151,177,214,0.12)"),
        yaxis=dict(title=None, gridcolor="rgba(151,177,214,0.12)"),
    )
    return fig


def esg_gauge(score: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 34, "color": "#eef6ff"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9eb2cc"},
                "bar": {"color": "#1edc8d"},
                "bgcolor": "rgba(255,255,255,0.04)",
                "bordercolor": "rgba(151,177,214,0.2)",
                "steps": [
                    {"range": [0, 60], "color": "rgba(251,113,133,0.18)"},
                    {"range": [60, 80], "color": "rgba(245,196,81,0.2)"},
                    {"range": [80, 100], "color": "rgba(30,220,141,0.18)"},
                ],
                "threshold": {"line": {"color": "#f5c451", "width": 4}, "thickness": 0.75, "value": 80},
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=235,
        margin=dict(t=8, b=8, l=8, r=8),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def esg_breakdown_chart(umkm: GreenMSME) -> go.Figure:
    labels = ["Lingkungan", "Sosial", "Tata Kelola"]
    values = [umkm.environmental_score, umkm.social_score, umkm.governance_score]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=["#1edc8d", "#38bdf8", "#f5c451"],
            text=[f"{value}/100" for value in values],
            textposition="auto",
            hovertemplate="%{y}: %{x}/100<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(t=8, b=8, l=8, r=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,12,24,0.55)",
        xaxis=dict(range=[0, 100], title=None, gridcolor="rgba(151,177,214,0.12)"),
        yaxis=dict(title=None, autorange="reversed"),
        showlegend=False,
    )
    return fig


@st.cache_data(show_spinner=False)
def build_funding_timeseries(months: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(20260720)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=months, freq="MS")
    funding = np.clip(np.linspace(38, 126, months) + rng.normal(0, 4.8, months).cumsum(), 25, None)
    umkm_growth = np.clip(np.linspace(16, 82, months) + rng.normal(0, 3.2, months), 12, None)
    return pd.DataFrame(
        {
            "bulan": dates,
            "pendanaan_miliar": funding.round(1),
            "umkm_terdanai": umkm_growth.round().astype(int),
        }
    )


def funding_growth_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["bulan"],
            y=df["umkm_terdanai"],
            name="UMKM Terdanai",
            marker_color="rgba(56,189,248,0.52)",
            yaxis="y2",
            hovertemplate="%{x|%b %Y}<br>UMKM: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["bulan"],
            y=df["pendanaan_miliar"],
            mode="lines+markers",
            name="Pendanaan Hijau",
            line=dict(color="#1edc8d", width=3),
            marker=dict(size=7),
            hovertemplate="%{x|%b %Y}<br>Pendanaan: Rp %{y:.1f} M<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=355,
        margin=dict(t=25, b=20, l=16, r=16),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,12,24,0.55)",
        legend=dict(orientation="h", y=1.07, x=0),
        xaxis=dict(title=None, gridcolor="rgba(151,177,214,0.12)"),
        yaxis=dict(title="Pendanaan (Miliar Rp)", rangemode="tozero", gridcolor="rgba(151,177,214,0.12)"),
        yaxis2=dict(title="UMKM", overlaying="y", side="right", showgrid=False, rangemode="tozero"),
    )
    return fig


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🧭 SinergiVest")
        st.caption("Prototype PC/Laptop untuk green financing, saham IDX, dan BotVes.")
        pages = {
            "home": "🏠 Home",
            "chat": "🤖 BotVes",
            "login": "🔐 Masuk/Daftar",
            "portfolio": "📈 Simulasi Portofolio",
        }
        selected = st.radio(
            "Navigasi",
            list(pages.keys()),
            index=list(pages.keys()).index(st.session_state.page),
            format_func=pages.get,
        )
        if selected != st.session_state.page:
            st.session_state.page = selected
            st.rerun()

        location = st.session_state.get("geo_location", {"city": "Semarang", "regionName": "Jawa Tengah"})
        st.markdown("---")
        st.caption(f"📍 Lokasi: {location.get('city')}, {location.get('regionName')}")
        st.caption("Data saham memakai yfinance bila koneksi tersedia; fallback sintetis hanya untuk menjaga UI tetap berjalan.")


def render_hero() -> None:
    st.markdown(
        """
        <section class="sv-hero">
            <span class="sv-badge">💹 Green Finance Intelligence</span>
            <h1>🌱 SinergiVest</h1>
            <p>
                Dashboard investasi hijau yang menggabungkan tracking 11 sektor saham Indonesia,
                kurasi UMKM hijau, ESG micro-scoring, simulasi compound growth, dan BotVes berbasis Gemini.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_key_metrics() -> None:
    total_funding = sum(item.funding_need for item in UMKM_PARTNERS)
    avg_esg = round(float(np.mean([item.esg_score for item in UMKM_PARTNERS])))
    avg_yield = float(np.mean([item.target_yield for item in UMKM_PARTNERS]))
    metrics = [
        ("💰 Pipeline Pendanaan UMKM", rupiah(total_funding), "4 proyek hijau terkurasi"),
        ("📊 Jumlah Saham Tracked", f"{len(ALL_TICKERS)} emiten", "Tersebar di 11 sektor IDX"),
        ("🧭 Rata-rata ESG", f"{avg_esg}/100", "Bobot E 40%, S 30%, G 30%"),
        ("📈 Imbal Hasil Simulasi", f"{avg_yield:.1f}% p.a.", "Rata-rata target UMKM"),
    ]
    cols = st.columns(4, gap="small")
    for col, (label, value, note) in zip(cols, metrics):
        col.markdown(
            f"""
            <div class="sv-metric">
                <small>{label}</small>
                <strong>{value}</strong>
                <span>{note}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_stock_carousel() -> None:
    ticker = ALL_TICKERS[st.session_state.stock_idx]
    company = STOCK_NAMES.get(ticker, ticker)
    sector = next((name for name, tickers in SEKTOR_SAHAM.items() if ticker in tickers), "Sektor IDX")
    series = fetch_stock_close(ticker)
    stats = compute_stock_stats(tuple(series.astype(float).round(6).tolist()))

    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    section_heading(f"📉 Saham Carousel: {company} ({ticker})")
    st.caption(f"🧩 {sector.replace('_', ' ')} | Data Close 1 tahun terakhir")
    st.plotly_chart(stock_line_chart(ticker, series), use_container_width=True, config={"displayModeBar": False})

    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Harga Close", rupiah(stats["last"]))
        c2.metric("Log-Return Harian", percent(stats["mu"], 3))
        c3.metric("Volatilitas", percent(stats["sigma"], 2))
        c4.metric("VaR 95% CF", percent(stats["var95_cf"], 2))

        with st.expander("🧠 Analisis Statistik & Saran MA30", expanded=False):
            st.write(stock_advice_narrative(ticker, stats))
            st.write(
                f"MA30 saat ini {rupiah(stats['ma30'])}. Skewness {stats['skewness']:.3f}, "
                f"excess kurtosis {stats['excess_kurtosis']:.3f}. Gunakan ini sebagai sinyal edukatif, bukan nasihat investasi."
            )
    else:
        st.warning("Data belum cukup untuk analisis statistik.")

    prev_col, next_col = st.columns(2, gap="small")
    if prev_col.button("⬅ Prev Saham", key="prev_stock"):
        st.session_state.stock_idx = (st.session_state.stock_idx - 1) % len(ALL_TICKERS)
        st.rerun()
    if next_col.button("Next Saham ➡", key="next_stock"):
        st.session_state.stock_idx = (st.session_state.stock_idx + 1) % len(ALL_TICKERS)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_video_carousel() -> None:
    video = OJK_VIDEOS[st.session_state.edu_idx]
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    section_heading("🎬 Video OJK Carousel")
    st.video(video["url"])
    st.caption(f"📚 {video['title']}")
    st.markdown(
        '<p class="sv-grid-note">Materi literasi keuangan digunakan sebagai pendamping edukasi sebelum investor mencoba simulasi.</p>',
        unsafe_allow_html=True,
    )
    prev_col, next_col = st.columns(2, gap="small")
    if prev_col.button("⬅ Prev Edu", key="prev_edu"):
        st.session_state.edu_idx = (st.session_state.edu_idx - 1) % len(OJK_VIDEOS)
        st.rerun()
    if next_col.button("Next Edu ➡", key="next_edu"):
        st.session_state.edu_idx = (st.session_state.edu_idx + 1) % len(OJK_VIDEOS)
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_umkm_card() -> None:
    partners = ordered_umkm_partners()
    umkm = partners[st.session_state.umkm_idx % len(partners)]
    location = st.session_state.get("geo_location", {"city": "Semarang", "regionName": "Jawa Tengah"})
    st.markdown(
        f"""
        <div class="sv-card">
            <span class="sv-pill">📍 Prioritas lokasi: {location.get('city')}, {location.get('regionName')}</span>
            <h3>🏪 UMKM Hijau: {umkm.name}</h3>
            <p class="sv-muted">{umkm.sector} | {umkm.location}</p>
            <p>{umkm.impact}</p>
            <hr>
            <p><strong>Kebutuhan dana:</strong> {rupiah(umkm.funding_need)}</p>
            <p><strong>Tenor:</strong> {umkm.tenor_months} bulan | <strong>Target yield:</strong> {umkm.target_yield:.1f}% p.a.</p>
            <p><strong>Penggunaan dana:</strong> {umkm.use_of_funds}</p>
            <p class="sv-grid-note">Jika lokasi IP berada di Jawa Tengah, carousel memprioritaskan Kopi Lestari Arunika dan Batik Tirta Warna.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    prev_col, next_col = st.columns(2, gap="small")
    if prev_col.button("⬅ UMKM sebelumnya", key="prev_umkm"):
        st.session_state.umkm_idx = (st.session_state.umkm_idx - 1) % len(partners)
        st.rerun()
    if next_col.button("UMKM berikutnya ➡", key="next_umkm"):
        st.session_state.umkm_idx = (st.session_state.umkm_idx + 1) % len(partners)
        st.rerun()


def render_esg_panel() -> None:
    partners = ordered_umkm_partners()
    umkm = partners[st.session_state.umkm_idx % len(partners)]
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    section_heading(f"🧪 ESG Panel: {umkm.name}")
    gauge_col, bar_col = st.columns([1, 1.12], gap="small")
    with gauge_col:
        st.plotly_chart(esg_gauge(umkm.esg_score), use_container_width=True, config={"displayModeBar": False})
        st.info(f"Status risiko: {umkm.risk_label}")
    with bar_col:
        st.plotly_chart(esg_breakdown_chart(umkm), use_container_width=True, config={"displayModeBar": False})
        st.caption("Skor mockup: Lingkungan 40%, Sosial 30%, Tata Kelola 30%.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_bottom_navigation() -> None:
    st.markdown("### 🧭 Navigasi Cepat")
    left_pad, nav_area, right_pad = st.columns([0.55, 3.2, 0.55])
    with nav_area:
        c1, c2, c3, c4, c5 = st.columns(5, gap="small")
        if c1.button("🤖 Mulai BotVes", key="nav_bot"):
            st.session_state.page = "chat"
            st.rerun()
        if c2.button("🔐 Masuk/Daftar", key="nav_login"):
            st.session_state.page = "login"
            st.rerun()
        if c3.button("📈 Simulasi", key="nav_portfolio"):
            st.session_state.page = "portfolio"
            st.rerun()
        if c4.button("🏪 UMKM", key="nav_umkm_focus"):
            st.session_state.umkm_idx = 0
            st.rerun()
        if c5.button("🏠 Home", key="nav_home"):
            st.session_state.page = "home"
            st.rerun()


def render_home() -> None:
    render_hero()
    render_key_metrics()

    row1_left, row1_right = st.columns(2, gap="medium")
    with row1_left:
        render_stock_carousel()
    with row1_right:
        render_video_carousel()

    row2_left, row2_right = st.columns(2, gap="medium")
    with row2_left:
        render_umkm_card()
    with row2_right:
        render_esg_panel()

    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    section_heading("📊 Tren Pendanaan & Pertumbuhan UMKM")
    st.plotly_chart(funding_growth_chart(build_funding_timeseries()), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    render_bottom_navigation()


@st.cache_data(show_spinner=False)
def compound_projection(principal: float, annual_return: float, years: int) -> pd.DataFrame:
    year_range = np.arange(0, years + 1)
    values = principal * np.power(1 + annual_return, year_range)
    return pd.DataFrame({"Tahun": year_range, "Nilai Masa Depan": values, "Imbal Hasil": values - principal})


def render_portfolio() -> None:
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    section_heading("📈 Simulasi Portofolio")
    st.caption("Hitung nilai masa depan aset berbasis compound annual growth dari input lot/saham Anda.")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("portfolio_form", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
        ticker = c1.selectbox("Saham", ALL_TICKERS, format_func=lambda x: f"{x} - {STOCK_NAMES.get(x, x)}")
        lots = c2.number_input("Lot", min_value=0, max_value=100_000, value=1, step=1)
        shares_extra = c3.number_input("Saham ekstra", min_value=0, max_value=99, value=0, step=1)
        manual_price = c4.number_input("Harga manual/lembar", min_value=0, value=0, step=50)
        target_return = st.slider("Target CAGR tahunan (%)", min_value=-20, max_value=60, value=12, step=1) / 100
        years = st.slider("Jangka waktu (tahun)", min_value=1, max_value=20, value=5, step=1)
        submitted = st.form_submit_button("➕ Tambahkan & Simulasikan")

    if submitted:
        series = fetch_stock_close(ticker)
        market_price = float(series.iloc[-1]) if len(series) else float(manual_price)
        price = float(manual_price or market_price)
        shares = int(lots * 100 + shares_extra)
        if shares <= 0:
            st.error("Masukkan minimal 1 saham atau 1 lot.")
        else:
            st.session_state.portfolio.append(
                {
                    "Ticker": ticker,
                    "Nama": STOCK_NAMES.get(ticker, ticker),
                    "Saham": shares,
                    "Harga": price,
                    "Nilai": shares * price,
                }
            )
            st.success(f"{ticker} ditambahkan: {shares:,} saham x {rupiah(price)}.")

    if st.session_state.portfolio:
        portfolio_df = pd.DataFrame(st.session_state.portfolio)
        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
        total_value = float(portfolio_df["Nilai"].sum())
        projection = compound_projection(total_value, target_return, years)

        m1, m2, m3 = st.columns(3)
        m1.metric("Nilai Awal", rupiah(total_value))
        m2.metric("Target CAGR", f"{target_return * 100:.0f}%")
        m3.metric("Nilai Akhir", rupiah(projection["Nilai Masa Depan"].iloc[-1]))

        fig = go.Figure(
            go.Bar(
                x=projection["Tahun"],
                y=projection["Nilai Masa Depan"],
                marker_color="#1edc8d",
                text=[rupiah(value) for value in projection["Nilai Masa Depan"]],
                textposition="outside",
                hovertemplate="Tahun %{x}<br>Nilai: Rp %{y:,.0f}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            height=390,
            margin=dict(t=28, b=25, l=16, r=16),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(4,12,24,0.55)",
            xaxis_title="Tahun",
            yaxis_title="Nilai Masa Depan",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        c1, c2 = st.columns(2)
        if c1.button("🧹 Kosongkan Portofolio"):
            st.session_state.portfolio = []
            st.rerun()
        if c2.button("🏠 Kembali ke Home"):
            st.session_state.page = "home"
            st.rerun()
    else:
        st.info("Portofolio masih kosong. Tambahkan lot/saham untuk melihat proyeksi.")
        if st.button("🏠 Kembali ke Home"):
            st.session_state.page = "home"
            st.rerun()


def parse_budget(text: str) -> int | None:
    clean = text.lower().replace(" ", "")
    match = re.search(r"(\d+(?:[,.]\d+)?)(miliar|milyar|billion|juta|jt|ribu|rb|k|m)?", clean)
    if not match:
        return None
    try:
        value = float(match.group(1).replace(",", "."))
    except ValueError:
        return None
    unit = match.group(2)
    if unit in {"miliar", "milyar", "billion"}:
        return int(value * 1_000_000_000)
    if unit in {"juta", "jt", "m"}:
        return int(value * 1_000_000)
    if unit in {"ribu", "rb", "k"}:
        return int(value * 1_000)
    if value < 10_000:
        return int(value * 1_000_000)
    return int(value)


def parse_risk_preference(text: str) -> str:
    lowered = text.lower()
    if any(word in lowered for word in ["konservatif", "aman", "rendah", "stabil", "defensif"]):
        return "conservative"
    if any(word in lowered for word in ["agresif", "tinggi", "growth", "cepat", "spekulatif"]):
        return "aggressive"
    return "balanced"


def parse_sector(text: str) -> str | None:
    lowered = text.lower()
    aliases = {
        "1_Keuangan": ["bank", "keuangan", "finansial", "syariah"],
        "2_Pertambangan_Energi": ["tambang", "pertambangan", "energi", "batubara", "gas", "emas"],
        "3_Kesehatan": ["kesehatan", "farmasi", "obat", "health"],
        "4_Infrastruktur": ["infrastruktur", "telkom", "tol", "telekomunikasi"],
        "5_Barang_Baku": ["barang baku", "semen", "timah", "material"],
        "6_Konsumen_Primer": ["konsumen primer", "makanan", "retail", "consumer staple"],
        "7_Konsumen_Non_Primer": ["otomotif", "gaya hidup", "konsumen non primer", "goto"],
        "8_Properti": ["properti", "real estate"],
        "9_Teknologi": ["teknologi", "tech", "digital"],
        "10_Perindustrian": ["industri", "perindustrian", "alat berat", "pakan"],
        "11_Transportasi_Logistik": ["transportasi", "logistik", "kapal", "taksi"],
    }
    for sector, words in aliases.items():
        if any(word in lowered for word in words):
            return sector
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def recommend_for_budget(budget: int, risk_pref: str = "balanced", sector: str | None = None, top_n: int = 3) -> list[dict[str, Any]]:
    tickers = SEKTOR_SAHAM.get(sector, ALL_TICKERS) if sector else ALL_TICKERS
    candidates: list[dict[str, Any]] = []
    for ticker in tickers:
        series = fetch_stock_close(ticker)
        stats = compute_stock_stats(tuple(series.astype(float).round(6).tolist()))
        if not stats:
            continue
        lot_price = stats["last"] * 100
        if lot_price > budget:
            continue
        if risk_pref == "conservative":
            score = -stats["sigma"] + max(stats["total_return"], -0.25) * 0.1
        elif risk_pref == "aggressive":
            score = stats["total_return"] - stats["var95_cf"]
        else:
            score = stats["mu"] / (stats["sigma"] + 1e-9)
        candidates.append(
            {
                "ticker": ticker,
                "company": STOCK_NAMES.get(ticker, ticker),
                "last": stats["last"],
                "lot_price": lot_price,
                "sigma": stats["sigma"],
                "total_return": stats["total_return"],
                "advice": stats["advice"],
                "score": score,
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:top_n]


def health_investment_correlation(risk_pref: str, budget: int) -> str:
    ratios = {"conservative": 0.30, "balanced": 0.20, "aggressive": 0.12}
    ratio = ratios.get(risk_pref, 0.20)
    health_fund = int(budget * ratio)
    investment_fund = budget - health_fund
    profile = {
        "conservative": "konservatif",
        "balanced": "seimbang",
        "aggressive": "agresif",
    }.get(risk_pref, "seimbang")
    return (
        f"Untuk profil {profile}, sisihkan sekitar {rupiah(health_fund)} "
        f"({ratio * 100:.0f}% dari modal) sebagai dana kesehatan/darurat, sehingga dana investasi efektif "
        f"sekitar {rupiah(investment_fund)}. Logikanya: makin kuat bantalan kesehatan, makin kecil peluang investasi "
        "terpaksa dicairkan saat ada kebutuhan mendadak."
    )


def uploaded_portfolio_summary() -> str:
    df = st.session_state.get("uploaded_df")
    if df is None or df.empty:
        return ""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    summary = f"File portofolio berisi {len(df)} baris dan {len(df.columns)} kolom."
    if numeric_cols:
        totals = df[numeric_cols].sum(numeric_only=True).head(4)
        summary += " Ringkasan numerik: " + ", ".join(f"{col}={value:,.0f}" for col, value in totals.items())
    return summary


def local_botves_reply(message: str) -> str:
    budget = parse_budget(message)
    risk_pref = parse_risk_preference(message)
    sector = parse_sector(message)

    if budget:
        st.session_state.user_profile["budget"] = budget
    if risk_pref:
        st.session_state.user_profile["risk_pref"] = risk_pref
    if sector:
        st.session_state.user_profile["sector"] = sector

    budget = st.session_state.user_profile.get("budget") or budget
    risk_pref = st.session_state.user_profile.get("risk_pref") or risk_pref
    sector = sector or st.session_state.user_profile.get("sector")
    lowered = message.lower()

    if "kesehatan" in lowered or "dana darurat" in lowered:
        if not budget:
            return "Sebutkan dulu modal Anda, misalnya 'modal 2 juta konservatif', agar saya bisa menghitung porsi dana kesehatan dan investasi."
        return health_investment_correlation(risk_pref, budget)

    if budget:
        recs = recommend_for_budget(budget, risk_pref, sector, top_n=3)
        sector_label = sector.replace("_", " ") if sector else "lintas sektor"
        if not recs:
            return (
                f"Dengan modal {rupiah(budget)}, saya belum menemukan saham 1 lot yang cocok pada {sector_label}. "
                "Pertimbangkan menambah modal, memakai DCA bertahap, atau memilih instrumen pecahan seperti reksa dana."
            )
        intro = (
            f"Dengan modal {rupiah(budget)} dan profil {risk_pref}, kandidat {sector_label} yang masih terjangkau adalah:\n"
        )
        rows = []
        for item in recs:
            rows.append(
                f"- {item['ticker']} ({item['company']}): harga/lembar {rupiah(item['last'])}, "
                f"1 lot {rupiah(item['lot_price'])}, volatilitas {percent(item['sigma'])}, sinyal {item['advice']}."
            )
        if "kesehatan" in lowered:
            rows.append(health_investment_correlation(risk_pref, budget))
        rows.append("Ini simulasi edukatif berbasis data historis, bukan nasihat keuangan profesional.")
        return intro + "\n".join(rows)

    ticker_match = re.search(r"\b([A-Za-z]{2,5})(?:\.JK)?\b", message)
    if ticker_match:
        ticker = ticker_match.group(1).upper()
        ticker = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
        if ticker in STOCK_NAMES:
            series = fetch_stock_close(ticker)
            stats = compute_stock_stats(tuple(series.astype(float).round(6).tolist()))
            if stats:
                return stock_advice_narrative(ticker, stats)

    if any(word in lowered for word in ["saham apa", "rekomendasi", "cocok"]):
        return "Sebutkan modal dan risiko, misalnya 'modal 3 juta konservatif sektor keuangan', supaya rekomendasinya bisa dipersonalisasi."

    return (
        "Saya bisa membaca modal, preferensi risiko, sektor, ticker saham, serta konteks dana kesehatan. "
        "Coba tulis: 'modal 2 juta konservatif sektor bank' atau 'analisis BBCA'."
    )


def gemini_reply(message: str) -> str | None:
    if not HAS_GEMINI or genai is None:
        return None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        return None

    local_context = local_botves_reply(message)
    system_context = (
        "Anda adalah BotVes, asisten edukasi investasi Indonesia. Jawab ringkas, konkret, "
        "tidak mengklaim sebagai nasihat keuangan profesional. Gunakan data konteks berikut sebagai basis: "
        f"{local_context} {uploaded_portfolio_summary()}"
    )
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([system_context, message])
        text = getattr(response, "text", None)
        if text:
            return text.strip()
    except Exception:
        return None
    return None


def render_chat_message(role: str, content: str) -> None:
    css_class = "sv-chat-user" if role == "user" else "sv-chat-bot"
    label = "🧑 Anda" if role == "user" else "🤖 BotVes"
    st.markdown(f"<div class='{css_class}'><strong>{label}</strong><br>{content}</div>", unsafe_allow_html=True)


def render_chat() -> None:
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    section_heading("🤖 BotVes Chat Page")
    st.caption("Chatbot edukatif terhubung Gemini bila `st.secrets['GEMINI_API_KEY']` tersedia; fallback lokal aktif otomatis.")
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("📎 Unggah ringkasan portofolio opsional (.csv/.xlsx)", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.uploaded_df = df
            st.success(f"File {uploaded.name} berhasil dibaca.")
            st.dataframe(df.head(8), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Gagal membaca file: {exc}")

    for item in st.session_state.chat_history[-30:]:
        render_chat_message(item["role"], item["content"])

    prompt = st.chat_input("Ketik modal, risiko, sektor, ticker, atau pertanyaan dana kesehatan...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("🤖 BotVes sedang menyusun jawaban..."):
            reply = gemini_reply(prompt) or local_botves_reply(prompt)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.session_state.chat_history = st.session_state.chat_history[-30:]
        st.rerun()

    c1, c2 = st.columns(2)
    if c1.button("🧹 Reset Chat"):
        st.session_state.chat_history = default_chat_history()
        st.rerun()
    if c2.button("🏠 Kembali ke Home"):
        st.session_state.page = "home"
        st.rerun()


def render_login() -> None:
    st.markdown('<div class="sv-card">', unsafe_allow_html=True)
    section_heading("🔐 Login Page")
    st.caption("Form demo registrasi/masuk dengan captcha sederhana. Gunakan kode demo: 1234.")
    st.markdown("</div>", unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["🚪 Masuk", "📝 Daftar"])
    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email atau username")
            password = st.text_input("Password", type="password")
            captcha = st.text_input("Captcha demo")
            submitted = st.form_submit_button("🔓 Masuk")
        if submitted:
            if not email or not password:
                st.error("Lengkapi email/username dan password.")
            elif captcha.strip() != "1234":
                st.error("Captcha salah. Kode demo adalah 1234.")
            else:
                st.session_state.user_profile["name"] = email
                st.success("Login demo berhasil.")
                st.session_state.page = "home"
                st.rerun()

    with tab_register:
        with st.form("register_form"):
            name = st.text_input("Nama lengkap")
            role = st.selectbox("Peran", ["Investor Ritel", "UMKM Hijau", "Lembaga Pendamping", "Admin Kurator"])
            reg_email = st.text_input("Email")
            reg_password = st.text_input("Password baru", type="password")
            reg_captcha = st.text_input("Captcha demo pendaftaran")
            registered = st.form_submit_button("✅ Daftar Demo")
        if registered:
            if not name or not reg_email or not reg_password:
                st.error("Lengkapi semua field pendaftaran.")
            elif reg_captcha.strip() != "1234":
                st.error("Captcha salah. Kode demo adalah 1234.")
            else:
                st.session_state.user_profile["name"] = name
                st.success(f"Akun demo {role} untuk {name} berhasil dibuat.")

    if st.button("🏠 Kembali ke Home"):
        st.session_state.page = "home"
        st.rerun()


def main() -> None:
    init_state()
    startup_loader()
    inject_css()
    apply_geolocation_priority()
    render_sidebar()

    if st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "portfolio":
        render_portfolio()
    elif st.session_state.page == "chat":
        render_chat()
    elif st.session_state.page == "login":
        render_login()
    else:
        st.session_state.page = "home"
        render_home()


if __name__ == "__main__":
    main()
