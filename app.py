# app.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="SinergiVest",
    page_icon="SV",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@dataclass(frozen=True)
class GreenMSME:
    name: str
    sector: str
    location: str
    funding_need: int
    target_yield: float
    tenor_months: int
    environmental_score: int
    social_score: int
    governance_score: int
    impact: str
    use_of_funds: str

    @property
    def esg_score(self) -> int:
        weighted_score = (
            self.environmental_score * 0.4
            + self.social_score * 0.3
            + self.governance_score * 0.3
        )
        return round(weighted_score)

    @property
    def risk_label(self) -> str:
        if self.esg_score >= 85:
            return "Rendah"
        if self.esg_score >= 75:
            return "Moderat"
        return "Perlu pendampingan"


UMKM_PARTNERS: list[GreenMSME] = [
    GreenMSME(
        name="Kopi Lestari Arunika",
        sector="Agrikultur regeneratif",
        location="Temanggung, Jawa Tengah",
        funding_need=180_000_000,
        target_yield=11.5,
        tenor_months=18,
        environmental_score=89,
        social_score=84,
        governance_score=78,
        impact="Pengolahan limbah kulit kopi menjadi kompos dan briket biomassa.",
        use_of_funds="Mesin pengering hemat energi dan perluasan rumah kompos.",
    ),
    GreenMSME(
        name="Batik Tirta Warna",
        sector="Fesyen sirkular",
        location="Pekalongan, Jawa Tengah",
        funding_need=125_000_000,
        target_yield=10.2,
        tenor_months=12,
        environmental_score=82,
        social_score=88,
        governance_score=80,
        impact="Pewarna alami, instalasi filtrasi air, dan pelatihan perajin perempuan.",
        use_of_funds="IPAL mikro, stok pewarna nabati, dan sertifikasi produksi bersih.",
    ),
    GreenMSME(
        name="Surya Desa Mandiri",
        sector="Energi terbarukan",
        location="Lombok Timur, NTB",
        funding_need=260_000_000,
        target_yield=12.8,
        tenor_months=24,
        environmental_score=93,
        social_score=81,
        governance_score=86,
        impact="Panel surya komunal untuk cold storage nelayan dan warung desa.",
        use_of_funds="Pembelian panel surya, baterai, dan sistem monitoring energi.",
    ),
    GreenMSME(
        name="DaurPack Nusantara",
        sector="Kemasan daur ulang",
        location="Bandung, Jawa Barat",
        funding_need=95_000_000,
        target_yield=9.6,
        tenor_months=10,
        environmental_score=77,
        social_score=79,
        governance_score=74,
        impact="Produksi kemasan dari kertas pascakonsumsi untuk pelaku kuliner lokal.",
        use_of_funds="Cetakan baru, bahan baku daur ulang, dan digitalisasi pembukuan.",
    ),
]


OJK_VIDEOS: list[dict[str, str]] = [
    {
        "title": "Literasi keuangan digital untuk masyarakat",
        "url": "https://www.youtube.com/watch?v=GFgbxJyCSmE",
    },
    {
        "title": "Cerdas mengelola keuangan dan investasi",
        "url": "https://www.youtube.com/watch?v=Is3BfJN3bp0",
    },
    {
        "title": "Edukasi investasi dan perlindungan konsumen",
        "url": "https://www.youtube.com/watch?v=FcMG-ZMQP1g",
    },
]


def default_chat_history() -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": (
                "Halo, saya BotVes Green. Sebutkan sektor, nominal pendanaan, "
                "atau target dampak yang ingin Anda simulasikan."
            ),
        }
    ]


def init_state() -> None:
    defaults: dict[str, Any] = {
        "page": "home",
        "umkm_idx": 0,
        "video_idx": 0,
        "chat_history": default_chat_history(),
        "funding_cart": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --sv-primary: #15803d;
            --sv-primary-soft: rgba(22, 163, 74, 0.14);
            --sv-border: rgba(120, 120, 120, 0.24);
            --sv-shadow: 0 12px 32px rgba(15, 23, 42, 0.10);
        }

        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .sv-hero {
            border: 1px solid var(--sv-border);
            border-radius: 8px;
            padding: clamp(1.25rem, 3vw, 2rem);
            background:
                linear-gradient(135deg, rgba(22, 163, 74, 0.18), transparent 46%),
                var(--background-color);
            color: var(--text-color);
            box-shadow: var(--sv-shadow);
        }

        .sv-hero h1 {
            margin: 0;
            line-height: 1.08;
            font-size: clamp(2rem, 4vw, 3.6rem);
            letter-spacing: 0;
        }

        .sv-hero p {
            margin: 0.75rem 0 0;
            max-width: 900px;
            color: var(--text-color);
            opacity: 0.86;
            font-size: 1.02rem;
        }

        .sv-card {
            border: 1px solid var(--sv-border);
            border-radius: 8px;
            padding: 1rem;
            background: var(--background-color);
            color: var(--text-color);
            box-shadow: var(--sv-shadow);
            height: 100%;
        }

        .sv-card-muted {
            border: 1px solid var(--sv-border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            background: var(--sv-primary-soft);
            color: var(--text-color);
        }

        .sv-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            border: 1px solid var(--sv-border);
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            background: var(--sv-primary-soft);
            color: var(--text-color);
            font-size: 0.88rem;
            font-weight: 600;
        }

        .sv-score {
            font-size: clamp(2.5rem, 6vw, 4.4rem);
            line-height: 1;
            font-weight: 800;
            color: var(--sv-primary);
        }

        .sv-small {
            color: var(--text-color);
            opacity: 0.72;
            font-size: 0.9rem;
        }

        .sv-chat-user,
        .sv-chat-bot {
            border: 1px solid var(--sv-border);
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin: 0.5rem 0;
            color: var(--text-color);
        }

        .sv-chat-user {
            background: var(--sv-primary-soft);
            margin-left: 12%;
        }

        .sv-chat-bot {
            background: var(--background-color);
            margin-right: 12%;
        }

        div.stButton > button {
            width: 100%;
            border-radius: 8px;
            min-height: 2.65rem;
            font-weight: 650;
            border: 1px solid var(--sv-border);
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--sv-border);
            border-radius: 8px;
            padding: 0.75rem;
            background: var(--background-color);
            color: var(--text-color);
        }

        @media (max-width: 760px) {
            .sv-chat-user,
            .sv-chat-bot {
                margin-left: 0;
                margin-right: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def rupiah(value: float | int) -> str:
    return f"Rp {value:,.0f}".replace(",", ".")


def current_umkm() -> GreenMSME:
    return UMKM_PARTNERS[st.session_state.umkm_idx]


@st.cache_data(show_spinner=False)
def build_funding_timeseries(seed: int, months: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=months, freq="MS")
    base = np.linspace(35, 100, months)
    noise = rng.normal(0, 4, months).cumsum()
    disbursement = np.clip(base + noise, 20, 120)
    beneficiaries = np.clip(np.linspace(18, 75, months) + rng.normal(0, 3, months), 10, 90)
    return pd.DataFrame(
        {
            "bulan": dates,
            "pendanaan_miliar": disbursement,
            "umkm_terdanai": beneficiaries.round().astype(int),
        }
    )


def funding_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["bulan"],
            y=df["pendanaan_miliar"],
            mode="lines+markers",
            name="Pendanaan hijau",
            line=dict(color="#16a34a", width=3),
            marker=dict(size=7),
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["bulan"],
            y=df["umkm_terdanai"],
            name="UMKM terdanai",
            marker_color="rgba(14, 165, 233, 0.55)",
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=360,
        margin=dict(t=25, r=20, b=20, l=20),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        yaxis=dict(title="Pendanaan (miliar Rp)", rangemode="tozero"),
        yaxis2=dict(
            title="Jumlah UMKM",
            overlaying="y",
            side="right",
            rangemode="tozero",
            showgrid=False,
        ),
        xaxis=dict(title=None),
    )
    return fig


def esg_gauge(score: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#15803d"},
                "steps": [
                    {"range": [0, 60], "color": "rgba(239, 68, 68, 0.25)"},
                    {"range": [60, 80], "color": "rgba(234, 179, 8, 0.28)"},
                    {"range": [80, 100], "color": "rgba(22, 163, 74, 0.24)"},
                ],
                "threshold": {
                    "line": {"color": "#0f766e", "width": 4},
                    "thickness": 0.75,
                    "value": 80,
                },
            },
        )
    )
    fig.update_layout(height=270, margin=dict(t=10, b=10, l=10, r=10))
    return fig


def esg_breakdown_chart(umkm: GreenMSME) -> go.Figure:
    labels = ["Lingkungan", "Sosial", "Tata kelola"]
    values = [umkm.environmental_score, umkm.social_score, umkm.governance_score]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=["#16a34a", "#0ea5e9", "#6366f1"],
            text=[f"{value}/100" for value in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        height=220,
        xaxis=dict(range=[0, 100], title=None),
        yaxis=dict(title=None, autorange="reversed"),
        margin=dict(t=10, b=10, l=10, r=10),
        template="plotly_white",
        showlegend=False,
    )
    return fig


def render_header() -> None:
    st.markdown(
        """
        <section class="sv-hero">
            <span class="sv-pill">Green financing mockup</span>
            <h1>SinergiVest: Pionir Green Financing & Investasi Hijau UMKM Inklusif</h1>
            <p>
                Platform simulasi pendanaan hijau untuk mempertemukan investor ritel,
                lembaga keuangan, dan UMKM yang memiliki dampak lingkungan, sosial,
                serta tata kelola terukur melalui AI-Driven Micro ESG Score.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_key_metrics() -> None:
    total_funding = sum(item.funding_need for item in UMKM_PARTNERS)
    avg_score = round(np.mean([item.esg_score for item in UMKM_PARTNERS]))
    avg_yield = np.mean([item.target_yield for item in UMKM_PARTNERS])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipeline pendanaan", rupiah(total_funding))
    c2.metric("UMKM mitra", f"{len(UMKM_PARTNERS)} terkurasi")
    c3.metric("Rata-rata ESG", f"{avg_score}/100")
    c4.metric("Imbal hasil simulasi", f"{avg_yield:.1f}% p.a.")


def render_umkm_card(umkm: GreenMSME) -> None:
    st.markdown(
        f"""
        <div class="sv-card">
            <span class="sv-pill">{umkm.sector}</span>
            <h2 style="margin-bottom:0.25rem;">{umkm.name}</h2>
            <div class="sv-small">{umkm.location}</div>
            <p>{umkm.impact}</p>
            <div class="sv-card-muted">
                <strong>Kebutuhan dana:</strong> {rupiah(umkm.funding_need)}<br>
                <strong>Tenor:</strong> {umkm.tenor_months} bulan<br>
                <strong>Target imbal hasil:</strong> {umkm.target_yield:.1f}% per tahun<br>
                <strong>Penggunaan dana:</strong> {umkm.use_of_funds}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_esg_panel(umkm: GreenMSME) -> None:
    st.markdown("#### AI-Driven Micro ESG Score")
    score_col, detail_col = st.columns([1.1, 1.2], gap="large")
    with score_col:
        st.plotly_chart(esg_gauge(umkm.esg_score), width="stretch", config={"displayModeBar": False})
        st.markdown(
            f"""
            <div class="sv-card-muted">
                <strong>Status risiko:</strong> {umkm.risk_label}<br>
                <span class="sv-small">
                    Skor ini adalah mockup untuk kebutuhan presentasi akademik,
                    bukan pemeringkatan kredit aktual.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with detail_col:
        st.plotly_chart(esg_breakdown_chart(umkm), width="stretch", config={"displayModeBar": False})
        st.caption(
            "Formula mock: Lingkungan 40%, Sosial 30%, Tata kelola 30%. "
            "Pada produk nyata, bobot dapat dikalibrasi dari data pembayaran, audit lapangan, dan dokumen legal."
        )


def render_chart_and_video() -> None:
    chart_col, video_col = st.columns([3, 2], gap="large")
    with chart_col:
        st.subheader("Tren Pendanaan & Pertumbuhan UMKM")
        df = build_funding_timeseries(seed=42)
        st.plotly_chart(funding_chart(df), width="stretch", config={"displayModeBar": False})
    with video_col:
        st.subheader("Video Edukasi Literasi Keuangan")
        video = OJK_VIDEOS[st.session_state.video_idx]
        st.video(video["url"])
        st.caption(video["title"])
        v_prev, v_next = st.columns(2)
        if v_prev.button("Video sebelumnya", width="stretch"):
            st.session_state.video_idx = (st.session_state.video_idx - 1) % len(OJK_VIDEOS)
            st.rerun()
        if v_next.button("Video berikutnya", width="stretch"):
            st.session_state.video_idx = (st.session_state.video_idx + 1) % len(OJK_VIDEOS)
            st.rerun()


def render_navigation_buttons() -> None:
    previous_col, next_col, bot_col, login_col, portfolio_col = st.columns(5, gap="small")
    if previous_col.button("UMKM sebelumnya", width="stretch"):
        st.session_state.umkm_idx = (st.session_state.umkm_idx - 1) % len(UMKM_PARTNERS)
        st.rerun()
    if next_col.button("UMKM berikutnya", width="stretch"):
        st.session_state.umkm_idx = (st.session_state.umkm_idx + 1) % len(UMKM_PARTNERS)
        st.rerun()
    if bot_col.button("BotVes Green", width="stretch"):
        st.session_state.page = "chat"
        st.rerun()
    if login_col.button("Masuk / Daftar", width="stretch"):
        st.session_state.page = "login"
        st.rerun()
    if portfolio_col.button("Portofolio hijau", width="stretch"):
        st.session_state.page = "portfolio"
        st.rerun()


def render_home() -> None:
    render_header()
    st.write("")
    render_key_metrics()
    st.write("")

    umkm = current_umkm()
    profile_col, esg_col = st.columns([1.15, 1], gap="large")
    with profile_col:
        render_umkm_card(umkm)
    with esg_col:
        render_esg_panel(umkm)

    st.write("")
    render_chart_and_video()
    st.write("")
    render_navigation_buttons()


def funding_projection(principal: int, annual_yield: float, months: int) -> pd.DataFrame:
    month_index = np.arange(0, months + 1)
    monthly_rate = annual_yield / 100 / 12
    projected_value = principal * ((1 + monthly_rate) ** month_index)
    return pd.DataFrame(
        {
            "bulan_ke": month_index,
            "nilai_proyeksi": projected_value,
            "imbal_hasil": projected_value - principal,
        }
    )


def render_portfolio() -> None:
    st.title("Simulasi Portofolio Pendanaan Hijau")
    st.caption("Gunakan halaman ini untuk memodelkan dampak pendanaan, tenor, dan estimasi hasil secara edukatif.")

    umkm_options = {item.name: item for item in UMKM_PARTNERS}
    with st.form("funding_form", clear_on_submit=False):
        selected_name = st.selectbox("Pilih UMKM mitra", options=list(umkm_options.keys()))
        amount = st.number_input(
            "Nominal pendanaan simulasi",
            min_value=100_000,
            max_value=500_000_000,
            value=5_000_000,
            step=100_000,
        )
        submitted = st.form_submit_button("Tambahkan ke portofolio")

    if submitted:
        item = umkm_options[selected_name]
        st.session_state.funding_cart.append(
            {
                "UMKM": item.name,
                "Sektor": item.sector,
                "Nominal": amount,
                "Tenor": item.tenor_months,
                "Yield": item.target_yield,
                "ESG": item.esg_score,
            }
        )
        st.success(f"{item.name} ditambahkan ke portofolio simulasi.")

    if st.session_state.funding_cart:
        df = pd.DataFrame(st.session_state.funding_cart)
        st.dataframe(df, width="stretch", hide_index=True)

        total = int(df["Nominal"].sum())
        weighted_esg = np.average(df["ESG"], weights=df["Nominal"])
        weighted_yield = np.average(df["Yield"], weights=df["Nominal"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total pendanaan", rupiah(total))
        c2.metric("ESG tertimbang", f"{weighted_esg:.0f}/100")
        c3.metric("Yield tertimbang", f"{weighted_yield:.1f}% p.a.")

        projection = funding_projection(total, weighted_yield, 24)
        fig = go.Figure(
            go.Scatter(
                x=projection["bulan_ke"],
                y=projection["nilai_proyeksi"],
                mode="lines+markers",
                line=dict(color="#16a34a", width=3),
                name="Nilai proyeksi",
            )
        )
        fig.update_layout(
            height=330,
            template="plotly_white",
            margin=dict(t=25, b=20, l=20, r=20),
            xaxis_title="Bulan ke-",
            yaxis_title="Nilai proyeksi",
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

        clear_col, back_col = st.columns([1, 1])
        if clear_col.button("Kosongkan portofolio", width="stretch"):
            st.session_state.funding_cart = []
            st.rerun()
        if back_col.button("Kembali ke beranda", width="stretch"):
            st.session_state.page = "home"
            st.rerun()
    else:
        st.info("Portofolio simulasi masih kosong. Tambahkan satu UMKM untuk melihat proyeksi.")
        if st.button("Kembali ke beranda", width="stretch"):
            st.session_state.page = "home"
            st.rerun()


def render_login() -> None:
    st.title("Masuk / Daftar")
    st.caption("Form demo untuk menggambarkan onboarding investor, UMKM, atau lembaga pendamping.")
    with st.form("login_form"):
        role = st.selectbox("Peran", ["Investor ritel", "UMKM hijau", "Lembaga pendamping", "Admin kurator"])
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Masuk")

    if submitted:
        if not email or not password:
            st.error("Lengkapi email dan password terlebih dahulu.")
        else:
            st.success(f"Login demo berhasil sebagai {role}.")
            st.session_state.page = "home"
            st.rerun()

    if st.button("Kembali ke beranda", width="stretch"):
        st.session_state.page = "home"
        st.rerun()


def botves_reply(message: str) -> str:
    text = message.lower()
    if any(keyword in text for keyword in ["esg", "score", "skor"]):
        best = max(UMKM_PARTNERS, key=lambda item: item.esg_score)
        return (
            f"Skor ESG tertinggi saat ini adalah {best.name} dengan {best.esg_score}/100. "
            "Penilaiannya menonjol pada aspek lingkungan karena dampak reduksi emisi dan efisiensi energi."
        )
    if any(keyword in text for keyword in ["energi", "surya", "panel"]):
        energy = next(item for item in UMKM_PARTNERS if "Energi" in item.sector.title())
        return (
            f"Untuk tema energi terbarukan, {energy.name} membutuhkan {rupiah(energy.funding_need)} "
            f"dengan tenor {energy.tenor_months} bulan dan skor ESG {energy.esg_score}/100."
        )
    if any(keyword in text for keyword in ["risiko", "aman", "konservatif"]):
        low_risk = sorted(UMKM_PARTNERS, key=lambda item: item.esg_score, reverse=True)[:2]
        names = ", ".join(item.name for item in low_risk)
        return (
            f"Untuk profil konservatif, prioritaskan UMKM dengan skor ESG tinggi dan tata kelola kuat: {names}. "
            "Tetap lakukan diversifikasi karena ini hanya simulasi edukatif."
        )
    if any(keyword in text for keyword in ["modal", "dana", "pendanaan", "investasi"]):
        return (
            "Coba mulai dari nominal kecil, misalnya Rp 500.000 sampai Rp 5.000.000, lalu sebar ke dua "
            "atau tiga UMKM hijau agar risiko tidak terkonsentrasi pada satu proyek."
        )
    return (
        "Saya bisa membantu membandingkan UMKM berdasarkan skor ESG, kebutuhan dana, sektor hijau, "
        "atau risiko pendanaan. Contoh: 'mana UMKM energi terbaik?'"
    )


def render_chat() -> None:
    st.title("BotVes Green")
    st.caption("Asisten edukatif untuk simulasi green financing UMKM. Bukan nasihat investasi resmi.")

    for message in st.session_state.chat_history[-20:]:
        css_class = "sv-chat-user" if message["role"] == "user" else "sv-chat-bot"
        st.markdown(f"<div class='{css_class}'>{message['content']}</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Tanyakan tentang ESG, sektor hijau, risiko, atau nominal pendanaan...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": botves_reply(prompt)})
        st.rerun()

    back_col, clear_col = st.columns(2)
    if back_col.button("Kembali ke beranda", width="stretch"):
        st.session_state.page = "home"
        st.rerun()
    if clear_col.button("Reset percakapan", width="stretch"):
        st.session_state.chat_history = default_chat_history()
        st.rerun()


def render_sidebar() -> None:
    with st.sidebar:
        st.header("SinergiVest")
        st.caption("Prototype green financing untuk kompetisi esai ilmiah.")
        selected = st.radio(
            "Navigasi",
            options=["home", "portfolio", "chat", "login"],
            format_func={
                "home": "Beranda",
                "portfolio": "Portofolio hijau",
                "chat": "BotVes Green",
                "login": "Masuk / Daftar",
            }.get,
            index=["home", "portfolio", "chat", "login"].index(st.session_state.page),
        )
        if selected != st.session_state.page:
            st.session_state.page = selected
            st.rerun()


def main() -> None:
    init_state()
    inject_css()
    render_sidebar()

    if st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "portfolio":
        render_portfolio()
    elif st.session_state.page == "chat":
        render_chat()
    elif st.session_state.page == "login":
        render_login()


if __name__ == "__main__":
    main()
