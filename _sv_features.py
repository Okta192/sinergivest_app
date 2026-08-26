# _sv_features.py - OHLCV, candlestick, helpers
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    yf = None
    HAS_YFINANCE = False


def pct_change_color(change):
    if change > 0: return "#10b981"
    if change < 0: return "#ef4444"
    return "#94a3b8"


def advice_badge_html(advice):
    colors = {
        "Buy": ("#10b981", "rgba(16,185,129,0.15)"),
        "Hold": ("#eab308", "rgba(234,179,8,0.15)"),
        "Wait": ("#ef4444", "rgba(239,68,68,0.15)"),
    }
    colors = {
        "Buy": ("#10b981", "rgba(16,185,129,0.15)"),
        "Hold": ("#eab308", "rgba(234,179,8,0.15)"),
        "Wait": ("#ef4444", "rgba(239,68,68,0.15)"),
    }
    fg, bg = colors.get(advice, ("#94a3b8", "rgba(148,163,184,0.15)"))
    s1 = "<span style=\"display:inline-flex;align-items:center;gap:4px;padding:4px 14px;border-radius:50px;font-size:0.82rem;font-weight:700;background:"
    s2 = bg + ";color:" + fg + ";border:1px solid " + fg + "33;\">"
    s3 = advice + "</span>"
    return s1 + s2 + s3
@st.cache_data(ttl=900, show_spinner=False)
def generate_synthetic_ohlcv(ticker, days=252):
    seed = abs(hash(ticker)) % (2**32)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    base_price = 500 + (seed % 50) * 80
    drift = 0.0002 + ((seed % 7) - 3) * 0.00004
    ret = rng.normal(drift, 0.018, len(dates))
    close = base_price * np.exp(np.cumsum(ret))
    dv = np.abs(rng.normal(0.008, 0.005, len(dates)))
    high = close * (1 + dv * rng.uniform(0.5, 1.5, len(dates)))
    low = close * (1 - dv * rng.uniform(0.5, 1.5, len(dates)))
    op = low + (high - low) * rng.uniform(0.2, 0.8, len(dates))
    high = np.maximum(high, np.maximum(op, close))
    low = np.minimum(low, np.minimum(op, close))
    bv = max(int(base_price * 50), 100000)
    vol = np.abs(rng.normal(bv, bv * 0.3, len(dates))).astype(int)
    return pd.DataFrame({"Date": dates, "Open": np.round(op, 2), "High": np.round(high, 2), "Low": np.round(low, 2), "Close": np.round(close, 2), "Volume": vol})


def _extract_ohlcv_from_yf(frame, ticker):
    if frame is None or frame.empty: return None
    needed = ["Open", "High", "Low", "Close", "Volume"]
    result = {}
    if isinstance(frame.columns, pd.MultiIndex):
        for col in needed:
            for cp in [(col, ticker), (ticker, col)]:
                if cp in frame.columns: result[col] = frame[cp].values; break
            if col not in result:
                try: result[col] = frame.xs(col, axis=1, level=0).iloc[:, 0].values
                except: return None
    else:
        for col in needed:
            if col in frame.columns: result[col] = frame[col].values
            else: return None
    df = pd.DataFrame(result)
    df.index = pd.to_datetime(frame.index)
    df = df.dropna(subset=["Close"])
    return df.reset_index() if len(df) > 0 else None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_stock_ohlcv(ticker, period="1y"):
    ticker = ticker.upper().strip()
    if HAS_YFINANCE and yf is not None:
        try:
            frame = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
            ohlcv = _extract_ohlcv_from_yf(frame, ticker)
            if ohlcv is not None and len(ohlcv) >= 5: return ohlcv
        except: pass
    return generate_synthetic_ohlcv(ticker)

def build_candlestick_chart(ohlcv, ticker, show_ma20=True, show_ma50=True, show_bb=False, show_rsi=False):
    has_rsi = show_rsi and len(ohlcv) >= 15
    if has_rsi:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.18, 0.27], subplot_titles=("", "Volume", "RSI"))
    else:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.72, 0.28], subplot_titles=("", "Volume"))
    dates = ohlcv["Date"]
    cs = ohlcv["Close"]
    fig.add_trace(go.Candlestick(x=dates, open=ohlcv["Open"], high=ohlcv["High"], low=ohlcv["Low"], close=ohlcv["Close"], increasing_line_color="#10b981", decreasing_line_color="#ef4444", increasing_fillcolor="#10b981", decreasing_fillcolor="#ef4444", name="OHLC", hovertemplate="<b>%{x|%d %b %Y}</b><br>Open: Rp %{open:,.0f}<br>High: Rp %{high:,.0f}<br>Low: Rp %{low:,.0f}<br>Close: Rp %{close:,.0f}<extra></extra>"), row=1, col=1)
    if show_ma20 and len(cs) >= 20:
        fig.add_trace(go.Scatter(x=dates, y=cs.rolling(20, min_periods=5).mean(), mode="lines", name="MA20", line=dict(color="#eab308", width=1.5, dash="dot")), row=1, col=1)
    if show_ma50 and len(cs) >= 50:
        fig.add_trace(go.Scatter(x=dates, y=cs.rolling(50, min_periods=10).mean(), mode="lines", name="MA50", line=dict(color="#06b6d4", width=1.5, dash="dot")), row=1, col=1)
    if show_bb and len(cs) >= 20:
        m = cs.rolling(20, min_periods=5).mean()
        s = cs.rolling(20, min_periods=5).std()
        fig.add_trace(go.Scatter(x=dates, y=m+2*s, mode="lines", name="BB Up", line=dict(color="rgba(16,185,129,0.4)", width=1), showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=m-2*s, mode="lines", name="BB Lo", line=dict(color="rgba(16,185,129,0.4)", width=1), fill="tonexty", fillcolor="rgba(16,185,129,0.06)", showlegend=False, hoverinfo="skip"), row=1, col=1)
    vc = ["#10b981" if c >= o else "#ef4444" for c, o in zip(ohlcv["Close"], ohlcv["Open"])]
    vr = 3 if has_rsi else 2
    fig.add_trace(go.Bar(x=dates, y=ohlcv["Volume"], name="Volume", marker_color=vc, opacity=0.6), row=vr, col=1)
    if has_rsi:
        d = cs.diff()
        g = d.where(d > 0, 0.0).rolling(14, min_periods=5).mean()
        l = (-d.where(d < 0, 0.0)).rolling(14, min_periods=5).mean()
        rsi = 100 - (100 / (1 + g / (l + 1e-10)))
        fig.add_trace(go.Scatter(x=dates, y=rsi, mode="lines", name="RSI(14)", line=dict(color="#8b5cf6", width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.5)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(16,185,129,0.5)", row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)
    fig.update_layout(template="plotly_dark", height=520 if has_rsi else 420, margin=dict(t=25, b=25, l=16, r=16), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,29,0.6)", showlegend=True, legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)), xaxis=dict(gridcolor="rgba(151,177,214,0.08)", rangeselector=dict(buttons=[dict(count=7, label="1W", step="day", stepmode="backward"), dict(count=1, label="1M", step="month", stepmode="backward"), dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"), dict(count=1, label="1Y", step="year", stepmode="backward"), dict(label="ALL", step="all")], bgcolor="rgba(30,41,59,0.8)", activecolor="#10b981", font=dict(color="#e2e8f0", size=11))), yaxis=dict(title="Harga (Rp)", gridcolor="rgba(151,177,214,0.08)"))
    if has_rsi:
        fig.update_xaxes(gridcolor="rgba(151,177,214,0.08)", row=3, col=1)
        fig.update_yaxes(title="RSI", gridcolor="rgba(151,177,214,0.08)", row=3, col=1)
    return fig