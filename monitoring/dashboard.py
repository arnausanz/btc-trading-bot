# monitoring/dashboard.py
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from sqlalchemy import text

sys.path.append(".")

from core.db.session import SessionLocal
from core.db.models import CandleDB

st.set_page_config(
    page_title="BTC Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_candles(timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
    session = SessionLocal()
    try:
        rows = (
            session.query(CandleDB)
            .filter_by(symbol="BTC/USDT", timeframe=timeframe)
            .order_by(CandleDB.timestamp.desc())
            .limit(limit)
            .all()
        )
        return pd.DataFrame([{
            "timestamp": r.timestamp, "open": r.open, "high": r.high,
            "low": r.low, "close": r.close, "volume": r.volume,
        } for r in rows]).sort_values("timestamp").reset_index(drop=True)
    finally:
        session.close()


def load_db_stats() -> dict:
    session = SessionLocal()
    try:
        result = session.execute(text("""
            SELECT timeframe, COUNT(*) as total, MIN(timestamp) as primera, MAX(timestamp) as ultima
            FROM candles GROUP BY timeframe ORDER BY timeframe
        """))
        return {row.timeframe: {"total": row.total, "primera": row.primera, "ultima": row.ultima}
                for row in result}
    finally:
        session.close()


def render_candlestick(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(
        x=df["timestamp"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    )])
    fig.update_layout(
        title="BTC/USDT", xaxis_title="Date", yaxis_title="Price (USDT)",
        template="plotly_dark", height=500, xaxis_rangeslider_visible=False,
    )
    return fig


def render_volume(df: pd.DataFrame) -> go.Figure:
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])]
    fig = go.Figure(data=[go.Bar(x=df["timestamp"], y=df["volume"], marker_color=colors)])
    fig.update_layout(title="Volume", template="plotly_dark", height=200, showlegend=False)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)
limit = st.sidebar.slider("Number of candles", 50, 500, 200)
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 BTC Trading Bot Dashboard")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ── DB Stats ──────────────────────────────────────────────────────────────────
st.subheader("📊 Database")
stats = load_db_stats()
cols = st.columns(len(stats))
for col, (tf, s) in zip(cols, stats.items()):
    col.metric(label=f"Candles {tf}", value=f"{s['total']:,}", delta=f"until {s['ultima'].strftime('%Y-%m-%d')}")

# ── Price chart ───────────────────────────────────────────────────────────────
st.subheader("🕯️ Price chart")
df = load_candles(timeframe=timeframe, limit=limit)

if df.empty:
    st.warning("No data available.")
else:
    st.plotly_chart(render_candlestick(df), use_container_width=True)
    st.plotly_chart(render_volume(df), use_container_width=True)

    st.subheader("📐 Technical indicators (last candle)")
    last = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current price", f"{last['close']:,.2f} USDT")
    c2.metric("High", f"{last['high']:,.2f} USDT")
    c3.metric("Low", f"{last['low']:,.2f} USDT")
    c4.metric("Volume", f"{last['volume']:,.2f}")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    st.rerun()