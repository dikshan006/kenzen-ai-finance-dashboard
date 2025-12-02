import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ---------- Page config ----------
st.set_page_config(
    page_title="KenZen AI Finance Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Global Styles ----------
st.markdown("""
<style>
/* Make the whole app dark navy, ignore Streamlit theme */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0b1221 !important;
    color: #e5e7eb !important;
}

/* Remove white header bar */
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0) !important;
}

/* Sidebar (if used) */
[data-testid="stSidebar"] {
    background-color: #08111e !important;
}

/* Dataframes / tables */
[data-testid="stDataFrame"] {
    background-color: #020617 !important;
    color: #e5e7eb !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-size: 2.0rem !important;
    font-weight: 700 !important;
    color: #4fc3f7 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.8rem !important;
    color: #9ca3af !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


# ---------- Mock Data Generator ----------
def generate_mock_data():
    np.random.seed(42)
    days = 90
    dates = [datetime.now().date() - timedelta(days=i) for i in range(days)]
    dates = sorted(dates)

    categories = ["Groceries", "Dining", "Transportation",
                  "Entertainment", "Utilities", "Shopping"]
    merchants_by_cat = {
        "Groceries": ["Whole Foods", "Trader Joe's", "Safeway"],
        "Dining": ["Chipotle", "Starbucks", "Thai Palace", "Sushi Bar"],
        "Transportation": ["Uber", "Shell Gas", "Toyota Service"],
        "Entertainment": ["Netflix", "Cinema", "Concert Hall"],
        "Utilities": ["Electric Co", "Water Bill", "Internet Provider"],
        "Shopping": ["Amazon", "Target", "Gap"],
    }

    base_amount = {
        "Groceries": 90,
        "Dining": 35,
        "Transportation": 40,
        "Entertainment": 20,
        "Utilities": 120,
        "Shopping": 70,
    }

    txns = []

    for d in dates:
        # 1–4 transactions per day
        n = np.random.randint(1, 4)
        for _ in range(n):
            cat = np.random.choice(categories)
            merchant = np.random.choice(merchants_by_cat[cat])
            amt = base_amount[cat] + np.random.normal(0, base_amount[cat] * 0.4)
            amt = max(5, abs(amt))
            txn_type = "Debit"

            txns.append(
                {
                    "Date": pd.to_datetime(d),
                    "Merchant": merchant,
                    "Category": cat,
                    "Amount": round(float(amt), 2),
                    "Type": txn_type,
                }
            )

    # Add income deposits roughly every 2 weeks
    for d in dates[::14]:
        txns.append(
            {
                "Date": pd.to_datetime(d),
                "Merchant": "Employer Payroll",
                "Category": "Income",
                "Amount": 3500.00,
                "Type": "Credit",
            }
        )

    df = pd.DataFrame(txns)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


df = generate_mock_data()

# ---------- KPI Calculations ----------
def compute_kpis(df: pd.DataFrame):
    credits = df[df["Type"] == "Credit"]["Amount"].sum()
    debits = df[df["Type"] == "Debit"]["Amount"].sum()

    # Use a synthetic starting balance to get nice-looking numbers
    starting_balance = 3000.0
    current_balance = starting_balance + credits - debits

    # Last 30 days vs previous 30 days
    today = df["Date"].max()
    last_30_start = today - timedelta(days=29)
    prev_30_start = last_30_start - timedelta(days=30)

    last_30 = df[(df["Date"] >= last_30_start) & (df["Type"] == "Debit")]
    prev_30 = df[
        (df["Date"] >= prev_30_start)
        & (df["Date"] < last_30_start)
        & (df["Type"] == "Debit")
    ]

    monthly_spend = last_30["Amount"].sum()
    prev_spend = prev_30["Amount"].sum()
    if prev_spend > 0:
        monthly_delta = (monthly_spend - prev_spend) / prev_spend * 100
    else:
        monthly_delta = 0.0

    net_savings_90 = credits - debits
    tx_count = len(df)

    # Income vs spend ratio (credits vs total volume)
    volume = credits + debits
    income_ratio = (credits / volume * 100) if volume > 0 else 0.0

    avg_per_txn = debits / tx_count if tx_count > 0 else 0.0

    return {
        "credits": credits,
        "debits": debits,
        "current_balance": current_balance,
        "monthly_spend": monthly_spend,
        "monthly_delta": monthly_delta,
        "net_savings_90": net_savings_90,
        "tx_count": tx_count,
        "income_ratio": income_ratio,
        "avg_per_txn": avg_per_txn,
    }


kpis = compute_kpis(df)

# ---------- Header ----------
st.markdown("# KenZen AI Finance Dashboard")
st.markdown(
    '<div class="subheadline">A dark-mode analytics cockpit for personal finance – live spending, anomalies, and insights.</div>',
    unsafe_allow_html=True,
)

# ---------- KPI Row ----------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Account Balance",
        f"${kpis['current_balance']:,.2f}",
        delta=f"+${(kpis['current_balance']-3000):,.0f}",
    )

with col2:
    st.metric(
        "Monthly Spending",
        f"${kpis['monthly_spend']:,.2f}",
        delta=f"{kpis['monthly_delta']:+.1f}%",
    )

with col3:
    st.metric(
        "Total Savings (90 days)",
        f"${kpis['net_savings_90']:,.2f}",
        delta=f"{(kpis['net_savings_90'] / max(kpis['debits'],1))*100:+.1f}% of spend",
    )

with col4:
    st.metric(
        "Transactions",
        f"{kpis['tx_count']}",
        delta=f"${kpis['avg_per_txn']:,.2f} avg/txn",
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Spending Analysis ----------
st.markdown("## Spending Analysis")

left, right = st.columns(2)

# Spending by Category
with left:
    debit_df = df[df["Type"] == "Debit"]
    by_cat = (
        debit_df.groupby("Category")["Amount"].sum().sort_values(ascending=True)
    )

    fig_cat = go.Figure(
        data=[
            go.Bar(
                x=by_cat.values,
                y=by_cat.index,
                orientation="h",
                marker=dict(color="#38bdf8"),
                text=[f"${v:,.2f}" for v in by_cat.values],
                textposition="auto",
            )
        ]
    )
    fig_cat.update_layout(
        title="Spending by Category",
        xaxis_title="Amount ($)",
        yaxis_title="Category",
        height=380,
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font=dict(color="#e5e7eb"),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#020617"),
        margin=dict(l=60, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

# Daily Spending Trend
with right:
    daily = (
        debit_df.groupby(df["Date"].dt.date)["Amount"].sum().reset_index()
    )
    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Scatter(
            x=daily["Date"],
            y=daily["Amount"],
            mode="lines+markers",
            line=dict(width=2, color="#38bdf8"),
            marker=dict(size=4),
        )
    )
    fig_daily.update_layout(
        title="Daily Spending Trend",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        height=380,
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        font=dict(color="#e5e7eb"),
        xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_daily, use_container_width=True)

# ---------- Anomaly Detection ----------
st.markdown("## Anomaly Detection")

def detect_anomalies(df: pd.DataFrame):
    debits = df[df["Type"] == "Debit"].copy()
    anomalies = []

    for cat in debits["Category"].unique():
        sub = debits[debits["Category"] == cat]
        mean = sub["Amount"].mean()
        threshold = mean * 3.0  # aggressive threshold so normal demo data is "clean"
        big = sub[sub["Amount"] > threshold]

        for _, row in big.iterrows():
            anomalies.append(
                {
                    "Date": row["Date"].date().isoformat(),
                    "Merchant": row["Merchant"],
                    "Category": row["Category"],
                    "Amount": f"${row['Amount']:,.2f}",
                    "Threshold": f"${threshold:,.2f}",
                }
            )

    return pd.DataFrame(anomalies) if anomalies else None


anoms = detect_anomalies(df)

if anoms is None or anoms.empty:
    st.markdown(
        '<div class="badge-ok">No anomalies detected. Your spending is within normal ranges.</div>',
        unsafe_allow_html=True,
    )
else:
    st.dataframe(anoms, use_container_width=True, hide_index=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- KenZen Insights ----------
st.markdown("## KenZen Insights")

debit_df = df[df["Type"] == "Debit"]
by_cat_full = debit_df.groupby("Category")["Amount"].sum()
top_cat = by_cat_full.idxmax()
top_cat_amt = by_cat_full.max()

# Recent vs earlier spending
today = df["Date"].max()
recent_14 = debit_df[debit_df["Date"] >= today - timedelta(days=13)]["Amount"].sum()
earlier_14 = debit_df[
    (debit_df["Date"] < today - timedelta(days=13))
    & (debit_df["Date"] >= today - timedelta(days=27))
]["Amount"].sum()
trend_label = "increasing" if recent_14 > earlier_14 else "decreasing"

dining_mean = (
    debit_df[debit_df["Category"] == "Dining"]["Amount"].mean()
    if "Dining" in debit_df["Category"].unique()
    else 0.0
)
dining_savings = dining_mean * 0.3 * 30 if dining_mean > 0 else 0.0

insights = [
    f"Your highest spending category is <strong>{top_cat}</strong> at <strong>${top_cat_amt:,.2f}</strong>. This is the primary lever for budget control.",
    f"Spending trend over the last two weeks is <strong>{trend_label}</strong>. Use this to decide whether to tighten or relax short-term budgets.",
    f"Income vs spend ratio is <strong>{kpis['income_ratio']:.1f}%</strong>. Values below ~40% usually indicate aggressive spending relative to income.",
    f"Net savings from all transactions over the last 90 days is <strong>${kpis['net_savings_90']:,.2f}</strong>. Automating transfers to a separate savings account would lock this in.",
    f"If you cut dining by 30%, you could free up roughly <strong>${dining_savings:,.2f}</strong> per month for savings or investing."
]

c1, c2 = st.columns(2)
for i, text in enumerate(insights):
    col = c1 if i % 2 == 0 else c2
    with col:
        st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Recent Transactions ----------
st.markdown("## Recent Transactions")

recent = df.sort_values("Date", ascending=False).head(20).copy()
recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
recent["Amount"] = recent["Amount"].apply(lambda x: f"${x:,.2f}")

st.dataframe(
    recent[["Date", "Merchant", "Category", "Amount", "Type"]],
    use_container_width=True,
    hide_index=True,
)

st.markdown(
    '<div class="data-caption">KenZen AI Finance Dashboard · Real-time analysis powered by Streamlit</div>',
    unsafe_allow_html=True,
)
