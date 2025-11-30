import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="KenZen AI Finance Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------
# Dark theme styling
# -------------------------------------------------
st.markdown(
    """
    <style>
    /* App background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at top left, #111827 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    [data-testid="stSidebar"] {
        background: #020617;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #9ca3af;
        font-weight: 500;
    }

    /* Typography */
    h1, h2, h3 {
        color: #e5e7eb !important;
    }
    .markdown-text-container, .stMarkdown, .stText {
        color: #e5e7eb !important;
    }

    /* Dataframe */
    .stDataFrame, .stDataFrame table {
        color: #e5e7eb !important;
        background-color: #020617 !important;
    }

    /* Divider spacing */
    hr {
        border-color: #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Generate mock transaction data
# -------------------------------------------------
@st.cache_data
def generate_mock_data():
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=x) for x in range(90)]
    dates.reverse()

    categories = ["Groceries", "Dining", "Transportation", "Entertainment", "Utilities", "Shopping"]
    merchants = {
        "Groceries": ["Whole Foods", "Trader Joe's", "Safeway"],
        "Dining": ["Chipotle", "Starbucks", "Thai Palace", "Sushi Bar"],
        "Transportation": ["Uber", "Shell Gas", "Toyota Service"],
        "Entertainment": ["Netflix", "Cinema", "Concert"],
        "Utilities": ["PG&E", "Comcast", "Water Bill"],
        "Shopping": ["Amazon", "Target", "Gap"],
    }

    transactions = []
    for date in dates:
        num_transactions = np.random.randint(1, 4)
        for _ in range(num_transactions):
            category = np.random.choice(categories)
            merchant = np.random.choice(merchants[category])

            base_amounts = {
                "Groceries": 80,
                "Dining": 25,
                "Transportation": 40,
                "Entertainment": 15,
                "Utilities": 100,
                "Shopping": 60,
            }

            amount = base_amounts[category] + np.random.normal(0, base_amounts[category] * 0.3)
            amount = max(5, abs(amount))

            transaction_type = "Credit" if np.random.random() < 0.1 else "Debit"

            transactions.append(
                {
                    "Date": date,
                    "Merchant": merchant,
                    "Category": category,
                    "Amount": round(amount, 2),
                    "Type": transaction_type,
                }
            )

    # Add some income transactions
    for i in range(0, 90, 30):
        if i < len(dates):
            transactions.append(
                {
                    "Date": dates[i],
                    "Merchant": "Employer Deposit",
                    "Category": "Income",
                    "Amount": 3000,
                    "Type": "Credit",
                }
            )

    df = pd.DataFrame(transactions)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


df = generate_mock_data()

# -------------------------------------------------
# KPI calculations
# -------------------------------------------------
def calculate_kpis():
    base_balance = 5000
    credits = df[df["Type"] == "Credit"]["Amount"].sum()
    debits = df[df["Type"] == "Debit"]["Amount"].sum()
    current_balance = base_balance + credits - debits

    current_month = df[df["Date"].dt.month == datetime.now().month]
    monthly_spending = current_month[current_month["Type"] == "Debit"]["Amount"].sum()

    last_month_date = datetime.now() - timedelta(days=30)
    prev_month = df[(df["Date"].dt.month == last_month_date.month) & (df["Type"] == "Debit")]
    prev_spending = prev_month["Amount"].sum()

    spending_change = ((monthly_spending - prev_spending) / prev_spending * 100) if prev_spending > 0 else 0

    total_savings = credits - debits
    transaction_count = len(df)

    return {
        "balance": current_balance,
        "monthly_spending": monthly_spending,
        "spending_change": spending_change,
        "total_savings": total_savings,
        "transaction_count": transaction_count,
        "total_credits": credits,
        "total_debits": debits,
    }


kpis = calculate_kpis()

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("## KenZen AI Finance Dashboard")
st.markdown("A dark-mode analytics cockpit for personal finance – live spending, anomalies, and insights.")

# -------------------------------------------------
# KPI Row
# -------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Account Balance",
        f"${kpis['balance']:,.2f}",
        delta=f"+${kpis['total_credits']:,.0f}" if kpis["total_credits"] > 0 else None,
    )

with col2:
    st.metric(
        "Monthly Spending",
        f"${kpis['monthly_spending']:,.2f}",
        delta=f"{kpis['spending_change']:+.1f}%" if kpis["spending_change"] != 0 else None,
        delta_color="inverse",
    )

with col3:
    st.metric(
        "Total Savings",
        f"${kpis['total_savings']:,.2f}",
        delta=(
            f"{(kpis['total_savings'] / kpis['total_credits'] * 100):.1f}% of income"
            if kpis["total_credits"] > 0
            else None
        ),
    )

with col4:
    avg_txn = kpis["total_debits"] / kpis["transaction_count"] if kpis["transaction_count"] > 0 else 0
    st.metric(
        "Transactions",
        kpis["transaction_count"],
        delta=f"${avg_txn:.2f} avg/txn",
    )

st.divider()

# -------------------------------------------------
# Charts
# -------------------------------------------------
st.markdown("### Spending Analysis")

chart_col1, chart_col2 = st.columns(2)

# Spending by Category
with chart_col1:
    spending_by_category = (
        df[df["Type"] == "Debit"].groupby("Category")["Amount"].sum().sort_values(ascending=True)
    )

    fig_category = go.Figure(
        data=[
            go.Bar(
                y=spending_by_category.index,
                x=spending_by_category.values,
                orientation="h",
                marker=dict(color="#38bdf8", line=dict(color="#0ea5e9", width=2)),
                text=[f"${x:.2f}" for x in spending_by_category.values],
                textposition="auto",
            )
        ]
    )
    fig_category.update_layout(
        title="Spending by Category",
        xaxis_title="Amount ($)",
        yaxis_title="Category",
        height=400,
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
    )
    st.plotly_chart(fig_category, use_container_width=True)

# Daily Spending Trend
with chart_col2:
    daily_spending = df[df["Type"] == "Debit"].groupby(df["Date"].dt.date)["Amount"].sum()

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=daily_spending.index,
            y=daily_spending.values,
            mode="lines+markers",
            name="Daily Spending",
            line=dict(color="#38bdf8", width=3),
            marker=dict(size=6, color="#0ea5e9"),
            fill="tozeroy",
            fillcolor="rgba(56, 189, 248, 0.15)",
        )
    )
    fig_trend.update_layout(
        title="Daily Spending Trend",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        height=400,
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e5e7eb"),
        showlegend=False,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# -------------------------------------------------
# Anomaly Detection
# -------------------------------------------------
st.markdown("### Anomaly Detection")

def detect_anomalies():
    anomalies = []
    debit_df = df[df["Type"] == "Debit"].copy()

    for category in debit_df["Category"].unique():
        category_transactions = debit_df[debit_df["Category"] == category]
        mean_amount = category_transactions["Amount"].mean()
        threshold = mean_amount * 2

        anomaly_txns = category_transactions[category_transactions["Amount"] > threshold]

        for _, txn in anomaly_txns.iterrows():
            anomalies.append(
                {
                    "Date": txn["Date"].strftime("%Y-%m-%d"),
                    "Merchant": txn["Merchant"],
                    "Category": txn["Category"],
                    "Amount": f"${txn['Amount']:.2f}",
                    "Threshold": f"${threshold:.2f}",
                    "Risk Level": "High" if txn["Amount"] > threshold * 1.5 else "Medium",
                }
            )

    return pd.DataFrame(anomalies) if anomalies else None


anomalies_df = detect_anomalies()

if anomalies_df is not None and len(anomalies_df) > 0:
    st.warning(f"Found {len(anomalies_df)} anomalous transactions (2x category average)")
    st.dataframe(anomalies_df, use_container_width=True, hide_index=True)
else:
    st.success("No anomalies detected. Your spending is within normal ranges.")

st.divider()

# -------------------------------------------------
# AI-Style Insights
# -------------------------------------------------
st.markdown("### KenZen Insights")

def generate_insights():
    spending_by_cat = df[df["Type"] == "Debit"].groupby("Category")["Amount"].sum()
    biggest_category = spending_by_cat.idxmax()
    biggest_amount = spending_by_cat.max()

    total_flow = kpis["total_credits"] + kpis["total_debits"]
    credit_ratio = (
        kpis["total_credits"] / total_flow * 100 if total_flow > 0 else 0
    )

    avg_dining = (
        df[(df["Category"] == "Dining") & (df["Type"] == "Debit")]["Amount"].mean()
    )
    potential_savings = avg_dining * 0.3 * 30 if avg_dining > 0 else 0

    recent_spending = df[
        (df["Date"] > datetime.now() - timedelta(days=14)) & (df["Type"] == "Debit")
    ]["Amount"].sum()
    earlier_spending = df[
        (df["Date"] > datetime.now() - timedelta(days=28))
        & (df["Date"] <= datetime.now() - timedelta(days=14))
        & (df["Type"] == "Debit")
    ]["Amount"].sum()
    trend = "increasing" if recent_spending > earlier_spending else "decreasing"

    insights = [
        f"Your highest spending category is **{biggest_category}** at **${biggest_amount:.2f}**. This is the primary lever for budget control.",
        f"Income vs spend ratio is **{credit_ratio:.1f}%**. Values below ~40% usually indicate aggressive spending relative to income.",
        f"If you cut dining by 30%, you could free up roughly **${potential_savings:.2f}** per month for savings or investing.",
        f"Spending trend over the last two weeks is **{trend}**. Use this to decide whether to tighten or relax short-term budgets.",
        f"Net savings from all transactions over the last 90 days is **${kpis['total_savings']:.2f}**. Automating transfers to a separate savings account would lock this in.",
    ]

    return insights


insights = generate_insights()

col1, col2 = st.columns(2)

with col1:
    for insight in insights[:3]:
        st.info(insight)

with col2:
    for insight in insights[3:]:
        st.info(insight)

st.divider()

# -------------------------------------------------
# Recent Transactions
# -------------------------------------------------
st.markdown("### Recent Transactions")
recent_tx = df.sort_values("Date", ascending=False).head(20).copy()
recent_tx["Date"] = recent_tx["Date"].dt.strftime("%Y-%m-%d")
recent_tx["Amount"] = recent_tx["Amount"].apply(lambda x: f"${x:.2f}")

st.dataframe(
    recent_tx[["Date", "Merchant", "Category", "Amount", "Type"]],
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")
st.markdown("KenZen AI Finance Dashboard · Real-time analysis powered by Streamlit")
