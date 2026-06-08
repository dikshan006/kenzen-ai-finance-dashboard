import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import io

# Ingestion + analytics engines
from kenzen.ingestion import load_transactions
from kenzen.analytics.spending import compute_advanced_kpis, merchant_breakdown, category_breakdown
from kenzen.analytics.insights import generate_insights, generate_savings_tips
from kenzen.analytics.anomaly import detect_anomalies

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

/* Insight cards */
.insight-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    color: #e5e7eb;
    font-size: 0.92rem;
    line-height: 1.55;
}
.insight-card strong {
    color: #4fc3f7;
}

/* Badge for no anomalies */
.badge-ok {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
    border: 1px solid #10b981;
    border-radius: 10px;
    padding: 14px 20px;
    color: #a7f3d0;
    font-size: 0.95rem;
}

/* Savings tip cards */
.savings-card {
    background: linear-gradient(135deg, #1a1c2e 0%, #1e293b 100%);
    border: 1px solid #4f46e5;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    color: #e5e7eb;
    font-size: 0.92rem;
    line-height: 1.55;
}
.savings-card strong {
    color: #a78bfa;
}

/* Section divider */
.section-divider {
    border-top: 1px solid #1e3a5f;
    margin: 28px 0;
}

/* Data caption */
.data-caption {
    text-align: center;
    color: #6b7280;
    font-size: 0.8rem;
    padding: 20px 0 10px;
}
</style>
""", unsafe_allow_html=True)


# ---------- Ingestion (thin delegate, cached) ----------
@st.cache_data
def parse_csv(uploaded_file, spending_is_negative=True):
    return load_transactions(uploaded_file, spending_is_negative)


# ---------- Mock Data Generator (fallback only when no file or parse fails) ----------
def generate_mock_data():
    np.random.seed(42)
    days = 90
    dates = sorted([datetime.now().date() - timedelta(days=i) for i in range(days)])

    categories = ["Groceries", "Dining", "Transportation", "Entertainment", "Utilities", "Shopping"]
    merchants_by_cat = {
        "Groceries": ["Whole Foods", "Trader Joe's", "Safeway"],
        "Dining": ["Chipotle", "Starbucks", "Thai Palace", "Sushi Bar"],
        "Transportation": ["Uber", "Shell Gas", "Toyota Service"],
        "Entertainment": ["Netflix", "Cinema", "Concert Hall"],
        "Utilities": ["Electric Co", "Water Bill", "Internet Provider"],
        "Shopping": ["Amazon", "Target", "Gap"],
    }
    base_amount = {"Groceries": 90, "Dining": 35, "Transportation": 40,
                   "Entertainment": 20, "Utilities": 120, "Shopping": 70}

    txns = []
    for d in dates:
        for _ in range(np.random.randint(1, 4)):
            cat = np.random.choice(categories)
            merchant = np.random.choice(merchants_by_cat[cat])
            amt = max(5, abs(base_amount[cat] + np.random.normal(0, base_amount[cat] * 0.4)))
            txns.append({"Date": pd.to_datetime(d), "Merchant": merchant,
                         "Category": cat, "Amount": round(float(amt), 2), "Type": "Debit"})
    for d in dates[::14]:
        txns.append({"Date": pd.to_datetime(d), "Merchant": "Employer Payroll",
                      "Category": "Income", "Amount": 3500.00, "Type": "Credit"})
    return pd.DataFrame(txns).sort_values("Date").reset_index(drop=True)


# ---------- Header with Upload ----------
header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown("# KenZen AI Finance Dashboard")
    st.markdown(
        '<div style="color:#9ca3af; font-size:1.05rem; margin-top:-8px;">'
        'A dark-mode analytics cockpit for personal finance – live spending, anomalies, and insights.</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown("### Upload Your Data")
    uploaded_file = st.file_uploader("CSV File", type=["csv", "xlsx", "xls", "pdf"], label_visibility="collapsed")
    spending_is_negative = st.checkbox("Spending is negative", value=True,
                                       help="Check if expenses are negative numbers in your CSV")

if uploaded_file is not None:
    parsed_result, diagnostics = parse_csv(uploaded_file, spending_is_negative)

    if parsed_result is None:
        st.error(f"❌ Parsing failed: {diagnostics}")
        st.info("📊 Showing demo data instead. Please check your file format.")
        df = generate_mock_data()
        data_mode = "Demo Mode (parsing failed)"
    else:
        df = parsed_result
        data_mode = f"Live Data ({len(df)} transactions)"

        with st.expander("📊 Data Quality Report"):
            dq1, dq2, dq3, dq4 = st.columns(4)
            with dq1:
                st.metric("Rows Loaded", diagnostics["valid_rows"])
            with dq2:
                st.metric("Rows Dropped", diagnostics["dropped_rows"])
            with dq3:
                st.metric("Uncategorized", diagnostics["uncategorized_count"])
            with dq4:
                st.metric("Date Range", "✓" if diagnostics["date_range"] else "✗")

            st.markdown("**Column Mapping:**")
            for key, val in diagnostics["column_mapping"].items():
                st.text(f"  • {key}: {val}")
            if diagnostics["date_range"]:
                st.markdown(f"**Date Range:** {diagnostics['date_range']}")
            if diagnostics["drop_reasons"]:
                st.markdown("**Drop Reasons** (showing first 10):")
                for reason in diagnostics["drop_reasons"][:10]:
                    st.text(f"  • {reason}")
else:
    df = generate_mock_data()
    data_mode = "Demo Mode"

st.caption(f"🔹 {data_mode}")

# ---------- Compute all analytics from actual data ----------
kpis = compute_advanced_kpis(df)

# ---------- KPI Row 1: Core Financial Health ----------
col1, col2, col3, col4 = st.columns(4)

with col1:
    cf_color = "normal" if kpis["net_cash_flow"] >= 0 else "inverse"
    st.metric(
        "Net Cash Flow",
        f"${kpis['net_cash_flow']:,.2f}",
        delta=f"{'Surplus' if kpis['net_cash_flow'] >= 0 else 'Deficit'}",
        delta_color=cf_color,
    )

with col2:
    st.metric(
        "Total Spending",
        f"${kpis['total_spending']:,.2f}",
        delta=f"{kpis['monthly_delta']:+.1f}% vs prior period" if kpis['prev30_spend'] > 0 else None,
    )

with col3:
    st.metric(
        "Total Income",
        f"${kpis['total_income']:,.2f}",
        delta=f"{kpis['ie_ratio']:.2f}x income/expense" if kpis['total_spending'] > 0 else None,
    )

with col4:
    st.metric(
        "Transactions",
        f"{kpis['tx_count']}",
        delta=f"${kpis['avg_txn']:,.2f} avg/txn",
    )

# ---------- KPI Row 2: Deeper Metrics ----------
col5, col6, col7, col8 = st.columns(4)

with col5:
    st.metric("Monthly Burn Rate", f"${kpis['monthly_burn']:,.2f}")

with col6:
    st.metric("Largest Expense",
              f"${kpis['largest_amount']:,.2f}",
              delta=kpis['largest_merchant'] if kpis['largest_merchant'] else None,
              delta_color="off")

with col7:
    st.metric("Top Merchant",
              kpis['top_merchant'] or "—",
              delta=f"${kpis['top_merchant_spend']:,.2f}" if kpis['top_merchant'] else None,
              delta_color="off")

with col8:
    st.metric("Most Visited",
              kpis['most_visited'] or "—",
              delta=f"{kpis['most_visited_count']} visits" if kpis['most_visited'] else None,
              delta_color="off")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Spending Analysis (existing charts, real data) ----------
st.markdown("## Spending Analysis")

left, right = st.columns(2)

with left:
    debit_df = df[df["Type"] == "Debit"]
    by_cat = debit_df.groupby("Category")["Amount"].sum().sort_values(ascending=True)

    fig_cat = go.Figure(
        data=[go.Bar(
            x=by_cat.values, y=by_cat.index, orientation="h",
            marker=dict(color="#38bdf8"),
            text=[f"${v:,.2f}" for v in by_cat.values], textposition="auto",
        )]
    )
    fig_cat.update_layout(
        title="Spending by Category", xaxis_title="Amount ($)", yaxis_title="Category",
        height=380, plot_bgcolor="#020617", paper_bgcolor="#020617",
        font=dict(color="#e5e7eb"), xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#020617"), margin=dict(l=60, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with right:
    daily = debit_df.groupby(debit_df["Date"].dt.date)["Amount"].sum().reset_index()
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(
        x=daily["Date"], y=daily["Amount"], mode="lines+markers",
        line=dict(width=2, color="#38bdf8"), marker=dict(size=4),
    ))
    fig_daily.update_layout(
        title="Daily Spending Trend", xaxis_title="Date", yaxis_title="Amount ($)",
        height=380, plot_bgcolor="#020617", paper_bgcolor="#020617",
        font=dict(color="#e5e7eb"), xaxis=dict(gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"), margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_daily, use_container_width=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Category Breakdown Table ----------
st.markdown("## Category Breakdown")
cat_df = category_breakdown(df)
if not cat_df.empty:
    display_cat = cat_df.copy()
    display_cat["Total Spend"] = display_cat["Total Spend"].apply(lambda x: f"${x:,.2f}")
    display_cat["Avg Spend"] = display_cat["Avg Spend"].apply(lambda x: f"${x:,.2f}")
    display_cat["% of Spending"] = display_cat["% of Spending"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_cat, use_container_width=True, hide_index=True)
else:
    st.info("No spending data to display.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Merchant Analytics Table ----------
st.markdown("## Merchant Analytics")
merch_df = merchant_breakdown(df)
if not merch_df.empty:
    display_merch = merch_df.head(20).copy()
    display_merch["Total Spend"] = display_merch["Total Spend"].apply(lambda x: f"${x:,.2f}")
    display_merch["Avg Spend"] = display_merch["Avg Spend"].apply(lambda x: f"${x:,.2f}")
    display_merch["% of Budget"] = display_merch["% of Budget"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_merch, use_container_width=True, hide_index=True)
else:
    st.info("No merchant data to display.")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Anomaly Detection ----------
st.markdown("## Anomaly Detection")
anoms = detect_anomalies(df)
if anoms.empty:
    st.markdown(
        '<div class="badge-ok">No anomalies detected. Your spending is within normal ranges.</div>',
        unsafe_allow_html=True,
    )
else:
    st.dataframe(anoms, use_container_width=True, hide_index=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- KenZen Insights (data-driven, personalized) ----------
st.markdown("## KenZen Insights")
insights = generate_insights(df, kpis)
c1, c2 = st.columns(2)
for i, text in enumerate(insights):
    col = c1 if i % 2 == 0 else c2
    with col:
        st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Savings Opportunities ----------
st.markdown("## Savings Opportunities")
tips = generate_savings_tips(df, kpis)
if tips:
    s1, s2 = st.columns(2)
    for i, tip in enumerate(tips):
        col = s1 if i % 2 == 0 else s2
        with col:
            st.markdown(f'<div class="savings-card">💡 {tip}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="badge-ok">Your spending patterns look healthy — no immediate savings opportunities flagged.</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------- Recent Transactions ----------
st.markdown("## Recent Transactions")
recent = df.sort_values("Date", ascending=False).head(20).copy()
recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
recent["Amount"] = recent["Amount"].apply(lambda x: f"${x:,.2f}")
st.dataframe(
    recent[["Date", "Merchant", "Category", "Amount", "Type"]],
    use_container_width=True, hide_index=True,
)

st.markdown(
    '<div class="data-caption">KenZen AI Finance Dashboard · Real-time analysis powered by Streamlit</div>',
    unsafe_allow_html=True,
)
