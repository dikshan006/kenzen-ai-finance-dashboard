import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="KenZen Finance",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== DARK FINTECH THEME ====================
dark_theme_css = """
<style>
    [data-testid="stAppViewContainer"] {
        background: #0f1729;
        color: #e5e7eb;
    }
    
    [data-testid="stSidebar"] {
        background: #0f1729;
    }
    
    .header-title {
        color: #ffffff;
        font-size: 48px;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    
    .header-subtitle {
        color: #cbd5e1;
        font-size: 16px;
        margin: 12px 0 0 0;
        font-weight: 400;
    }
    
    .section-title {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        margin: 40px 0 24px 0;
    }
    
    .kpi-value {
        color: #06b6d4;
        font-size: 36px;
        font-weight: 700;
    }
    
    .kpi-label {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 500;
    }
    
    .kpi-delta {
        color: #10b981;
        font-size: 12px;
        margin-top: 8px;
        font-weight: 500;
    }
    
    .kpi-delta-negative {
        color: #ef4444;
    }
    
    .anomaly-box {
        background: #0d3b2e;
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 16px;
        color: #6ee7b7;
        font-size: 15px;
        line-height: 1.6;
    }
    
    .insight-card {
        background: #1e3a5f;
        border: 1px solid #1e40af;
        border-radius: 8px;
        padding: 20px;
        color: #60a5fa;
        font-size: 14px;
        line-height: 1.6;
        height: 100%;
    }
    
    .insight-card-text {
        color: #60a5fa;
        line-height: 1.6;
    }
    
    [data-testid="stMetric"] {
        background: transparent;
        border: none;
        padding: 0;
    }
    
    [data-testid="stMetricDeltaText"] {
        display: none;
    }
</style>
"""

st.markdown(dark_theme_css, unsafe_allow_html=True)

# ==================== MOCK DATA GENERATION ====================

@st.cache_data
def generate_mock_transactions():
    """Generate realistic mock transaction data"""
    np.random.seed(42)
    
    merchants = {
        'Groceries': ['Whole Foods', 'Trader Joe', 'Safeway', 'Kroger', 'Costco', 'Gap'],
        'Utilities': ['Con Edison', 'Verizon', 'Comcast', 'water dept'],
        'Shopping': ['Gap', 'Amazon', 'Target', 'Nike', 'Best Buy'],
        'Transportation': ['Uber', 'Toyota Service', 'Shell Gas', 'Lyft'],
        'Dining': ['Blue Hill', 'Carbone', 'Nobu', 'Restaurant'],
        'Entertainment': ['Netflix', 'Cinema', 'Spotify']
    }
    
    amount_ranges = {
        'Groceries': (60, 150),
        'Utilities': (80, 200),
        'Shopping': (50, 150),
        'Transportation': (30, 100),
        'Dining': (40, 120),
        'Entertainment': (15, 50)
    }
    
    transactions = []
    base_date = datetime.now() - timedelta(days=60)
    
    for day_offset in range(60):
        current_date = base_date + timedelta(days=day_offset)
        num_transactions = np.random.poisson(3)
        
        for _ in range(num_transactions):
            category = np.random.choice(list(merchants.keys()))
            merchant = np.random.choice(merchants[category])
            min_amt, max_amt = amount_ranges[category]
            amount = np.random.uniform(min_amt, max_amt)
            
            transactions.append({
                'Date': current_date.strftime('%Y-%m-%d'),
                'Merchant': merchant,
                'Category': category,
                'Amount': f"${amount:.2f}",
                'Type': 'Debit'
            })
    
    return pd.DataFrame(transactions).sort_values('Date', ascending=False).reset_index(drop=True)

@st.cache_data
def get_spending_by_category():
    """Get spending totals by category"""
    return {
        'Groceries': 2547.96,
        'Utilities': 2097.52,
        'Shopping': 1956.27,
        'Transportation': 1041.70,
        'Dining': 807.26,
        'Entertainment': 441.96
    }

@st.cache_data
def get_daily_spending():
    """Get daily spending data"""
    base_date = datetime.now() - timedelta(days=60)
    dates = []
    amounts = []
    
    np.random.seed(42)
    for day_offset in range(60):
        dates.append((base_date + timedelta(days=day_offset)).strftime('%Y-%m-%d'))
        amounts.append(np.random.uniform(50, 300))
    
    return dates, amounts

# ==================== MAIN APP ====================

# Header
st.markdown('''
<div style="margin-bottom: 48px;">
    <p class="header-title">KenZen AI Finance Dashboard</p>
    <p class="header-subtitle">A dark-mode analytics cockpit for personal finance – live spending, anomalies, and insights.</p>
</div>
''', unsafe_allow_html=True)

# KPI Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Account Balance", "$6,466.99", "+$10,360")

with col2:
    st.metric("Monthly Spending", "$3,051.93", "+3.5%")

with col3:
    st.metric("Total Savings", "$1,466.99", "+14.2% of income")

with col4:
    st.metric("Transactions", "196", "+$45.37 avg/txn")

st.markdown("<br>", unsafe_allow_html=True)

# Spending Analysis
st.markdown('<p class="section-title">Spending Analysis</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

spending_data = get_spending_by_category()

with col1:
    # Spending by Category
    fig_category = go.Figure(data=[
        go.Bar(
            y=list(spending_data.keys()),
            x=list(spending_data.values()),
            orientation='h',
            marker=dict(
                color='#06b6d4',
                line=dict(color='#0f3a5f', width=2)
            ),
            text=[f'${v:.2f}' for v in spending_data.values()],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>%{x:$,.2f}<extra></extra>'
        )
    ])
    
    fig_category.update_layout(
        title={
            'text': 'Spending by Category',
            'x': 0.0,
            'xanchor': 'left',
            'font': {'color': '#cbd5e1', 'size': 14}
        },
        plot_bgcolor='#0f1729',
        paper_bgcolor='#0f1729',
        font=dict(color='#94a3b8', size=12),
        hovermode='y unified',
        height=350,
        margin=dict(l=120, r=50, t=50, b=40),
        xaxis=dict(
            title='Amount ($)',
            titlefont=dict(color='#94a3b8'),
            tickfont=dict(color='#94a3b8'),
            gridcolor='#1e293b'
        ),
        yaxis=dict(
            tickfont=dict(color='#94a3b8')
        )
    )
    st.plotly_chart(fig_category, use_container_width=True, config={'displayModeBar': False})

with col2:
    # Daily Spending Trend
    dates, amounts = get_daily_spending()
    
    fig_daily = go.Figure(data=[
        go.Scatter(
            x=dates,
            y=amounts,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#06b6d4', width=2),
            fillcolor='rgba(6, 182, 212, 0.1)',
            hovertemplate='<b>%{x}</b><br>$%{y:.2f}<extra></extra>'
        )
    ])
    
    fig_daily.update_layout(
        title={
            'text': 'Daily Spending Trend',
            'x': 0.0,
            'xanchor': 'left',
            'font': {'color': '#cbd5e1', 'size': 14}
        },
        plot_bgcolor='#0f1729',
        paper_bgcolor='#0f1729',
        font=dict(color='#94a3b8', size=12),
        hovermode='x unified',
        height=350,
        margin=dict(l=50, r=50, t=50, b=40),
        xaxis=dict(
            title='Date',
            titlefont=dict(color='#94a3b8'),
            tickfont=dict(color='#94a3b8'),
            gridcolor='#1e293b'
        ),
        yaxis=dict(
            title='Amount ($)',
            titlefont=dict(color='#94a3b8'),
            tickfont=dict(color='#94a3b8'),
            gridcolor='#1e293b'
        )
    )
    st.plotly_chart(fig_daily, use_container_width=True, config={'displayModeBar': False})

# Anomaly Detection
st.markdown('<p class="section-title">Anomaly Detection</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="anomaly-box">No anomalies detected. Your spending is within normal ranges.</div>',
    unsafe_allow_html=True
)

# KenZen Insights
st.markdown('<p class="section-title">KenZen Insights</p>', unsafe_allow_html=True)

insights = [
    "Your highest spending category is Groceries at $2547.96. This is the primary lever for budget control.",
    "Spending trend over the last two weeks is increasing. Use this to decide whether to tighten or relax short-term budgets.",
    "Income vs spend ratio is 53.8%. Values below -40% usually indicate aggressive spending relative to income.",
    "Net savings from all transactions over the last 90 days is $1466.99. Automating transfers to a separate savings account would lock this in.",
    "If you cut dining by 30%, you could free up roughly $250.53 per month for savings or investing."
]

col1, col2, col3 = st.columns(3)

for idx, insight in enumerate(insights):
    if idx % 3 == 0:
        col = col1
    elif idx % 3 == 1:
        col = col2
    else:
        col = col3
    
    with col:
        st.markdown(f'<div class="insight-card"><div class="insight-card-text">{insight}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Recent Transactions
st.markdown('<p class="section-title">Recent Transactions</p>', unsafe_allow_html=True)

transactions_df = generate_mock_transactions()
display_df = transactions_df.head(10)[['Date', 'Merchant', 'Category', 'Amount', 'Type']]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        'Date': st.column_config.TextColumn('Date', width='small'),
        'Merchant': st.column_config.TextColumn('Merchant'),
        'Category': st.column_config.TextColumn('Category', width='small'),
        'Amount': st.column_config.TextColumn('Amount', width='small'),
        'Type': st.column_config.TextColumn('Type', width='small'),
    }
)
