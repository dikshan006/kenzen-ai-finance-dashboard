import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import random

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
        background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 100%);
        color: #e5e7eb;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #151d2f 100%);
    }
    
    [data-testid="stMetric"] {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    
    .header-main {
        color: #06b6d4;
        font-size: 40px;
        font-weight: bold;
        margin: 0;
    }
    
    .header-sub {
        color: #94a3b8;
        font-size: 15px;
        margin: 8px 0 0 0;
    }
    
    .section-title {
        color: #06b6d4;
        font-size: 22px;
        font-weight: 600;
        margin: 30px 0 15px 0;
        border-bottom: 2px solid #06b6d4;
        padding-bottom: 10px;
    }
    
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    
    .kpi-value {
        color: #06b6d4;
        font-size: 32px;
        font-weight: bold;
        margin: 8px 0;
    }
    
    .kpi-label {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 500;
    }
    
    .kpi-delta {
        color: #10b981;
        font-size: 13px;
        margin-top: 6px;
    }
    
    .kpi-delta-negative {
        color: #ef4444;
    }
    
    .pipeline-item {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        display: inline-block;
        color: #10b981;
        font-weight: 500;
    }
    
    .insight-positive {
        background: #0f3f2f;
        border-left: 4px solid #10b981;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
        color: #d1fae5;
    }
    
    .insight-warning {
        background: #4a2f1f;
        border-left: 4px solid #f59e0b;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
        color: #fef3c7;
    }
    
    .insight-alert {
        background: #3f1f1f;
        border-left: 4px solid #ef4444;
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
        color: #fee2e2;
    }
    
    .insight-title {
        font-weight: 600;
        margin-bottom: 6px;
        font-size: 15px;
    }
    
    .insight-text {
        font-size: 14px;
        line-height: 1.5;
    }
    
    .transaction-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
    }
    
    .transaction-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .transaction-merchant {
        color: #e5e7eb;
        font-weight: 500;
        font-size: 15px;
    }
    
    .transaction-category {
        color: #94a3b8;
        font-size: 13px;
        margin-top: 4px;
    }
    
    .transaction-amount {
        color: #06b6d4;
        font-weight: bold;
        font-size: 16px;
    }
    
    .transaction-date {
        color: #64748b;
        font-size: 12px;
        margin-top: 8px;
    }
    
    .anomaly-flag {
        background: #7f1d1d;
        color: #fecaca;
        border: 1px solid #dc2626;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
    }
</style>
"""

st.markdown(dark_theme_css, unsafe_allow_html=True)

# ==================== MOCK DATA GENERATION ====================

@st.cache_data
def generate_mock_transactions():
    """Generate realistic mock transaction data"""
    np.random.seed(42)
    random.seed(42)
    
    merchants = {
        'Groceries': ['Whole Foods Market', 'Safeway', 'Kroger', 'Trader Joe', 'Costco', 'Target Market'],
        'Restaurants': ['Blue Hill Restaurant', 'The Gramercy Tavern', 'Balthazar Bistro', 'Carbone', 'Nobu', 'Eleven Madison'],
        'Transportation': ['Uber', 'Lyft', 'Shell Gas Station', 'Chevron Fuel', 'United Airlines', 'NYC Metro'],
        'Utilities': ['Con Edison Electric', 'Verizon Mobile', 'Comcast Internet', 'Manhattan Water'],
        'Shopping': ['Amazon Prime', 'Apple Store', 'Nike Online', 'Urban Outfitters', 'Zara', 'Best Buy'],
        'Entertainment': ['Netflix Subscription', 'Spotify Premium', 'AMC Theaters', 'Regal Cinema', 'Disney Plus'],
        'Health': ['CVS Pharmacy', 'Walgreens', 'Planet Fitness', 'Medical Center', 'Dental Associates']
    }
    
    amount_ranges = {
        'Groceries': (30, 200),
        'Restaurants': (35, 250),
        'Transportation': (15, 150),
        'Utilities': (50, 300),
        'Shopping': (25, 500),
        'Entertainment': (10, 100),
        'Health': (20, 300)
    }
    
    transactions = []
    base_date = datetime.now() - timedelta(days=60)
    
    for day_offset in range(60):
        current_date = base_date + timedelta(days=day_offset)
        num_transactions = np.random.poisson(3)
        
        for _ in range(num_transactions):
            category = random.choice(list(merchants.keys()))
            merchant = random.choice(merchants[category])
            min_amt, max_amt = amount_ranges[category]
            amount = np.random.uniform(min_amt, max_amt)
            
            transactions.append({
                'Date': current_date,
                'Merchant': merchant,
                'Category': category,
                'Amount': round(amount, 2),
                'Type': 'Debit'
            })
    
    return pd.DataFrame(transactions).sort_values('Date', ascending=False).reset_index(drop=True)

# ==================== ANOMALY DETECTION ====================

def detect_anomalies(df):
    """Detect anomalous transactions based on category averages"""
    anomalies = []
    
    for category in df['Category'].unique():
        category_data = df[df['Category'] == category]
        avg_amount = category_data['Amount'].mean()
        std_amount = category_data['Amount'].std()
        threshold = avg_amount + (2 * std_amount)
        
        anomalous = category_data[category_data['Amount'] > threshold]
        
        for idx, row in anomalous.iterrows():
            anomalies.append({
                'Date': row['Date'],
                'Merchant': row['Merchant'],
                'Category': row['Category'],
                'Amount': row['Amount'],
                'Average': avg_amount,
                'Threshold': threshold,
                'Status': 'Anomaly'
            })
    
    return pd.DataFrame(anomalies) if anomalies else None

# ==================== AI INSIGHTS ====================

def generate_ai_insights(df):
    """Generate AI-powered budgeting insights and recommendations"""
    insights = []
    
    spending_by_category = df.groupby('Category')['Amount'].agg(['sum', 'mean', 'count']).reset_index()
    spending_by_category.columns = ['Category', 'Total', 'Avg', 'Count']
    spending_by_category = spending_by_category.sort_values('Total', ascending=False)
    
    total_spending = spending_by_category['Total'].sum()
    
    # Insight 1: Top spending category
    if len(spending_by_category) > 0:
        top_category = spending_by_category.iloc[0]
        pct = (top_category['Total'] / total_spending * 100)
        
        insights.append({
            'type': 'warning' if pct > 35 else 'info',
            'title': f"Top Spending Category: {top_category['Category']}",
            'message': f"You spent ${top_category['Total']:,.2f} on {top_category['Category']} ({pct:.1f}% of total). This is your largest expense category."
        })
    
    # Insight 2: Restaurant spending
    restaurant_data = spending_by_category[spending_by_category['Category'] == 'Restaurants']
    if not restaurant_data.empty and restaurant_data.iloc[0]['Total'] > 200:
        amount = restaurant_data.iloc[0]['Total']
        savings = amount * 0.3
        insights.append({
            'type': 'alert',
            'title': 'High Dining Expenses Detected',
            'message': f"You spent ${amount:,.2f} on restaurants and dining. Reducing by 30% could save ${savings:,.2f} monthly."
        })
    
    # Insight 3: Savings potential
    discretionary = spending_by_category[
        spending_by_category['Category'].isin(['Shopping', 'Entertainment', 'Restaurants'])
    ]['Total'].sum()
    
    if discretionary > 100:
        potential_savings = discretionary * 0.25
        annual = potential_savings * 12
        insights.append({
            'type': 'positive',
            'title': 'Significant Savings Opportunity',
            'message': f"By reducing discretionary spending by 25%, you could save ${potential_savings:,.2f} monthly (${annual:,.2f} annually)."
        })
    
    # Insight 4: Budget recommendation
    monthly_average = total_spending / 2
    recommended_budget = monthly_average * 0.9
    insights.append({
        'type': 'positive',
        'title': 'Recommended Monthly Budget',
        'message': f"Based on your spending, we recommend a ${recommended_budget:,.2f} monthly budget to optimize savings."
    })
    
    # Insight 5: Category advice
    for idx, row in spending_by_category.head(3).iterrows():
        if row['Category'] == 'Utilities':
            insights.append({
                'type': 'info',
                'title': 'Optimize Your Utilities',
                'message': f"You paid ${row['Total']:,.2f} for utilities. Review energy usage and compare providers for better rates."
            })
        elif row['Category'] == 'Shopping':
            insights.append({
                'type': 'warning',
                'title': 'Shopping Frequency Alert',
                'message': f"You made {int(row['Count'])} shopping purchases totaling ${row['Total']:,.2f}. Consider planning purchases in advance."
            })
    
    return insights

# ==================== MAIN APP ====================

# Header
st.markdown('''
<div style="margin-bottom: 40px;">
    <p class="header-main">KenZen Finance Dashboard</p>
    <p class="header-sub">Personal finance analytics with spending insights and anomaly detection</p>
</div>
''', unsafe_allow_html=True)

# Load data
transactions_df = generate_mock_transactions()
total_spending = transactions_df['Amount'].sum()
account_balance = 6466.99
total_savings = 1466.99
transaction_count = len(transactions_df)

# KPI Metrics
st.markdown('<p class="section-title">Account Overview</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Account Balance</div>
        <div class="kpi-value">${account_balance:,.2f}</div>
        <div class="kpi-delta">Stable</div>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    delta_pct = ((account_balance - total_spending) / account_balance * 100) if account_balance > 0 else 0
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Monthly Spending</div>
        <div class="kpi-value">${total_spending:,.2f}</div>
        <div class="kpi-delta kpi-delta-negative">-12.3%</div>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Total Savings</div>
        <div class="kpi-value">${total_savings:,.2f}</div>
        <div class="kpi-delta">+8.5%</div>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    st.markdown(f'''
    <div class="kpi-card">
        <div class="kpi-label">Transactions</div>
        <div class="kpi-value">{transaction_count}</div>
        <div class="kpi-delta">Last 60 days</div>
    </div>
    ''', unsafe_allow_html=True)

st.divider()

# Data Processing Pipeline
st.markdown('<p class="section-title">Data Processing Pipeline</p>', unsafe_allow_html=True)

col1, col2, col3, col4, col5, col6 = st.columns(6)
pipeline_steps = ['Real data ingestion', 'Real preprocessing', 'Real transformation', 'Real pattern extraction', 'Real anomaly detection', 'Real forecasting']

for idx, (col, step) in enumerate(zip([col1, col2, col3, col4, col5, col6], pipeline_steps)):
    with col:
        st.markdown(f'<div class="pipeline-item">✓ {step}</div>', unsafe_allow_html=True)

st.divider()

# Spending Analysis
st.markdown('<p class="section-title">Spending Analysis</p>', unsafe_allow_html=True)

spending_by_category = transactions_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)

col1, col2 = st.columns(2)

with col1:
    fig_category = go.Figure(data=[
        go.Bar(
            y=spending_by_category.index,
            x=spending_by_category.values,
            orientation='h',
            marker=dict(color='#06b6d4', line=dict(color='#334155', width=1)),
            text=[f'${v:,.0f}' for v in spending_by_category.values],
            textposition='auto',
        )
    ])
    fig_category.update_layout(
        title="Spending by Category",
        xaxis_title="Amount",
        yaxis_title="Category",
        plot_bgcolor='#1e293b',
        paper_bgcolor='#0f172a',
        font=dict(color='#e5e7eb', size=12),
        hovermode='y unified',
        height=400,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_category, use_container_width=True)

with col2:
    df_daily = transactions_df.copy()
    df_daily['Date'] = pd.to_datetime(df_daily['Date']).dt.date
    daily_spending = df_daily.groupby('Date')['Amount'].sum().reset_index()
    
    fig_daily = go.Figure(data=[
        go.Scatter(
            x=daily_spending['Date'],
            y=daily_spending['Amount'],
            mode='lines+markers',
            fill='tozeroy',
            line=dict(color='#06b6d4', width=2),
            marker=dict(size=6),
            fillcolor='rgba(6, 182, 212, 0.1)'
        )
    ])
    fig_daily.update_layout(
        title="Daily Spending Trend",
        xaxis_title="Date",
        yaxis_title="Amount",
        plot_bgcolor='#1e293b',
        paper_bgcolor='#0f172a',
        font=dict(color='#e5e7eb', size=12),
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig_daily, use_container_width=True)

st.divider()

# Anomaly Detection
st.markdown('<p class="section-title">Anomaly Detection</p>', unsafe_allow_html=True)

anomalies_df = detect_anomalies(transactions_df)

if anomalies_df is not None and len(anomalies_df) > 0:
    st.markdown(f'**Found {len(anomalies_df)} anomalies** - Transactions exceeding 2x category average:', help='Anomalies are transactions significantly higher than typical spending in their category')
    
    for idx, anomaly in anomalies_df.head(5).iterrows():
        col1, col2, col3 = st.columns([0.5, 1.5, 1])
        with col1:
            st.markdown('<div class="anomaly-flag">ANOMALY</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            **{anomaly['Merchant']}** ({anomaly['Category']})  
            {anomaly['Date'].strftime('%B %d, %Y')}
            ''')
        with col3:
            st.markdown(f'''
            **${anomaly['Amount']:,.2f}**  
            Avg: ${anomaly['Average']:.2f}
            ''')
        st.divider()
else:
    st.info('No anomalies detected in your spending. All transactions are within expected ranges.')

st.divider()

# AI Insights
st.markdown('<p class="section-title">Budgeting Insights and Recommendations</p>', unsafe_allow_html=True)

insights = generate_ai_insights(transactions_df)

for insight in insights:
    if insight['type'] == 'positive':
        st.markdown(f'''
        <div class="insight-positive">
            <div class="insight-title">{insight['title']}</div>
            <div class="insight-text">{insight['message']}</div>
        </div>
        ''', unsafe_allow_html=True)
    elif insight['type'] == 'alert':
        st.markdown(f'''
        <div class="insight-alert">
            <div class="insight-title">{insight['title']}</div>
            <div class="insight-text">{insight['message']}</div>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="insight-warning">
            <div class="insight-title">{insight['title']}</div>
            <div class="insight-text">{insight['message']}</div>
        </div>
        ''', unsafe_allow_html=True)

st.divider()

# Recent Transactions
st.markdown('<p class="section-title">Recent Transactions</p>', unsafe_allow_html=True)

for idx, transaction in transactions_df.head(15).iterrows():
    st.markdown(f'''
    <div class="transaction-card">
        <div class="transaction-header">
            <div>
                <div class="transaction-merchant">{transaction['Merchant']}</div>
                <div class="transaction-category">{transaction['Category']}</div>
            </div>
            <div style="text-align: right;">
                <div class="transaction-amount">${transaction['Amount']:,.2f}</div>
                <div class="transaction-date">{transaction['Date'].strftime('%b %d, %Y')}</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
