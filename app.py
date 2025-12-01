import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import io
from pandas.errors import ParserError

# Page config
st.set_page_config(page_title="KenZen AI Finance Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 600;
        color: #06b6d4;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    .main {
        padding: 2.5rem 3rem;
        background-color: #0f172a;
    }
    h1 {
        color: #f1f5f9;
        margin-bottom: 0.5rem;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    h2 {
        color: #cbd5e1;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-size: 1.5rem;
        font-weight: 600;
        border-bottom: 1px solid #334155;
        padding-bottom: 0.75rem;
    }
    .stInfo {
        background-color: #1e293b;
        border-left: 3px solid #06b6d4;
        border-radius: 0.5rem;
        padding: 1rem;
        color: #e2e8f0;
    }
    .stSuccess {
        background-color: #1e293b;
        border-left: 3px solid #10b981;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .stWarning {
        background-color: #1e293b;
        border-left: 3px solid #f59e0b;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    [data-testid="stDataFrame"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 0.5rem;
    }
    .stCaption {
        color: #64748b;
        font-size: 0.8rem;
    }
    .stDivider {
        border-top: 1px solid #334155;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Generate mock transaction data
@st.cache_data
def generate_mock_data():
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=x) for x in range(90)]
    dates.reverse()
    
    categories = ['Groceries', 'Dining', 'Transportation', 'Entertainment', 'Utilities', 'Shopping']
    merchants = {
        'Groceries': ['Whole Foods', 'Trader Joe\'s', 'Safeway'],
        'Dining': ['Chipotle', 'Starbucks', 'Thai Palace', 'Sushi Bar'],
        'Transportation': ['Uber', 'Shell Gas', 'Toyota Service'],
        'Entertainment': ['Netflix', 'Cinema', 'Concert'],
        'Utilities': ['PG&E', 'Comcast', 'Water Bill'],
        'Shopping': ['Amazon', 'Target', 'Gap']
    }
    
    transactions = []
    for date in dates:
        num_transactions = np.random.randint(1, 4)
        for _ in range(num_transactions):
            category = np.random.choice(categories)
            merchant = np.random.choice(merchants[category])
            
            # Base amounts by category
            base_amounts = {
                'Groceries': 80,
                'Dining': 25,
                'Transportation': 40,
                'Entertainment': 15,
                'Utilities': 100,
                'Shopping': 60
            }
            
            amount = base_amounts[category] + np.random.normal(0, base_amounts[category] * 0.3)
            amount = max(5, abs(amount))
            
            # 80% debit, 20% credit (income)
            transaction_type = 'Credit' if np.random.random() < 0.1 else 'Debit'
            
            transactions.append({
                'Date': date,
                'Merchant': merchant,
                'Category': category,
                'Amount': round(amount, 2),
                'Type': transaction_type
            })
    
    # Add some income transactions
    for i in range(0, 90, 30):
        if i < len(dates):
            transactions.append({
                'Date': dates[i],
                'Merchant': 'Employer Deposit',
                'Category': 'Income',
                'Amount': 3000,
                'Type': 'Credit'
            })
    
    return pd.DataFrame(transactions)

def standardize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes any bank CSV into standard schema:
    [Date, Merchant, Category, Amount, Type]
    """
    df = df.copy()
    
    # Step 1: Standardize column names (strip whitespace, lowercase for detection)
    df.columns = df.columns.str.strip()
    col_lower = {col: col.lower() for col in df.columns}
    
    # Step 2: Find and rename Date column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break
    
    if date_col:
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    else:
        raise KeyError("Could not find a Date column in your CSV. Please include a 'Date' column.")
    
    # Step 3: Find and rename Merchant/Description column
    merchant_col = None
    for col in df.columns:
        col_lower_name = col.lower()
        if any(x in col_lower_name for x in ['description', 'details', 'merchant', 'narration']):
            merchant_col = col
            break
    
    if merchant_col:
        df['Merchant'] = df[merchant_col].fillna('Unknown')
    else:
        df['Merchant'] = 'Unknown'
    
    # Step 4: Find Category column or create default
    category_col = None
    for col in df.columns:
        if 'category' in col.lower():
            category_col = col
            break
    
    if category_col:
        df['Category'] = df[category_col].fillna('Uncategorized')
    else:
        df['Category'] = 'Uncategorized'
    
    # Step 5: Find Amount/Debit/Credit columns
    amount_col = None
    debit_col = None
    credit_col = None
    
    # Look for separate debit/credit columns
    for col in df.columns:
        col_lower_name = col.lower()
        if 'debit' in col_lower_name and 'credit card' not in col_lower_name:
            debit_col = col
        if 'credit' in col_lower_name and 'card' not in col_lower_name:
            credit_col = col
    
    # If we found debit/credit, use them
    if debit_col is not None or credit_col is not None:
        df['Amount'] = pd.to_numeric(df.get(debit_col, 0), errors='coerce').fillna(0) + \
                       pd.to_numeric(df.get(credit_col, 0), errors='coerce').fillna(0)
        
        # Determine Type
        if debit_col and credit_col:
            df['Type'] = df.apply(
                lambda row: 'Debit' if row[debit_col] > 0 else ('Credit' if row[credit_col] > 0 else 'Debit'),
                axis=1
            )
        else:
            df['Type'] = 'Debit' if debit_col else 'Credit'
    else:
        # Look for single Amount column
        for col in df.columns:
            col_lower_name = col.lower()
            if col_lower_name == 'amount' or col_lower_name.endswith(' amt'):
                amount_col = col
                break
        
        if amount_col is None:
            raise KeyError("Could not find an Amount / Debit / Credit column in your CSV.")
        
        df['Amount'] = pd.to_numeric(df[amount_col], errors='coerce').abs()
        
        # Infer Type from amount sign if present
        type_col = None
        for col in df.columns:
            if col.lower() == 'type' or col.lower() == 'transaction type':
                type_col = col
                break
        
        if type_col:
            df['Type'] = df[type_col].apply(lambda x: str(x).strip())
        else:
            # Infer from original amount sign
            df['Type'] = df[amount_col].apply(lambda x: 'Debit' if float(x) < 0 else 'Credit')
    
    # Step 6: Clean up
    df = df.dropna(subset=['Date', 'Amount'])
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').abs()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    return df[['Date', 'Merchant', 'Category', 'Amount', 'Type']].sort_values('Date')


def load_transactions(uploaded_file):
    """
    Load transactions from uploaded file or use mock data.
    Returns: (dataframe, source_label)
    """
    if uploaded_file is None:
        # No file uploaded, use mock data
        return generate_mock_data(), "Mock demo data (KenZen simulator)"
    
    # Check if file is PDF
    if uploaded_file.type == 'application/pdf' or uploaded_file.name.lower().endswith('.pdf'):
        return generate_mock_data(), "PDF not supported - using mock data"
    
    try:
        # Try standard CSV read
        df = pd.read_csv(uploaded_file)
    except (UnicodeDecodeError, ParserError):
        try:
            # Fallback: read as latin1 with error handling
            uploaded_file.seek(0)
            raw_text = uploaded_file.read().decode('latin1', errors='ignore')
            df = pd.read_csv(io.StringIO(raw_text), sep=None, engine='python', on_bad_lines='skip')
        except Exception as e:
            raise Exception(f"Failed to parse CSV: {str(e)}")
    
    # Standardize the transactions
    df = standardize_transactions(df)
    
    return df, "Your uploaded CSV"

col_header, col_upload = st.columns([2, 1])

with col_header:
    st.markdown("# KenZen AI Finance Dashboard")
    st.markdown("Personal finance analytics with spending insights and anomaly detection")

with col_upload:
    uploaded_file = st.file_uploader("Upload Bank Statement", type=["csv", "pdf"], label_visibility="collapsed")
    
    if uploaded_file is None:
        st.caption("Data source: Mock demo data")
    elif uploaded_file.type == 'application/pdf' or uploaded_file.name.lower().endswith('.pdf'):
        st.info("PDF statements not supported. Please export as CSV from your bank.")
        st.caption("Data source: Mock demo data")
    else:
        st.caption("Data source: Your uploaded CSV")

try:
    df, data_source = load_transactions(uploaded_file)
except Exception as e:
    st.error(f"Error parsing statement: {str(e)}\n\nUsing demo data instead.")
    df, data_source = load_transactions(None)

# Calculate KPIs
def calculate_kpis():
    base_balance = 5000
    credits = df[df['Type'] == 'Credit']['Amount'].sum()
    debits = df[df['Type'] == 'Debit']['Amount'].sum()
    current_balance = base_balance + credits - debits
    
    # Monthly spending (current month)
    current_month = df[df['Date'].dt.month == datetime.now().month]
    monthly_spending = current_month[current_month['Type'] == 'Debit']['Amount'].sum()
    
    # Previous month spending
    last_month_date = datetime.now() - timedelta(days=30)
    prev_month = df[(df['Date'].dt.month == last_month_date.month) & (df['Type'] == 'Debit')]
    prev_spending = prev_month['Amount'].sum()
    
    spending_change = ((monthly_spending - prev_spending) / prev_spending * 100) if prev_spending > 0 else 0
    
    total_savings = credits - debits
    transaction_count = len(df)
    
    return {
        'balance': current_balance,
        'monthly_spending': monthly_spending,
        'spending_change': spending_change,
        'total_savings': total_savings,
        'transaction_count': transaction_count,
        'total_credits': credits,
        'total_debits': debits
    }

kpis = calculate_kpis()

st.divider()

st.markdown("## Spending Analysis")

chart_col1, chart_col2 = st.columns(2)

# Spending by Category
with chart_col1:
    spending_by_category = df[df['Type'] == 'Debit'].groupby('Category')['Amount'].sum().sort_values(ascending=True)
    
    fig_category = go.Figure(data=[
        go.Bar(
            y=spending_by_category.index,
            x=spending_by_category.values,
            orientation='h',
            marker=dict(color='#06b6d4', line=dict(color='#0891b2', width=1.5)),
            text=[f'${x:.2f}' for x in spending_by_category.values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>%{x:$.2f}<extra></extra>',
        )
    ])
    fig_category.update_layout(
        title="Spending by Category",
        xaxis_title="Amount",
        yaxis_title="",
        height=400,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='#1e293b',
        paper_bgcolor='#0f172a',
        font=dict(color='#cbd5e1', size=12),
        title_font=dict(size=14, color='#cbd5e1'),
        xaxis=dict(gridcolor='#334155'),
    )
    st.plotly_chart(fig_category, use_container_width=True)

# Daily Spending Trend
with chart_col2:
    daily_spending = df[df['Type'] == 'Debit'].groupby(df['Date'].dt.date)['Amount'].sum()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily_spending.index,
        y=daily_spending.values,
        mode='lines+markers',
        name='Daily Spending',
        line=dict(color='#06b6d4', width=2.5),
        marker=dict(size=5, color='#0891b2'),
        fill='tozeroy',
        fillcolor='rgba(6, 182, 212, 0.1)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>%{y:$.2f}<extra></extra>',
    ))
    fig_trend.update_layout(
        title="Daily Spending Trend",
        xaxis_title="Date",
        yaxis_title="Amount",
        height=400,
        hovermode='x unified',
        plot_bgcolor='#1e293b',
        paper_bgcolor='#0f172a',
        font=dict(color='#cbd5e1', size=12),
        title_font=dict(size=14, color='#cbd5e1'),
        xaxis=dict(gridcolor='#334155'),
        yaxis=dict(gridcolor='#334155'),
        showlegend=False
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

st.markdown("## Anomaly Detection")

def detect_anomalies():
    anomalies = []
    debit_df = df[df['Type'] == 'Debit'].copy()
    
    for category in debit_df['Category'].unique():
        category_transactions = debit_df[debit_df['Category'] == category]
        mean_amount = category_transactions['Amount'].mean()
        threshold = mean_amount * 2
        
        anomaly_txns = category_transactions[category_transactions['Amount'] > threshold]
        
        for _, txn in anomaly_txns.iterrows():
            risk_level = 'High' if txn['Amount'] > threshold * 1.5 else 'Medium'
            anomalies.append({
                'Date': txn['Date'].strftime('%Y-%m-%d'),
                'Merchant': txn['Merchant'],
                'Category': txn['Category'],
                'Amount': f"${txn['Amount']:.2f}",
                'Threshold': f"${threshold:.2f}",
                'Risk': risk_level
            })
    
    return pd.DataFrame(anomalies) if anomalies else None

anomalies_df = detect_anomalies()

if anomalies_df is not None and len(anomalies_df) > 0:
    st.warning(f"Found {len(anomalies_df)} anomalous transactions exceeding 2x category average")
    st.dataframe(anomalies_df, use_container_width=True, hide_index=True)
else:
    st.success("No anomalies detected. Spending within normal ranges.")

st.divider()

st.markdown("## AI-Powered Insights")

def generate_insights():
    spending_by_cat = df[df['Type'] == 'Debit'].groupby('Category')['Amount'].sum()
    biggest_category = spending_by_cat.idxmax()
    biggest_amount = spending_by_cat.max()
    
    credit_ratio = kpis['total_credits'] / (kpis['total_credits'] + kpis['total_debits']) * 100 if (kpis['total_credits'] + kpis['total_debits']) > 0 else 0
    
    avg_dining = df[(df['Category'] == 'Dining') & (df['Type'] == 'Debit')]['Amount'].mean()
    potential_savings = avg_dining * 0.3 * 30 if avg_dining > 0 else 0
    
    recent_spending = df[(df['Date'] > datetime.now() - timedelta(days=14)) & (df['Type'] == 'Debit')]['Amount'].sum()
    earlier_spending = df[(df['Date'] > datetime.now() - timedelta(days=28)) & (df['Date'] <= datetime.now() - timedelta(days=14)) & (df['Type'] == 'Debit')]['Amount'].sum()
    trend = "increasing" if recent_spending > earlier_spending else "decreasing"
    
    insights = [
        f"Top spending category: {biggest_category} at ${biggest_amount:.2f}. Consider setting spending limits.",
        f"Credit ratio: {credit_ratio:.1f}%. {('Strong income flow' if credit_ratio > 40 else 'Higher spending relative to income')}.",
        f"Potential savings: ${potential_savings:.2f}/month by reducing dining expenses by 30%.",
        f"Spending trend: {trend.capitalize()} over the past two weeks.",
        f"Total savings: ${kpis['total_savings']:.2f}. Consider automating transfers to savings."
    ]
    
    return insights

insights = generate_insights()

col1, col2 = st.columns([1, 1])

with col1:
    for insight in insights[:3]:
        st.info(insight)

with col2:
    for insight in insights[3:]:
        st.info(insight)

st.divider()

st.markdown("## Recent Transactions")
recent_tx = df.sort_values('Date', ascending=False).head(20).copy()
recent_tx['Date'] = recent_tx['Date'].dt.strftime('%Y-%m-%d')
recent_tx['Amount'] = recent_tx['Amount'].apply(lambda x: f"${x:.2f}")

st.dataframe(
    recent_tx[['Date', 'Merchant', 'Category', 'Amount', 'Type']],
    use_container_width=True,
    hide_index=True
)

st.markdown("---")
st.markdown("KenZen AI Finance Dashboard • Powered by Streamlit")
