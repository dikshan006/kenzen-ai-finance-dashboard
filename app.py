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
    try:
        df = df.copy()
        
        # Step 1: Standardize column names (strip whitespace, lowercase for detection)
        df.columns = df.columns.str.strip()
        original_cols = df.columns.tolist()
        
        # Step 2: Find and rename Date column
        date_col = None
        date_patterns = ['date', 'transaction date', 'posted date', 'posting date', 'tdate', 'post', 'trans date']
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in date_patterns or any(pattern in col_lower for pattern in date_patterns):
                date_col = col
                break
        
        if not date_col and len(df.columns) > 0:
            # If no date found, try first column as fallback
            date_col = df.columns[0]
        
        if date_col:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
            df = df.dropna(subset=['Date'])
        else:
            st.error(f"Could not find a Date column. Columns: {', '.join(original_cols[:5])}")
            return None
        
        # Step 3: Find and rename Merchant/Description column
        merchant_col = None
        merchant_patterns = ['description', 'details', 'merchant', 'narration', 'transaction desc', 'memo', 'name', 'desc']
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in merchant_patterns or any(pattern in col_lower for pattern in merchant_patterns):
                merchant_col = col
                break
        
        if merchant_col:
            df['Merchant'] = df[merchant_col].astype(str).fillna('Unknown')
        else:
            df['Merchant'] = 'Unknown'
        
        # Step 4: Find Category column or create default
        category_col = None
        for col in df.columns:
            if 'category' in col.lower().strip():
                category_col = col
                break
        
        if category_col:
            df['Category'] = df[category_col].astype(str).fillna('Uncategorized')
        else:
            df['Category'] = 'Uncategorized'
        
        # Step 5: Find Amount/Debit/Credit columns with robust detection
        amount_col = None
        debit_col = None
        credit_col = None
        
        amount_patterns = ['amount', 'amt', 'value', 'transaction amount', 'total', 'balance']
        debit_patterns = ['debit', 'withdrawal', 'paid', 'out', 'expense']
        credit_patterns = ['credit', 'deposit', 'received', 'in', 'income']
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(pattern in col_lower for pattern in debit_patterns) and 'card' not in col_lower:
                debit_col = col
            if any(pattern in col_lower for pattern in credit_patterns) and 'card' not in col_lower:
                credit_col = col
            if any(pattern in col_lower for pattern in amount_patterns) and debit_col is None and credit_col is None:
                amount_col = col
        
        # If we found debit/credit, use them
        if debit_col is not None or credit_col is not None:
            debit_vals = pd.to_numeric(df.get(debit_col, 0), errors='coerce').fillna(0).abs()
            credit_vals = pd.to_numeric(df.get(credit_col, 0), errors='coerce').fillna(0).abs()
            df['Amount'] = debit_vals + credit_vals
            
            df['Type'] = df.apply(
                lambda row: 'Debit' if (debit_col and row.get(debit_col, 0) != 0) else 'Credit',
                axis=1
            )
        elif amount_col:
            df['Amount'] = pd.to_numeric(df[amount_col], errors='coerce').abs()
            df['Type'] = 'Debit'
        else:
            st.error(f"Could not find Amount columns. Available: {', '.join(original_cols[:5])}")
            return None
        
        # Step 6: Clean up
        df = df.dropna(subset=['Date', 'Amount'])
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').abs()
        df = df[df['Amount'] > 0]
        
        return df[['Date', 'Merchant', 'Category', 'Amount', 'Type']].sort_values('Date')
    
    except Exception as e:
        st.error(f"Standardization error: {str(e)[:80]}")
        return None

def load_transactions(uploaded_file):
    """
    Load transactions from uploaded file or use mock data.
    Returns: (dataframe, source_label)
    """
    if uploaded_file is None:
        return generate_mock_data(), "Mock demo data (KenZen simulator)"
    
    # Check if file is PDF
    if uploaded_file.type == 'application/pdf' or uploaded_file.name.lower().endswith('.pdf'):
        st.warning("PDF statements not supported. Please export as CSV from your bank.")
        return generate_mock_data(), "Mock demo data (PDF not supported)"
    
    try:
        # Try multiple encoding strategies
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(
                    uploaded_file, 
                    encoding=encoding,
                    on_bad_lines='skip',
                    engine='python',  # use python engine for more flexible parsing
                    dtype=str  # read all as strings first to avoid type issues
                )
                break
            except Exception:
                continue
        
        if df is None:
            st.error("Could not parse CSV with any encoding. Please verify file format.")
            return generate_mock_data(), "Mock demo data (encoding error)"
        
        # Standardize the transactions
        standardized_df = standardize_transactions(df)
        
        if standardized_df is None:
            return generate_mock_data(), "Mock demo data (column mapping failed)"
        
        return standardized_df, "Your uploaded CSV"
    
    except Exception as e:
        st.error(f"Error parsing CSV: {str(e)[:100]}")
        return generate_mock_data(), "Mock demo data (parse error)"

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
