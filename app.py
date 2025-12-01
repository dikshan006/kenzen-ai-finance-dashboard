import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import io
import re
from io import StringIO

# Page config
st.set_page_config(page_title="KenZen AI Finance Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    body { background-color: #0f172a; color: #e2e8f0; }
    [data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 600; color: #06b6d4; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem; color: #94a3b8; font-weight: 400; letter-spacing: 0.5px; }
    .main { padding: 2.5rem 3rem; background-color: #0f172a; }
    h1 { color: #f1f5f9; margin-bottom: 0.5rem; font-size: 2.8rem; font-weight: 700; letter-spacing: -0.5px; }
    h2 { color: #cbd5e1; margin-top: 2.5rem; margin-bottom: 1.5rem; font-size: 1.5rem; font-weight: 600; border-bottom: 1px solid #334155; padding-bottom: 0.75rem; }
    .stInfo { background-color: #1e293b; border-left: 3px solid #06b6d4; border-radius: 0.5rem; padding: 1rem; color: #e2e8f0; }
    .stSuccess { background-color: #1e293b; border-left: 3px solid #10b981; border-radius: 0.5rem; padding: 1rem; }
    .stWarning { background-color: #1e293b; border-left: 3px solid #f59e0b; border-radius: 0.5rem; padding: 1rem; }
    [data-testid="stDataFrame"] { background-color: #1e293b; border: 1px solid #334155; border-radius: 0.5rem; }
    .stCaption { color: #64748b; font-size: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None
if 'use_uploaded' not in st.session_state:
    st.session_state.use_uploaded = False

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
            base_amounts = {
                'Groceries': 80, 'Dining': 25, 'Transportation': 40,
                'Entertainment': 15, 'Utilities': 100, 'Shopping': 60
            }
            amount = base_amounts[category] + np.random.normal(0, base_amounts[category] * 0.3)
            amount = max(5, abs(amount))
            transaction_type = 'Credit' if np.random.random() < 0.1 else 'Debit'
            
            transactions.append({
                'Date': date, 'Merchant': merchant, 'Category': category,
                'Amount': round(amount, 2), 'Type': transaction_type
            })
    
    for i in range(0, 90, 30):
        if i < len(dates):
            transactions.append({
                'Date': dates[i], 'Merchant': 'Employer Deposit', 'Category': 'Income',
                'Amount': 3000, 'Type': 'Credit'
            })
    
    return pd.DataFrame(transactions)

def parse_csv_file(uploaded_file):
    """New aggressive CSV parser that works with any column names"""
    try:
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'ascii']
        df = None
        
        for encoding in encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, on_bad_lines='skip', dtype=str)
                if len(df) > 0:
                    break
            except:
                continue
        
        if df is None or len(df) == 0:
            return None
        
        st.write("[DEBUG] CSV loaded. Columns:", df.columns.tolist())
        st.write("[DEBUG] First row:", df.iloc[0].to_dict() if len(df) > 0 else "No data")
        
        return df
    except Exception as e:
        st.error(f"Failed to parse: {str(e)}")
        return None

def standardize_transactions(df):
    """More aggressive column detection - tries every column combination"""
    try:
        df = df.copy()
        
        # Normalize column names
        df.columns = [col.strip().lower() for col in df.columns]
        
        # Debug: show what columns we have
        st.write("[DEBUG] Normalized columns:", df.columns.tolist())
        
        # Find date column - check every column
        date_col = None
        for col in df.columns:
            if any(x in col for x in ['date', 'posted', 'trans', 'tdate', 'post', 'transaction']):
                date_col = col
                break
        
        if not date_col:
            date_col = df.columns[0]
        
        st.write(f"[DEBUG] Using '{date_col}' as date column")
        
        # Convert to datetime
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
        df = df.dropna(subset=['Date'])
        
        if len(df) == 0:
            st.error("No valid dates found")
            return None
        
        # Find merchant/description
        merchant_col = None
        for col in df.columns:
            if any(x in col for x in ['merchant', 'description', 'desc', 'details', 'narration', 'memo', 'notes', 'payee']):
                merchant_col = col
                break
        
        if merchant_col:
            df['Merchant'] = df[merchant_col].astype(str).fillna('Unknown').str.strip()
        else:
            df['Merchant'] = 'Transaction'
        
        # Find category
        category_col = None
        for col in df.columns:
            if 'category' in col:
                category_col = col
                break
        
        if category_col:
            df['Category'] = df[category_col].astype(str).fillna('Other').str.strip()
        else:
            df['Category'] = 'Other'
        
        # Find amount columns - be aggressive
        amount_col = None
        debit_col = None
        credit_col = None
        
        for col in df.columns:
            if any(x in col for x in ['debit', 'withdrawal', 'out']) and 'card' not in col:
                debit_col = col
            if any(x in col for x in ['credit', 'deposit', 'in']) and 'card' not in col:
                credit_col = col
            if any(x in col for x in ['amount', 'amt', 'value', 'total', 'transaction']):
                amount_col = col
        
        st.write(f"[DEBUG] debit_col={debit_col}, credit_col={credit_col}, amount_col={amount_col}")
        
        def clean_amount(val):
            if pd.isna(val) or val == '':
                return 0
            val_str = str(val).replace('$', '').replace(',', '').strip()
            val_str = val_str.replace('(', '-').replace(')', '')
            try:
                return float(val_str)
            except:
                return 0
        
        # Build Amount column
        if debit_col or credit_col:
            df['Amount'] = 0.0
            if debit_col:
                df['Amount'] += df[debit_col].apply(clean_amount)
            if credit_col:
                df['Amount'] += df[credit_col].apply(clean_amount)
            df['Type'] = df.apply(lambda r: 'Credit' if (credit_col and clean_amount(r.get(credit_col, 0)) > 0) else 'Debit', axis=1)
        elif amount_col:
            df['Amount'] = df[amount_col].apply(clean_amount)
            df['Type'] = 'Debit'
        else:
            # Last resort: find any numeric column
            numeric_cols = []
            for col in df.columns:
                try:
                    df[col].apply(lambda x: float(str(x).replace('$', '').replace(',', '')))
                    numeric_cols.append(col)
                except:
                    pass
            
            if numeric_cols:
                df['Amount'] = df[numeric_cols[0]].apply(clean_amount)
                df['Type'] = 'Debit'
            else:
                st.error("Could not find any amount columns")
                return None
        
        df['Amount'] = df['Amount'].apply(lambda x: abs(float(x)))
        df = df[df['Amount'] > 0]
        df = df.dropna(subset=['Date', 'Amount'])
        
        result = df[['Date', 'Merchant', 'Category', 'Amount', 'Type']].sort_values('Date').reset_index(drop=True)
        st.write(f"[DEBUG] Successfully parsed {len(result)} transactions")
        return result
    
    except Exception as e:
        st.error(f"Standardization error: {str(e)}")
        return None

# Header and upload
col_header, col_upload = st.columns([2, 1])

with col_header:
    st.markdown("# KenZen AI Finance Dashboard")
    st.markdown("Personal finance analytics with spending insights and anomaly detection")

with col_upload:
    uploaded_file = st.file_uploader("Upload Bank Statement", type=["csv", "pdf"], label_visibility="collapsed")

if uploaded_file is not None:
    with st.expander("Processing upload...", expanded=True):
        raw_df = parse_csv_file(uploaded_file)
        
        if raw_df is not None:
            standardized_df = standardize_transactions(raw_df)
            if standardized_df is not None and len(standardized_df) > 0:
                st.session_state.uploaded_df = standardized_df
                st.session_state.use_uploaded = True
                st.success(f"Loaded {len(standardized_df)} transactions from your statement")
            else:
                st.warning("Could not standardize data, using demo")
                st.session_state.use_uploaded = False
        else:
            st.warning("Could not parse file, using demo")
            st.session_state.use_uploaded = False

# Use uploaded data or mock
if st.session_state.use_uploaded and st.session_state.uploaded_df is not None:
    df = st.session_state.uploaded_df
    data_source = "Your uploaded statement"
else:
    df = generate_mock_data()
    data_source = "Mock demo data"

st.caption(f"Data source: {data_source}")

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
