import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import io
import re
from io import StringIO
from pypdf import PdfReader

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
    """Generate realistic mock transaction data for demo"""
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

def parse_pdf_file(uploaded_file):
    """Extract transactions from PDF bank statement"""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        # Parse transactions using regex patterns
        # Pattern: Date, Description/Merchant, Amount
        transaction_pattern = r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s+(.+?)\s+(\d+[,.]?\d*\.?\d{0,2})'
        transactions = []
        
        lines = text.split('\n')
        for line in lines:
            # Look for date patterns
            date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', line)
            if date_match:
                # Extract amount (look for currency amounts)
                amount_match = re.search(r'([\d,]+\.?\d{0,2})\s*(?:DR|CR|Debit|Credit)?', line)
                if amount_match:
                    date_str = date_match.group(1)
                    merchant = line.split(date_str)[-1].split(amount_match.group(1))[0].strip()
                    amount_str = amount_match.group(1).replace(',', '')
                    
                    if merchant and amount_str:
                        try:
                            date_obj = pd.to_datetime(date_str, format='%m/%d/%Y', errors='coerce')
                            if pd.isna(date_obj):
                                date_obj = pd.to_datetime(date_str, format='%d/%m/%Y', errors='coerce')
                            
                            if not pd.isna(date_obj):
                                transactions.append({
                                    'Date': date_obj,
                                    'Merchant': merchant[:50],
                                    'Amount': float(amount_str),
                                    'Type': 'Credit' if 'CR' in line or 'Credit' in line else 'Debit'
                                })
                        except:
                            pass
        
        if transactions:
            df = pd.DataFrame(transactions)
            df['Category'] = 'Other'
            return df[['Date', 'Merchant', 'Category', 'Amount', 'Type']]
        return None
    except Exception as e:
        return None

def parse_csv_file(uploaded_file):
    """Parse CSV bank statements with intelligent column detection"""
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read()
        
        # Try multiple encodings
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252', 'utf-16']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    on_bad_lines='skip',
                    dtype=str,
                    thousands=','
                )
                if len(df) > 0 and len(df.columns) > 0:
                    break
            except:
                continue
        
        if df is None or len(df) == 0:
            return None
        
        df = df.dropna(how='all')
        df = df[df.columns[df.notna().any()]]
        
        return df
    except Exception as e:
        return None

def standardize_transactions(df):
    """Smart column detection that works with ANY CSV format"""
    try:
        if df is None or len(df) == 0:
            return None
        
        df = df.copy()
        
        # Normalize column names
        df.columns = [str(col).strip().lower().replace(' ', '_').replace('.', '') for col in df.columns]
        
        # Remove header rows
        df = df.dropna(how='all')
        df = df[~df.iloc[:, 0].astype(str).str.contains('date|posted|transaction|description', case=False, na=False).fillna(False)]
        
        if len(df) == 0:
            return None
        
        # Find Date column
        date_col = None
        date_keywords = ['date', 'posted', 'trans_date', 'post_date', 'transaction_date', 'tdate']
        
        for col in df.columns:
            if any(keyword in col for keyword in date_keywords):
                date_col = col
                break
        
        # If not found, try first column
        if not date_col:
            try:
                test_dates = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                if test_dates.notna().sum() > len(df) * 0.5:
                    date_col = df.columns[0]
            except:
                pass
        
        if not date_col:
            return None
        
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce', infer_datetime_format=True)
        df = df.dropna(subset=['Date'])
        
        if len(df) == 0:
            return None
        
        # Find Merchant column
        merchant_col = None
        merchant_keywords = ['merchant', 'description', 'desc', 'details', 'narration', 'memo', 'payee', 'vendor', 'name']
        
        for col in df.columns:
            if any(keyword in col for keyword in merchant_keywords):
                merchant_col = col
                break
        
        df['Merchant'] = df[merchant_col].astype(str).fillna('Unknown').str.strip() if merchant_col else 'Transaction'
        
        # Find Category column
        category_col = None
        for col in df.columns:
            if 'category' in col:
                category_col = col
                break
        
        df['Category'] = df[category_col].astype(str).fillna('Other').str.strip() if category_col else 'Other'
        
        # Find Amount columns
        def clean_amount(val):
            if pd.isna(val) or val == '' or val == '0':
                return 0.0
            val_str = str(val).strip()
            val_str = val_str.replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
            try:
                return float(val_str)
            except:
                return 0.0
        
        debit_col = None
        credit_col = None
        amount_col = None
        
        for col in df.columns:
            if 'debit' in col and 'card' not in col:
                debit_col = col
            elif 'credit' in col and 'card' not in col:
                credit_col = col
            elif any(x in col for x in ['amount', 'amt', 'value', 'total', 'withdrawal', 'deposit']):
                amount_col = col
        
        # Build Amount column
        if debit_col or credit_col:
            df['Amount'] = 0.0
            if debit_col:
                df['Amount'] = df['Amount'] + df[debit_col].apply(clean_amount)
            if credit_col:
                df['Amount'] = df['Amount'] + df[credit_col].apply(clean_amount)
            df['Type'] = df.apply(
                lambda r: 'Credit' if (credit_col and clean_amount(r.get(credit_col, 0)) > 0) else 'Debit',
                axis=1
            )
        elif amount_col:
            df['Amount'] = df[amount_col].apply(clean_amount)
            df['Type'] = 'Debit'
        else:
            # Find any numeric column
            for col in df.columns:
                try:
                    test_vals = df[col].dropna().head(5).astype(str)
                    numeric_vals = test_vals.apply(lambda x: clean_amount(x))
                    if numeric_vals.sum() > 0 and numeric_vals.max() > 1:
                        df['Amount'] = df[col].apply(clean_amount)
                        df['Type'] = 'Debit'
                        amount_col = col
                        break
                except:
                    pass
            
            if amount_col is None:
                return None
        
        df['Amount'] = df['Amount'].apply(lambda x: abs(float(x)))
        df = df[df['Amount'] > 0]
        df = df.dropna(subset=['Date', 'Amount'])
        
        if len(df) == 0:
            return None
        
        result = df[['Date', 'Merchant', 'Category', 'Amount', 'Type']].sort_values('Date').reset_index(drop=True)
        return result
    
    except Exception as e:
        return None

# Header and upload
col_header, col_upload = st.columns([2, 1])

with col_header:
    st.markdown("# KenZen AI Finance Dashboard")
    st.markdown("Personal finance analytics with spending insights and anomaly detection")

with col_upload:
    uploaded_file = st.file_uploader("Upload Bank Statement", type=["csv", "pdf"], label_visibility="collapsed", help="Export your bank statement as CSV or PDF")

# Process upload
if uploaded_file is not None:
    if uploaded_file.name.endswith('.pdf'):
        standardized_df = parse_pdf_file(uploaded_file)
    else:
        raw_df = parse_csv_file(uploaded_file)
        standardized_df = standardize_transactions(raw_df) if raw_df is not None else None
    
    if standardized_df is not None and len(standardized_df) > 0:
        st.session_state.uploaded_df = standardized_df
        st.session_state.use_uploaded = True

# Use uploaded data or mock
if st.session_state.use_uploaded and st.session_state.uploaded_df is not None:
    df = st.session_state.uploaded_df
    data_source = "Your uploaded statement"
else:
    df = generate_mock_data()
    data_source = "Demo data"

st.caption(f"Data source: {data_source}")

# Calculate KPIs
base_balance = 5000
credits = df[df['Type'] == 'Credit']['Amount'].sum()
debits = df[df['Type'] == 'Debit']['Amount'].sum()
current_balance = base_balance + credits - debits

current_month = df[df['Date'].dt.month == datetime.now().month]
monthly_spending = current_month[current_month['Type'] == 'Debit']['Amount'].sum()

last_month_date = datetime.now() - timedelta(days=30)
prev_month = df[(df['Date'].dt.month == last_month_date.month) & (df['Type'] == 'Debit')]
prev_spending = prev_month['Amount'].sum()
spending_change = ((monthly_spending - prev_spending) / prev_spending * 100) if prev_spending > 0 else 0

total_savings = credits - debits

# Display KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Account Balance", f"${current_balance:,.2f}")
with col2:
    st.metric("Monthly Spending", f"${monthly_spending:,.2f}", f"{spending_change:+.1f}%")
with col3:
    st.metric("Total Savings", f"${total_savings:,.2f}")
with col4:
    st.metric("Transactions", len(df))

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
    st.dataframe(anomalies_df, use_container_width=True, hide_index=True)
else:
    st.info("No anomalies detected in your spending")

st.divider()

st.markdown("## AI Insights")

def generate_insights():
    debit_df = df[df['Type'] == 'Debit']
    
    insights = []
    
    # Biggest spending category
    top_category = debit_df.groupby('Category')['Amount'].sum().idxmax()
    top_amount = debit_df.groupby('Category')['Amount'].sum().max()
    insights.append(f"Top spending category is {top_category} at ${top_amount:.2f}")
    
    # Savings potential
    average_daily = debit_df['Amount'].mean()
    insights.append(f"Your average transaction is ${average_daily:.2f}")
    
    # Spending trend
    recent_week = df[df['Date'] > datetime.now() - timedelta(days=7)][df['Type'] == 'Debit']['Amount'].sum()
    insights.append(f"Last 7 days spending: ${recent_week:.2f}")
    
    # Frequency analysis
    transaction_count = len(debit_df)
    avg_per_day = transaction_count / 90
    insights.append(f"You average {avg_per_day:.1f} transactions per day")
    
    # Largest transaction
    largest = debit_df['Amount'].max()
    insights.append(f"Your largest transaction was ${largest:.2f}")
    
    return insights

insights = generate_insights()

for insight in insights:
    st.info(insight)

st.divider()

st.markdown("## Recent Transactions")

recent_tx = df.sort_values('Date', ascending=False).head(20)
display_tx = recent_tx.copy()
display_tx['Date'] = display_tx['Date'].dt.strftime('%Y-%m-%d')
display_tx['Amount'] = display_tx['Amount'].apply(lambda x: f"${x:.2f}")

st.dataframe(display_tx, use_container_width=True, hide_index=True)
