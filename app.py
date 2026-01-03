import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import io

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


# ---------- CSV Parsing Functions ----------
@st.cache_data
def parse_csv(uploaded_file, spending_is_negative=True):
    """Parse messy CSV with flexible column detection"""
    diagnostics = {
        "total_rows": 0,
        "valid_rows": 0,
        "dropped_rows": 0,
        "drop_reasons": [],
        "column_mapping": {},
        "date_range": None,
        "uncategorized_count": 0
    }
    
    try:
        # Try reading CSV with multiple encoding attempts
        content = uploaded_file.getvalue()
        raw_df = None
        
        # Try different combinations to handle messy CSVs
        for encoding in ['utf-8', 'latin-1', 'cp1252']:  # Multiple encodings
            for sep in [',', None, ';', '\t']:  # Multiple separators, None = auto-detect
                for skip in range(4):  # Try skiprows 0-3 to handle preamble
                    try:
                        if sep is None:
                            # Auto-detect separator
                            raw_df = pd.read_csv(
                                io.BytesIO(content), 
                                encoding=encoding, 
                                skip_blank_lines=True,
                                skiprows=skip,
                                dtype=str,  # Use dtype=str
                                on_bad_lines='skip',  # Skip bad lines
                                engine='python',
                                sep=None
                            )
                        else:
                            raw_df = pd.read_csv(
                                io.BytesIO(content), 
                                encoding=encoding, 
                                skip_blank_lines=True,
                                skiprows=skip,
                                sep=sep,
                                dtype=str,  # Use dtype=str
                                on_bad_lines='skip'  # Skip bad lines
                            )
                        
                        # Check if we got at least 2 columns
                        if raw_df is not None and len(raw_df.columns) >= 2:
                            break
                    except:
                        continue
                if raw_df is not None and len(raw_df.columns) >= 2:
                    break
            if raw_df is not None and len(raw_df.columns) >= 2:
                break
        
        if raw_df is None or len(raw_df.columns) < 2:
            return None, "Could not parse CSV with any encoding/separator combination"
        
        diagnostics["total_rows"] = len(raw_df)
        
        # Strip whitespace from column names
        raw_df.columns = raw_df.columns.str.strip()
        
        # Column detection (case-insensitive partial matching)
        def find_column(df, options):
            cols_lower = {c.lower(): c for c in df.columns}
            for opt in options:
                for col_lower, col_orig in cols_lower.items():
                    if opt.lower() in col_lower:
                        return col_orig
            return None
        
        # Find date column
        date_col = find_column(raw_df, ["date", "transaction date", "posted date", "time", "timestamp"])
        if not date_col:
            return None, "No date column found. Expected columns like 'Date', 'Transaction Date', etc."
        diagnostics["column_mapping"]["date"] = date_col
        
        # Find description column
        desc_col = find_column(raw_df, ["description", "merchant", "name", "details", "memo"])
        if not desc_col:
            desc_col = "description"  # Create a default
            raw_df[desc_col] = "Transaction"
        diagnostics["column_mapping"]["description"] = desc_col
        
        # Find amount column(s)
        amount_col = find_column(raw_df, ["amount", "amt", "value"])
        debit_col = find_column(raw_df, ["debit", "withdrawal"])
        credit_col = find_column(raw_df, ["credit", "deposit"])
        
        if amount_col:
            diagnostics["column_mapping"]["amount"] = amount_col
        elif debit_col and credit_col:
            diagnostics["column_mapping"]["amount"] = f"{debit_col} + {credit_col}"
        else:
            return None, "No amount column found. Expected 'Amount' or separate 'Debit'/'Credit' columns."
        
        # Find category column (optional)
        cat_col = find_column(raw_df, ["category", "type"])
        diagnostics["column_mapping"]["category"] = cat_col or "Not found (will use 'Uncategorized')"
        
        # Parse data
        parsed_rows = []
        
        for idx, row in raw_df.iterrows():
            try:
                # Parse date with errors="coerce" and dayfirst retry
                date_val = row[date_col]
                if pd.isna(date_val) or str(date_val).strip() == "":
                    diagnostics["drop_reasons"].append(f"Row {idx}: Missing date")
                    continue
                
                # Try parsing date with errors="coerce"
                parsed_date = pd.to_datetime(date_val, errors='coerce')
                
                # If parse failed (NaT), try with dayfirst=True
                if pd.isna(parsed_date):
                    parsed_date = pd.to_datetime(date_val, errors='coerce', dayfirst=True)
                
                if pd.isna(parsed_date):
                    diagnostics["drop_reasons"].append(f"Row {idx}: Invalid date format '{date_val}'")
                    continue
                
                # Handle single amount column vs split debit/credit with correct sign logic
                if amount_col:
                    # Single amount column
                    amount_val = row[amount_col]
                    
                    if pd.isna(amount_val) or str(amount_val).strip() == "":
                        diagnostics["drop_reasons"].append(f"Row {idx}: Missing amount")
                        continue
                    
                    # Clean amount: remove $, commas, handle parentheses as negative
                    amount_str = str(amount_val).strip()
                    amount_str = amount_str.replace('$', '').replace(',', '').replace(' ', '')
                    
                    # Handle parentheses as negative
                    if amount_str.startswith('(') and amount_str.endswith(')'):
                        amount_str = '-' + amount_str[1:-1]
                    
                    try:
                        amount_float = float(amount_str)
                    except:
                        diagnostics["drop_reasons"].append(f"Row {idx}: Invalid amount '{amount_val}'")
                        continue
                    
                    # Correct sign logic for spending_is_negative toggle
                    if spending_is_negative:
                        # Negative = Debit (spending), Positive = Credit (income)
                        if amount_float < 0:
                            txn_type = "Debit"
                            amount_magnitude = abs(amount_float)
                        else:
                            txn_type = "Credit"
                            amount_magnitude = amount_float
                    else:
                        # Positive = Debit (spending), Negative = Credit (income)
                        if amount_float > 0:
                            txn_type = "Debit"
                            amount_magnitude = amount_float
                        else:
                            txn_type = "Credit"
                            amount_magnitude = abs(amount_float)
                    
                else:
                    # Split debit/credit columns - parse both, determine which is non-zero
                    debit_val = row[debit_col]
                    credit_val = row[credit_col]
                    
                    # Clean and parse debit
                    debit_num = None
                    if not pd.isna(debit_val) and str(debit_val).strip() != "":
                        debit_str = str(debit_val).strip().replace('$', '').replace(',', '').replace(' ', '')
                        if debit_str.startswith('(') and debit_str.endswith(')'):
                            debit_str = '-' + debit_str[1:-1]
                        try:
                            debit_num = float(debit_str)
                        except:
                            pass
                    
                    # Clean and parse credit
                    credit_num = None
                    if not pd.isna(credit_val) and str(credit_val).strip() != "":
                        credit_str = str(credit_val).strip().replace('$', '').replace(',', '').replace(' ', '')
                        if credit_str.startswith('(') and credit_str.endswith(')'):
                            credit_str = '-' + credit_str[1:-1]
                        try:
                            credit_num = float(credit_str)
                        except:
                            pass
                    
                    # Determine type and magnitude
                    if credit_num is not None and credit_num != 0:
                        txn_type = "Credit"
                        amount_magnitude = abs(credit_num)
                    elif debit_num is not None and debit_num != 0:
                        txn_type = "Debit"
                        amount_magnitude = abs(debit_num)
                    else:
                        diagnostics["drop_reasons"].append(f"Row {idx}: Missing debit/credit amount")
                        continue
                
                # Get description
                description = str(row[desc_col]) if not pd.isna(row[desc_col]) else "Transaction"
                
                # Get category
                if cat_col and not pd.isna(row[cat_col]):
                    category = str(row[cat_col]).strip()
                else:
                    category = "Uncategorized"
                    diagnostics["uncategorized_count"] += 1
                
                parsed_rows.append({
                    "Date": parsed_date,
                    "Merchant": description,
                    "Category": category,
                    "Amount": round(amount_magnitude, 2),
                    "Type": txn_type
                })
                
            except Exception as e:
                diagnostics["drop_reasons"].append(f"Row {idx}: Error - {str(e)}")
                continue
        
        if not parsed_rows:
            return None, "No valid transactions found after parsing"
        
        # Create DataFrame
        result_df = pd.DataFrame(parsed_rows)
        result_df = result_df.sort_values("Date").reset_index(drop=True)
        
        diagnostics["valid_rows"] = len(result_df)
        diagnostics["dropped_rows"] = diagnostics["total_rows"] - diagnostics["valid_rows"]
        diagnostics["date_range"] = f"{result_df['Date'].min().date()} to {result_df['Date'].max().date()}"
        
        return result_df, diagnostics
        
    except Exception as e:
        return None, f"Error parsing CSV: {str(e)}"


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


# ---------- Header with CSV Upload ----------
header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown("# KenZen AI Finance Dashboard")
    st.markdown(
        '<div class="subheadline">A dark-mode analytics cockpit for personal finance – live spending, anomalies, and insights.</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown("### Upload Your Data")
    uploaded_file = st.file_uploader("CSV File", type=["csv"], label_visibility="collapsed")
    spending_is_negative = st.checkbox("Spending is negative", value=True, help="Check if expenses are negative numbers in your CSV")

if uploaded_file is not None:
    parsed_result, diagnostics = parse_csv(uploaded_file, spending_is_negative)
    
    if parsed_result is None:
        st.error(f"❌ CSV parsing failed: {diagnostics}")
        st.info("📊 Showing demo data instead. Please check your CSV format.")
        df = generate_mock_data()
        data_mode = "Demo Mode (CSV parsing failed)"
    else:
        df = parsed_result
        data_mode = f"Live Data ({len(df)} transactions)"
        
        # Show data quality expander
        with st.expander("📊 Data Quality Report"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows Loaded", diagnostics["valid_rows"])
            with col2:
                st.metric("Rows Dropped", diagnostics["dropped_rows"])
            with col3:
                st.metric("Uncategorized", diagnostics["uncategorized_count"])
            with col4:
                st.metric("Date Range", "✓" if diagnostics["date_range"] else "✗")
            
            st.markdown("**Column Mapping:**")
            for key, val in diagnostics["column_mapping"].items():
                st.text(f"  • {key}: {val}")
            
            if diagnostics["date_range"]:
                st.markdown(f"**Date Range:** {diagnostics['date_range']}")
            
            if diagnostics["drop_reasons"]:
                st.markdown(f"**Drop Reasons** (showing first 10):")
                for reason in diagnostics["drop_reasons"][:10]:
                    st.text(f"  • {reason}")
else:
    df = generate_mock_data()
    data_mode = "Demo Mode"

st.caption(f"🔹 {data_mode}")

# Calculate KPIs with current data
kpis = compute_kpis(df)

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
        debit_df.groupby(debit_df["Date"].dt.date)["Amount"].sum().reset_index()
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
    f"You spent the most on <strong>{top_cat}</strong> (${top_cat_amt:,.2f}). This is the biggest place you can save money.",
    f"Your income is higher than your spending (<strong>{kpis['income_ratio']:.1f}%</strong>). This is a healthy sign.",
    f"If you cut Dining by 30%, you could save about <strong>${dining_savings:,.2f}</strong> every month.",
    f"Your spending in the last 2 weeks is <strong>{trend_label}</strong>. This means your recent trend is moving in that direction.",
    f"You saved <strong>${kpis['net_savings_90']:,.2f}</strong> in the last 90 days. Moving some of this to a savings account would lock it in."
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

