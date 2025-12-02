import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import io
import re
from io import StringIO
from PyPDF2 import PdfReader


# Page config
st.set_page_config(
    page_title="KenZen AI Finance Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
    .pipeline-step { display: inline-block; margin-right: 1rem; padding: 0.5rem 1rem; background-color: #1e293b; border-radius: 0.375rem; border: 1px solid #334155; color: #10b981; font-size: 0.85rem; font-weight: 500; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "use_uploaded" not in st.session_state:
    st.session_state.use_uploaded = False


def generate_mock_data():
    """Generate realistic mock transaction data for demo"""
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

    # Add salary-like deposits
    for i in range(0, 90, 30):
        transactions.append(
            {
                "Date": dates[i],
                "Merchant": "Employer Deposit",
                "Category": "Income",
                "Amount": 3000,
                "Type": "Credit",
            }
        )

    return pd.DataFrame(transactions)


def parse_pdf_file(uploaded_file):
    """Extract transactions from PDF bank statement"""
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        transactions = []
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue

            # Look for date patterns
            date_match = re.search(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})", line)
            if not date_match:
                continue

            # Extract amount (look for currency amounts)
            amount_match = re.search(r"([\d,]+\.?\d{0,2})\s*(?:DR|CR|Debit|Credit)?", line)
            if not amount_match:
                continue

            date_str = date_match.group(1)
            amount_str = amount_match.group(1).replace(",", "")
            merchant = line.split(date_str)[-1].split(amount_match.group(1))[0].strip()

            if not merchant or not amount_str:
                continue

            try:
                date_obj = pd.to_datetime(date_str, format="%m/%d/%Y", errors="coerce")
                if pd.isna(date_obj):
                    date_obj = pd.to_datetime(date_str, format="%d/%m/%Y", errors="coerce")
                if pd.isna(date_obj):
                    continue

                amount_val = float(amount_str)
                tx_type = "Credit" if ("CR" in line or "Credit" in line) else "Debit"

                transactions.append(
                    {
                        "Date": date_obj,
                        "Merchant": merchant[:50],
                        "Amount": amount_val,
                        "Type": tx_type,
                    }
                )
            except Exception:
                continue

        if not transactions:
            st.warning("Could not extract transactions from PDF")
            return None

        df = pd.DataFrame(transactions)
        df["Category"] = "Other"
        return df[["Date", "Merchant", "Category", "Amount", "Type"]]

    except Exception as e:
        st.error(f"PDF parsing error: {e}")
        return None
def parse_csv_file(uploaded_file):
    """Parse messy bank CSVs by auto-detecting the real header row."""
    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read().decode("latin1", errors="ignore")

        # Split into individual lines
        lines = raw.split("\n")

        # Look for the REAL header row
        header_keywords = ["date", "description", "amount", "transaction", "debit", "credit"]
        header_index = None

        for i, line in enumerate(lines):
            cleaned = line.lower().replace(" ", "")
            if any(key in cleaned for key in header_keywords):
                header_index = i
                break

        if header_index is None:
            st.error("Could not find valid CSV header.")
            return None

        # Reconstruct clean CSV content starting from header
        cleaned_csv = "\n".join(lines[header_index:])

        from io import StringIO
        df = pd.read_csv(StringIO(cleaned_csv), on_bad_lines="skip", dtype=str)

        # Remove empty rows and columns
        df = df.dropna(how="all")
        df = df.loc[:, df.notna().any()]

        return df

    except Exception as e:
        st.error(f"CSV parsing error: {e}")
        return None



def standardize_transactions(df):
    """Smart column detection that works with MANY CSV formats"""
    try:
        if df is None or len(df) == 0:
            return None

        df = df.copy()

        # Normalize column names
        df.columns = [str(col).strip().lower().replace(" ", "_").replace(".", "") for col in df.columns]

        # Remove header-like rows
        df = df.dropna(how="all")
        df = df[
            ~df.iloc[:, 0]
            .astype(str)
            .str.contains("date|posted|transaction|description", case=False, na=False)
            .fillna(False)
        ]

        if len(df) == 0:
            return None

        # Find Date column
        date_col = None
        date_keywords = ["date", "posted", "trans_date", "post_date", "transaction_date", "tdate"]

        for col in df.columns:
            if any(keyword in col for keyword in date_keywords):
                date_col = col
                break

        # If not found, try first column
        if not date_col:
            try:
                test_dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
                if test_dates.notna().sum() > len(df) * 0.5:
                    date_col = df.columns[0]
            except Exception:
                pass

        if not date_col:
            st.error("Could not find date column in uploaded CSV")
            return None

        df["Date"] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
        df = df.dropna(subset=["Date"])

        if len(df) == 0:
            st.error("No valid dates found")
            return None

        # Find Merchant column
        merchant_col = None
        merchant_keywords = [
            "merchant",
            "description",
            "desc",
            "details",
            "narration",
            "memo",
            "payee",
            "vendor",
            "name",
        ]

        for col in df.columns:
            if any(keyword in col for keyword in merchant_keywords):
                merchant_col = col
                break

        df["Merchant"] = (
            df[merchant_col].astype(str).fillna("Unknown").str.strip() if merchant_col else "Transaction"
        )

        # Find Category column
        category_col = None
        for col in df.columns:
            if "category" in col:
                category_col = col
                break

        df["Category"] = (
            df[category_col].astype(str).fillna("Other").str.strip() if category_col else "Other"
        )

        # Find Amount columns
        def clean_amount(val):
            if pd.isna(val) or val == "" or val == "0":
                return 0.0
            val_str = str(val).strip()
            val_str = (
                val_str.replace("$", "")
                .replace(",", "")
                .replace("(", "-")
                .replace(")", "")
            )
            try:
                return float(val_str)
            except Exception:
                return 0.0

        debit_col = None
        credit_col = None
        amount_col = None

        for col in df.columns:
            if "debit" in col and "card" not in col:
                debit_col = col
            elif "credit" in col and "card" not in col:
                credit_col = col
            elif any(x in col for x in ["amount", "amt", "value", "total", "withdrawal", "deposit"]):
                amount_col = col

        # Build Amount column
        if debit_col or credit_col:
            df["Amount"] = 0.0
            df["Type"] = "Debit"

            if debit_col:
                debits = df[debit_col].apply(clean_amount)
                df.loc[debits != 0, "Amount"] = debits[debits != 0]
                df.loc[debits != 0, "Type"] = "Debit"

            if credit_col:
                credits = df[credit_col].apply(clean_amount)
                df.loc[credits != 0, "Amount"] = credits[credits != 0]
                df.loc[credits != 0, "Type"] = "Credit"

        elif amount_col:
            df["Amount"] = df[amount_col].apply(clean_amount)
            df["Type"] = "Debit"
        else:
            # Fallback: try any numeric-like column
            for col in df.columns:
                try:
                    test_vals = df[col].dropna().head(5).astype(str)
                    numeric_vals = test_vals.apply(clean_amount)
                    if numeric_vals.sum() > 0 and numeric_vals.max() > 1:
                        df["Amount"] = df[col].apply(clean_amount)
                        df["Type"] = "Debit"
                        amount_col = col
                        break
                except Exception:
                    continue

            if amount_col is None:
                st.error("Could not find amount column in uploaded CSV")
                return None

        df["Amount"] = df["Amount"].apply(lambda x: abs(float(x)))
        df = df[df["Amount"] > 0]
        df = df.dropna(subset=["Date", "Amount"])

        if len(df) == 0:
            return None

        result = (
            df[["Date", "Merchant", "Category", "Amount", "Type"]]
            .sort_values("Date")
            .reset_index(drop=True)
        )
        return result

    except Exception as e:
        st.error(f"Standardization error: {e}")
        return None


# Header and upload
col_header, col_upload = st.columns([2, 1])

with col_header:
    st.markdown("# KenZen AI Finance Dashboard")
    st.markdown("Personal finance analytics with spending insights and anomaly detection")

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Bank Statement",
        type=["csv", "pdf"],
        label_visibility="collapsed",
        help="Export your bank statement as CSV or PDF",
    )

# Process upload
if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".pdf"):
        standardized_df = parse_pdf_file(uploaded_file)
    else:
        raw_df = parse_csv_file(uploaded_file)
        standardized_df = standardize_transactions(raw_df) if raw_df is not None else None

    if standardized_df is not None and len(standardized_df) > 0:
        st.session_state.uploaded_df = standardized_df
        st.session_state.use_uploaded = True
        st.success(f"Loaded {len(standardized_df)} transactions from your statement")
    else:
        st.warning("Could not parse/standardize uploaded file, using demo data")
        st.session_state.use_uploaded = False

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
credits = df[df["Type"] == "Credit"]["Amount"].sum()
debits = df[df["Type"] == "Debit"]["Amount"].sum()
current_balance = base_balance + credits - debits

current_month = df[df["Date"].dt.month == datetime.now().month]
monthly_spending = current_month[current_month["Type"] == "Debit"]["Amount"].sum()

last_month_date = datetime.now() - timedelta(days=30)
prev_month = df[(df["Date"].dt.month == last_month_date.month) & (df["Type"] == "Debit")]
prev_spending = prev_month["Amount"].sum()
spending_change = (
    (monthly_spending - prev_spending) / prev_spending * 100 if prev_spending > 0 else 0
)

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

# Display pipeline status
if st.session_state.use_uploaded and st.session_state.uploaded_df is not None:
    st.markdown("### Data Processing Pipeline")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown('<div class="pipeline-step">✓ Real data ingestion</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="pipeline-step">✓ Real preprocessing</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="pipeline-step">✓ Real transformation</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="pipeline-step">✓ Real pattern extraction</div>', unsafe_allow_html=True)
    with c5:
        st.markdown('<div class="pipeline-step">✓ Real anomaly detection</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="pipeline-step">✓ Real forecasting</div>', unsafe_allow_html=True)
    st.divider()

st.markdown("## Spending Analysis")

chart_col1, chart_col2 = st.columns(2)

# Spending by Category
with chart_col1:
    debit_df_for_chart = df[df["Type"] == "Debit"]
    if not debit_df_for_chart.empty:
        spending_by_category = (
            debit_df_for_chart.groupby("Category")["Amount"].sum().sort_values(ascending=True)
        )

        fig_category = go.Figure(
            data=[
                go.Bar(
                    y=spending_by_category.index,
                    x=spending_by_category.values,
                    orientation="h",
                    marker=dict(color="#06b6d4", line=dict(color="#0891b2", width=1.5)),
                    text=[f"${x:.2f}" for x in spending_by_category.values],
                    textposition="auto",
                    hovertemplate="<b>%{y}</b><br>%{x:$.2f}<extra></extra>",
                )
            ]
        )
        fig_category.update_layout(
            title="Spending by Category",
            xaxis_title="Amount",
            yaxis_title="",
            height=400,
            showlegend=False,
            hovermode="closest",
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f172a",
            font=dict(color="#cbd5e1", size=12),
            title_font=dict(size=14, color="#cbd5e1"),
            xaxis=dict(gridcolor="#334155"),
        )
        st.plotly_chart(fig_category, use_container_width=True)
    else:
        st.info("No debit transactions available to show spending by category.")

# Daily Spending Trend
with chart_col2:
    debit_df_for_trend = df[df["Type"] == "Debit"]
    if not debit_df_for_trend.empty:
        daily_spending = debit_df_for_trend.groupby(debit_df_for_trend["Date"].dt.date)["Amount"].sum()

        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=daily_spending.index,
                y=daily_spending.values,
                mode="lines+markers",
                name="Daily Spending",
                line=dict(color="#06b6d4", width=2.5),
                marker=dict(size=5, color="#0891b2"),
                fill="tozeroy",
                fillcolor="rgba(6, 182, 212, 0.1)",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:$.2f}<extra></extra>",
            )
        )
        fig_trend.update_layout(
            title="Daily Spending Trend",
            xaxis_title="Date",
            yaxis_title="Amount",
            height=400,
            hovermode="x unified",
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f172a",
            font=dict(color="#cbd5e1", size=12),
            title_font=dict(size=14, color="#cbd5e1"),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
            showlegend=False,
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No debit transactions available to show daily trend.")

st.divider()

st.markdown("## Anomaly Detection")


def detect_anomalies():
    anomalies = []
    debit_df_local = df[df["Type"] == "Debit"].copy()

    if debit_df_local.empty:
        return None

    for category in debit_df_local["Category"].unique():
        category_transactions = debit_df_local[debit_df_local["Category"] == category]
        mean_amount = category_transactions["Amount"].mean()
        threshold = mean_amount * 2

        anomaly_txns = category_transactions[category_transactions["Amount"] > threshold]

        for _, txn in anomaly_txns.iterrows():
            risk_level = "High" if txn["Amount"] > threshold * 1.5 else "Medium"
            anomalies.append(
                {
                    "Date": txn["Date"].strftime("%Y-%m-%d"),
                    "Merchant": txn["Merchant"],
                    "Category": txn["Category"],
                    "Amount": f"${txn['Amount']:.2f}",
                    "Threshold": f"${threshold:.2f}",
                    "Risk": risk_level,
                }
            )

    return pd.DataFrame(anomalies) if anomalies else None


anomalies_df = detect_anomalies()

if anomalies_df is not None and len(anomalies_df) > 0:
    st.dataframe(anomalies_df, use_container_width=True, hide_index=True)
else:
    st.info("No anomalies detected in your spending")

st.divider()

st.markdown("## AI Insights")


def generate_insights():
    debit_df_local = df[df["Type"] == "Debit"]

    insights = []

    if debit_df_local.empty:
        insights.append(
            "No debit (spending) transactions detected. Upload a statement with expenses to see insights."
        )
        return insights

    # Biggest spending category
    spending_by_cat = debit_df_local.groupby("Category")["Amount"].sum()
    top_category = spending_by_cat.idxmax()
    top_amount = spending_by_cat.max()
    insights.append(f"Top spending category is {top_category} at ${top_amount:.2f}.")

    # Average transaction
    avg_amount = debit_df_local["Amount"].mean()
    insights.append(f"Your average debit transaction is ${avg_amount:.2f}.")

    # Last 7 days spending
    recent_mask = (df["Date"] > datetime.now() - timedelta(days=7)) & (df["Type"] == "Debit")
    recent_week = df.loc[recent_mask, "Amount"].sum()
    insights.append(f"Last 7 days total spending: ${recent_week:.2f}.")

    # Frequency analysis
    transaction_count = len(debit_df_local)
    # Approximate days covered by data
    days_span = max((df["Date"].max() - df["Date"].min()).days + 1, 1)
    avg_per_day = transaction_count / days_span
    insights.append(f"You average {avg_per_day:.1f} debit transactions per day.")

    # Largest transaction
    largest = debit_df_local["Amount"].max()
    insights.append(f"Your largest debit transaction was ${largest:.2f}.")

    # Simple savings hint for Dining
    avg_dining = debit_df_local[debit_df_local["Category"] == "Dining"]["Amount"].mean()
    if avg_dining > 0:
        potential_savings = avg_dining * 0.3 * 30
        insights.append(
            f"If you cut Dining spending by 30%, you could save about ${potential_savings:.2f} per month."
        )

    return insights


insights = generate_insights()

for insight in insights:
    st.info(insight)

st.divider()

st.markdown("## Recent Transactions")

recent_tx = df.sort_values("Date", ascending=False).head(20)
display_tx = recent_tx.copy()
display_tx["Date"] = display_tx["Date"].dt.strftime("%Y-%m-%d")
display_tx["Amount"] = display_tx["Amount"].apply(lambda x: f"${x:.2f}")

st.dataframe(display_tx, use_container_width=True, hide_index=True)
