"""
Robust CSV Parser for Multi-Bank Transaction Formats
Supports common formats and flexible column aliases.

Canonical output columns:
- date (datetime)
- description (str)
- amount_signed (float)  negative=spend, positive=income
- category (str)         default "Uncategorized" if missing
- balance (float)        NaN if missing
- transaction_id (str)   optional
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import hashlib
import re


COLUMN_ALIASES = {
    "date": [
        "Date", "Transaction Date", "Posting Date", "Post Date", "Trans Date", "Posted Date"
    ],
    "description": [
        "Description", "Merchant", "Name", "Details", "Memo"
    ],
    "amount": [
        "Amount", "Transaction Amount", "Amt", "Value"
    ],
    "debit": [
        "Debit", "Withdrawal", "Withdrawals", "Charge", "Spent"
    ],
    "credit": [
        "Credit", "Deposit", "Deposits", "Payment", "Received"
    ],
    "balance": [
        "Balance", "Running Balance", "Available Balance"
    ],
    "category": [
        "Category", "Type", "Classification"
    ],
    "reference": [
        "Reference Number", "Transaction ID", "ID", "Ref", "Confirmation Number"
    ],
}


def normalize_column_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def find_column(columns: List[str], aliases: List[str]) -> Optional[str]:
    normalized_cols = {normalize_column_name(c): c for c in columns}
    for a in aliases:
        key = normalize_column_name(a)
        if key in normalized_cols:
            return normalized_cols[key]
    return None


def parse_amount(val: any) -> float:
    """
    Parse amount from string or number.
    Returns float; missing/invalid -> np.nan

    Handles: $4.50, (4.50), -4.50, +3500, 3,500.00, etc.
    """
    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()
    if not val_str:
        return np.nan

    # Remove $ and whitespace
    val_str = val_str.replace("$", "").strip()

    # Handle parentheses as negative (even if spaces exist)
    is_negative = False
    if val_str.startswith("(") and val_str.endswith(")"):
        is_negative = True
        val_str = val_str[1:-1].strip()

    # Remove commas
    val_str = val_str.replace(",", "")

    # Remove leading +
    val_str = val_str.lstrip("+")

    try:
        amount = float(val_str)
        if is_negative:
            amount = -abs(amount)
        return amount
    except ValueError:
        return np.nan


def parse_date(val: any) -> Optional[datetime]:
    """
    Parse date from multiple formats.
    Returns datetime or None.
    """
    if pd.isna(val):
        return None

    val_str = str(val).strip()
    if not val_str:
        return None

    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m/%d/%y",
        "%d/%m/%y",
        "%b %d %Y",
        "%d-%b-%Y",
        "%Y/%m/%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(val_str, fmt)
        except ValueError:
            continue

    # Fallback: let pandas try
    try:
        d = pd.to_datetime(val_str, errors="coerce")
        if pd.isna(d):
            return None
        return d.to_pydatetime()
    except Exception:
        return None


def normalize_description(desc: str) -> str:
    if pd.isna(desc):
        return ""
    return str(desc).strip().upper()


def create_dedup_hash(date: datetime, description: str, amount: float, reference: str = "") -> str:
    base = f"{date.date()}|{normalize_description(description)}|{amount:.2f}|{(reference or '').strip()}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def check_error_rate(df: pd.DataFrame, field_name: str, invalid_count: int, threshold: float = 0.05) -> Tuple[bool, str]:
    total = len(df)
    if total == 0:
        return False, "❌ CSV is empty after cleaning. Please upload a valid transaction CSV."
    rate = invalid_count / total
    if rate > threshold:
        return False, (
            f"❌ Too many invalid {field_name} values ({invalid_count}/{total}, {rate:.1%}). "
            f"Please check your CSV formatting."
        )
    return True, ""


class CSVParser:
    """
    Main CSV parser for bank transaction files.
    Required: date + description + (amount OR debit+credit)
    Optional: category, balance, transaction_id
    """

    def __init__(self):
        self.parsed_df = None
        self.original_row_count = 0
        self.duplicates_removed = 0
        self.empty_rows_skipped = 0
        self.invalid_rows = 0
        self.error_summary = ""

    def parse(self, file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
        try:
            df = pd.read_csv(file_path)
            self.original_row_count = len(df)

            if len(df) == 0:
                return None, "❌ CSV is empty. Please upload a file with at least one transaction."

            # Step 1: Find required columns
            date_col = find_column(df.columns, COLUMN_ALIASES["date"])
            desc_col = find_column(df.columns, COLUMN_ALIASES["description"])

            # Optional columns (we fill defaults if missing)
            balance_col = find_column(df.columns, COLUMN_ALIASES["balance"])
            category_col = find_column(df.columns, COLUMN_ALIASES["category"])
            reference_col = find_column(df.columns, COLUMN_ALIASES["reference"])

            # Amount can be single column OR separate debit/credit
            amount_col = find_column(df.columns, COLUMN_ALIASES["amount"])
            debit_col = find_column(df.columns, COLUMN_ALIASES["debit"])
            credit_col = find_column(df.columns, COLUMN_ALIASES["credit"])

            # Required: date + description + amount (or debit/credit pair)
            if not date_col:
                return None, (
                    f"❌ Missing required date column. Expected one of: {', '.join(COLUMN_ALIASES['date'])}\n"
                    f"Your CSV has: {', '.join(df.columns.tolist())}"
                )
            if not desc_col:
                return None, (
                    f"❌ Missing required description column. Expected one of: {', '.join(COLUMN_ALIASES['description'])}\n"
                    f"Your CSV has: {', '.join(df.columns.tolist())}"
                )
            if not amount_col and not (debit_col and credit_col):
                return None, (
                    f"❌ Missing required amount column. Provide one of: {', '.join(COLUMN_ALIASES['amount'])} "
                    f"OR both Debit and Credit columns.\nYour CSV has: {', '.join(df.columns.tolist())}"
                )

            # Fill defaults for optional fields
            if not category_col:
                df["__category_fallback__"] = "Uncategorized"
                category_col = "__category_fallback__"
            if not balance_col:
                df["__balance_fallback__"] = np.nan
                balance_col = "__balance_fallback__"
            if not reference_col:
                df["__reference_fallback__"] = ""
                reference_col = "__reference_fallback__"

            # Step 2: Drop fully empty rows
            df = df.dropna(how="all")
            self.empty_rows_skipped = self.original_row_count - len(df)

            # Step 3: Parse dates
            df["date_parsed"] = df[date_col].apply(parse_date)
            invalid_dates = df["date_parsed"].isna().sum()
            ok, err = check_error_rate(df, "date", invalid_dates)
            if not ok:
                return None, err

            # Step 4: Parse amounts
            if amount_col:
                df["amount_parsed"] = df[amount_col].apply(parse_amount)
            else:
                debit_vals = df[debit_col].apply(parse_amount)
                credit_vals = df[credit_col].apply(parse_amount)
                df["amount_parsed"] = credit_vals.fillna(0) - debit_vals.fillna(0)

            invalid_amounts = df["amount_parsed"].isna().sum()
            ok, err = check_error_rate(df, "amount", invalid_amounts)
            if not ok:
                return None, err

            # Step 5: Parse balance (optional)
            df["balance_parsed"] = df[balance_col].apply(parse_amount)

            # Step 6: Reference (optional)
            df["reference_parsed"] = df[reference_col].astype(str).str.strip()

            # Step 7: Filter invalid rows
            df = df[df["date_parsed"].notna() & df["amount_parsed"].notna()].copy()
            self.invalid_rows = int(invalid_dates + invalid_amounts)

            # Step 8: Deduplicate
            df["dedup_hash"] = df.apply(
                lambda row: create_dedup_hash(
                    row["date_parsed"],
                    row[desc_col],
                    row["amount_parsed"],
                    row["reference_parsed"],
                ),
                axis=1,
            )
            before = len(df)
            df = df.drop_duplicates(subset=["dedup_hash"], keep="first")
            self.duplicates_removed = before - len(df)

            # Step 9: Canonical output
            result_df = pd.DataFrame(
                {
                    "date": df["date_parsed"],
                    "description": df[desc_col].astype(str).str.strip(),
                    "amount_signed": df["amount_parsed"].astype(float),
                    "category": df[category_col].astype(str).str.strip().replace({"nan": "Uncategorized"}),
                    "balance": df["balance_parsed"].astype(float),
                    "transaction_id": df["reference_parsed"].astype(str),
                }
            )

            result_df = result_df.sort_values("date").reset_index(drop=True)

            self.error_summary = (
                "✅ **CSV Parsed Successfully**\n"
                f"- **Rows uploaded**: {self.original_row_count}\n"
                f"- **Rows kept**: {len(result_df)}\n"
                f"- **Empty rows skipped**: {self.empty_rows_skipped}\n"
                f"- **Duplicates removed**: {self.duplicates_removed}\n"
                f"- **Invalid rows**: {self.invalid_rows}\n"
            )

            self.parsed_df = result_df
            return result_df, ""

        except pd.errors.ParserError as e:
            return None, f"❌ CSV parsing error: {str(e)}\nPlease ensure your file is a valid CSV."
        except Exception as e:
            return None, f"❌ Unexpected error: {str(e)}"


def generate_csv_template() -> str:
    """Generate a CSV template with correct headers."""
    return """Date,Description,Amount,Category,Balance
2025-01-15,STARBUCKS COFFEE,-4.50,Dining,1245.32
2025-01-14,SALARY DEPOSIT,3500.00,Income,1249.82
2025-01-13,WHOLE FOODS MARKET,-87.42,Groceries,2500.00
"""
