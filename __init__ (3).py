"""Shared normalization for any tabular (CSV/XLSX) raw frame.

Takes a raw all-string dataframe, auto-detects columns via fuzzy matching,
and emits the standardized schema. Vectorized for large files.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from kenzen.cleaners.cleaning import (
    clean_amount_series,
    collapse_whitespace_series,
    drop_duplicate_transactions,
    parse_date_series,
    split_signed_amount,
)
from kenzen.utils import schema
from kenzen.utils.logging_config import get_logger

log = get_logger("parsers.tabular")


def normalize_table(
    raw: pd.DataFrame,
    source_file: str,
    spending_is_negative: bool = True,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Normalize a raw tabular frame into the standard schema.

    Returns ``(df, info)``. ``df`` is ``None`` on hard failure (no usable
    date or amount columns). ``info`` always carries keys the existing
    dashboard's Data-Quality expander reads.
    """
    info: Dict = {
        "total_rows": int(len(raw)),
        "valid_rows": 0,
        "dropped_rows": 0,
        "drop_reasons": [],
        "column_mapping": {},
        "date_range": None,
        "uncategorized_count": 0,
        "duplicates_removed": 0,
    }

    if raw is None or raw.empty or len(raw.columns) < 2:
        return None, "Could not detect a usable table (need at least 2 columns)."

    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    cols: List[str] = list(raw.columns)

    date_col = schema.best_column_for(cols, "date")
    if not date_col:
        return None, "No date column found (looked for Date / Transaction Date / Posted Date...)."
    info["column_mapping"]["date"] = date_col

    desc_col = schema.best_column_for(cols, "description", exclude=[date_col])
    info["column_mapping"]["description"] = desc_col or "(none - defaulted)"

    amount_col = schema.best_column_for(cols, "amount", exclude=[date_col])
    debit_col = schema.best_column_for(cols, "debit", exclude=[date_col, amount_col or ""])
    credit_col = schema.best_column_for(cols, "credit", exclude=[date_col, amount_col or "", debit_col or ""])
    cat_col = schema.best_column_for(cols, "category", exclude=[date_col, desc_col or ""])

    out = pd.DataFrame(index=raw.index)
    out["date"] = parse_date_series(raw[date_col])

    if desc_col:
        out["description"] = collapse_whitespace_series(raw[desc_col])
    else:
        out["description"] = "Transaction"

    if amount_col:
        info["column_mapping"]["amount"] = amount_col
        signed = clean_amount_series(raw[amount_col])
        magnitude, ttype = split_signed_amount(signed, spending_is_negative)
        out["amount"] = magnitude
        out["transaction_type"] = ttype.values
    elif debit_col or credit_col:
        info["column_mapping"]["amount"] = f"{debit_col or '-'} / {credit_col or '-'} (split)"
        debit = clean_amount_series(raw[debit_col]).abs() if debit_col else pd.Series(np.nan, index=raw.index)
        credit = clean_amount_series(raw[credit_col]).abs() if credit_col else pd.Series(np.nan, index=raw.index)
        is_credit = credit.fillna(0) != 0
        out["amount"] = np.where(is_credit, credit, debit)
        out["transaction_type"] = np.where(is_credit, "Credit", "Debit")
    else:
        return None, "No amount column found (looked for Amount / Debit / Credit...)."

    if cat_col:
        info["column_mapping"]["category"] = cat_col
        out["category"] = raw[cat_col].astype("string").str.strip()
    else:
        info["column_mapping"]["category"] = "Not found (auto-categorized)"
        out["category"] = pd.NA

    out["source_file"] = source_file

    # Drop rows lacking a usable date or amount (vectorized).
    bad_date = out["date"].isna()
    bad_amt = out["amount"].isna()
    drops = int((bad_date | bad_amt).sum())
    if int(bad_date.sum()):
        info["drop_reasons"].append(f"{int(bad_date.sum())} row(s): invalid/missing date")
    if int((bad_amt & ~bad_date).sum()):
        info["drop_reasons"].append(f"{int((bad_amt & ~bad_date).sum())} row(s): invalid/missing amount")
    out = out[~(bad_date | bad_amt)].reset_index(drop=True)

    out, dupes = drop_duplicate_transactions(out)
    info["duplicates_removed"] = dupes
    if dupes:
        info["drop_reasons"].append(f"{dupes} duplicate row(s) removed")

    if out.empty:
        return None, "No valid transactions found after cleaning."

    info["uncategorized_count"] = int(
        out["category"].isna().sum()
        + out["category"].astype("string").str.lower().eq("uncategorized").sum()
    )
    info["valid_rows"] = int(len(out))
    info["dropped_rows"] = info["total_rows"] - info["valid_rows"]
    info["date_range"] = f"{out['date'].min().date()} to {out['date'].max().date()}"
    log.info("Normalized %s: %d/%d rows kept", source_file, info["valid_rows"], info["total_rows"])
    return out, info
