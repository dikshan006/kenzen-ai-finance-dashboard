"""Unified ingestion entry point.

``load_transactions(file, spending_is_negative)`` auto-detects the file type,
routes to the right parser, runs categorization, and returns a dataframe in the
EXACT column shape the existing dashboard already consumes
(Date, Merchant, Category, Amount, Type) plus a diagnostics dict whose keys
match what the current Data-Quality expander reads.

Contract (drop-in replacement for the old ``parse_csv``):
    df, diagnostics = load_transactions(uploaded_file, spending_is_negative)
    * success -> (DataFrame, dict)
    * failure -> (None, "error message string")
So the app's existing ``if df is None: show demo`` branch keeps working.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import pandas as pd

from kenzen.classifiers.categorizer import Categorizer
from kenzen.parsers.csv_parser import parse_csv
from kenzen.parsers.excel_parser import parse_excel
from kenzen.parsers.pdf_parser import parse_pdf
from kenzen.utils.logging_config import get_logger

log = get_logger("ingestion")
_CATEGORIZER = Categorizer()

LEGACY_COLUMNS = ["Date", "Merchant", "Category", "Amount", "Type"]


def _detect_kind(name: str, content: bytes) -> Optional[str]:
    """Detect csv/xlsx/xls/pdf from extension first, then content sniff."""
    lname = (name or "").lower()
    if lname.endswith(".csv") or lname.endswith(".tsv") or lname.endswith(".txt"):
        return "csv"
    if lname.endswith(".xlsx") or lname.endswith(".xlsm"):
        return "xlsx"
    if lname.endswith(".xls"):
        return "xls"
    if lname.endswith(".pdf"):
        return "pdf"
    head = content[:8] if content else b""
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):  # zip -> xlsx
        return "xlsx"
    if head.startswith(b"\xd0\xcf\x11\xe0"):  # OLE2 -> xls
        return "xls"
    return "csv"  # last resort: assume delimited text


def to_legacy_ui_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map the standard schema to the dashboard's existing column names."""
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["date"]),
        "Merchant": df["description"].astype(str),
        "Category": df["category"].astype(str),
        "Amount": df["amount"].round(2).astype(float),
        "Type": df["transaction_type"].astype(str),
    })
    return out.sort_values("Date").reset_index(drop=True)


def load_transactions(
    file: Union[bytes, "object"],
    spending_is_negative: bool = True,
) -> Tuple[Optional[pd.DataFrame], Union[Dict, str]]:
    """Load any supported file into the legacy dashboard dataframe.

    ``file`` may be a Streamlit UploadedFile (has ``.name`` and ``.getvalue()``)
    or a (content, name) handled by callers. Returns (df|None, diagnostics|err).
    """
    try:
        name = getattr(file, "name", "upload.csv")
        content = file.getvalue() if hasattr(file, "getvalue") else file
        if not content:
            return None, "Uploaded file is empty."

        kind = _detect_kind(name, content)
        log.info("Ingesting %s as %s (%d bytes)", name, kind, len(content))

        if kind == "csv":
            norm, info = parse_csv(content, name, spending_is_negative)
        elif kind in ("xlsx", "xls"):
            norm, info = parse_excel(content, name, spending_is_negative)
        elif kind == "pdf":
            norm, info = parse_pdf(content, name, spending_is_negative)
        else:
            return None, f"Unsupported file type: {kind}"

        if not isinstance(norm, pd.DataFrame):
            # info holds the human-readable error string in the failure path
            return None, info

        # Smart categorization fills blanks / 'Uncategorized' only.
        norm = _CATEGORIZER.categorize_frame(norm)
        if isinstance(info, dict):
            info["uncategorized_count"] = int(
                norm["category"].astype("string").str.lower().eq("other").sum()
            )

        legacy = to_legacy_ui_frame(norm)
        return legacy, info
    except Exception as exc:  # never crash the app; let it fall back to demo
        log.exception("Ingestion failed")
        return None, f"Ingestion error: {exc}"
