"""Reusable, mostly-vectorized cleaning utilities.

Handles currency symbols, parenthesis negatives, thousands separators,
missing values, mixed/invalid dates, multi-line descriptions and duplicate
transactions. Designed for 20k+ rows: per-cell Python loops are avoided in
favor of pandas string/datetime vectorization.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

_CURRENCY_RE = re.compile(r"[^\d\-\.\(\)]")  # strip everything but digits/sign/parens/dot
_WS_RE = re.compile(r"\s+")


def clean_amount_series(s: pd.Series) -> pd.Series:
    """Vectorized parse of messy money strings into floats (NaN if unparseable).

    Handles ``$1,234.50``, ``(45.00)`` -> -45.00, ``1.234,50`` (EU) best-effort,
    trailing ``CR``/``DR`` markers and stray whitespace.
    """
    raw = s.astype("string").fillna("")
    txt = raw.str.strip()

    # Trailing CR/DR markers -> sign hints (CR positive, DR negative)
    is_dr = txt.str.contains(r"\bdr\b", case=False, regex=True, na=False)
    is_cr = txt.str.contains(r"\bcr\b", case=False, regex=True, na=False)
    txt = txt.str.replace(r"\b[dc]r\b", "", case=False, regex=True)

    # Parentheses => negative
    paren = txt.str.match(r"^\s*\(.*\)\s*$", na=False)

    # European decimal heuristic: "1.234,50" -> "1234.50"
    eu_mask = txt.str.match(r"^[^\d]*\d{1,3}(\.\d{3})+,\d{1,2}[^\d]*$", na=False)
    txt = txt.mask(eu_mask, txt.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))

    # Strip currency symbols/letters/commas, keep digits, sign, dot, parens
    cleaned = txt.str.replace(_CURRENCY_RE, "", regex=True)
    cleaned = cleaned.str.replace("(", "", regex=False).str.replace(")", "", regex=False)
    cleaned = cleaned.replace("", np.nan)

    vals = pd.to_numeric(cleaned, errors="coerce")
    vals = vals.where(~paren, -vals.abs())
    vals = vals.where(~is_dr, -vals.abs())
    vals = vals.where(~is_cr, vals.abs())
    return vals


def parse_date_series(s: pd.Series) -> pd.Series:
    """Vectorized date parsing with a day-first retry for the failures only."""
    raw = s.astype("string").str.strip()
    # format="mixed" parses each value independently; pandas>=2.0 otherwise
    # locks onto one inferred format and silently NaTs mixed-format columns.
    out = pd.to_datetime(raw, errors="coerce", format="mixed")
    bad = out.isna() & raw.notna() & (raw != "")
    if bad.any():
        retry = pd.to_datetime(raw[bad], errors="coerce", format="mixed", dayfirst=True)
        out.loc[bad] = retry
    return out


def collapse_whitespace_series(s: pd.Series) -> pd.Series:
    """Collapse multi-line / repeated whitespace in descriptions to single spaces."""
    return (
        s.astype("string")
        .fillna("")
        .str.replace(r"[\r\n\t]+", " ", regex=True)
        .str.replace(_WS_RE, " ", regex=True)
        .str.strip()
    )


def drop_duplicate_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate (date, description, amount, type) rows.

    Returns the deduped frame and the number of rows removed.
    """
    if df.empty:
        return df, 0
    keys = [c for c in ("date", "description", "amount", "transaction_type") if c in df.columns]
    before = len(df)
    out = df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    return out, before - len(out)


def split_signed_amount(
    amount: pd.Series, spending_is_negative: bool = True
) -> tuple[pd.Series, pd.Series]:
    """Map a single signed amount column to (magnitude, transaction_type).

    ``spending_is_negative`` mirrors the existing dashboard toggle.
    """
    amt = pd.to_numeric(amount, errors="coerce")
    magnitude = amt.abs()
    if spending_is_negative:
        ttype = np.where(amt < 0, "Debit", "Credit")
    else:
        ttype = np.where(amt > 0, "Debit", "Credit")
    return magnitude, pd.Series(ttype, index=amount.index)
