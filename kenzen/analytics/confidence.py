"""Extraction confidence scoring.

Produces a 0-100 score plus a factor breakdown for any candidate normalized
frame. Used by the PDF parser to choose the best of several extraction engines,
but engine-agnostic so it works for CSV/XLSX sanity checks too.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class ConfidenceReport:
    score: float
    factors: Dict[str, float] = field(default_factory=dict)
    rows: int = 0


def score_frame(df: pd.DataFrame) -> ConfidenceReport:
    """Score a normalized frame on date/amount recognition, volume and dupes.

    Factors (each 0-1, weighted):
        date_rate     - fraction of rows with a valid date
        amount_rate   - fraction of rows with a valid amount
        volume        - saturating reward for having transactions
        uniqueness    - 1 - duplicate_rate
        consistency   - rows where both date and amount parsed
    """
    if df is None or len(df) == 0:
        return ConfidenceReport(0.0, {"empty": 0.0}, 0)

    n = len(df)
    date_ok = df["date"].notna() if "date" in df.columns else pd.Series([False] * n)
    amt_ok = df["amount"].notna() if "amount" in df.columns else pd.Series([False] * n)

    date_rate = float(date_ok.mean())
    amount_rate = float(amt_ok.mean())
    consistency = float((date_ok & amt_ok).mean())
    volume = min(1.0, n / 15.0)  # saturates quickly; 15+ rows is "plenty"

    if {"date", "description", "amount"}.issubset(df.columns):
        dup_rate = float(df.duplicated(subset=["date", "description", "amount"]).mean())
    else:
        dup_rate = 0.0
    uniqueness = 1.0 - dup_rate

    weights = {
        "date_rate": 0.32,
        "amount_rate": 0.32,
        "consistency": 0.18,
        "volume": 0.10,
        "uniqueness": 0.08,
    }
    factors = {
        "date_rate": date_rate,
        "amount_rate": amount_rate,
        "consistency": consistency,
        "volume": volume,
        "uniqueness": uniqueness,
    }
    score = sum(factors[k] * w for k, w in weights.items()) * 100.0
    return ConfidenceReport(round(score, 1), {k: round(v, 3) for k, v in factors.items()}, n)
