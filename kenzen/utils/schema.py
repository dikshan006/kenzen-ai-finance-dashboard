"""Canonical schema definitions and intelligent column matching.

The internal normalized schema is:

    date            -> pandas Timestamp
    description     -> str
    amount          -> float, positive magnitude
    transaction_type-> "Debit" | "Credit"
    category        -> str
    source_file     -> str

A small adapter (``to_legacy_ui_frame`` in ingestion.py) maps this to the
column names the existing dashboard already consumes (Date, Merchant,
Category, Amount, Type) so the UI never has to change.
"""
from __future__ import annotations

from typing import Dict, List, Optional

STANDARD_COLUMNS: List[str] = [
    "date",
    "description",
    "amount",
    "transaction_type",
    "category",
    "source_file",
]

# Synonyms used for header detection. Order does not matter; scoring handles
# ambiguity. No hardcoded positional assumptions about the source file.
FIELD_SYNONYMS: Dict[str, List[str]] = {
    "date": [
        "date",
        "transaction date",
        "posted date",
        "post date",
        "posting date",
        "trans date",
        "value date",
        "timestamp",
        "time",
    ],
    "description": [
        "description",
        "merchant",
        "name",
        "details",
        "detail",
        "memo",
        "narrative",
        "payee",
        "particulars",
        "reference",
        "transaction",
    ],
    "amount": ["amount", "amt", "value", "transaction amount"],
    "debit": ["debit", "withdrawal", "withdrawals", "money out", "paid out", "dr"],
    "credit": ["credit", "deposit", "deposits", "money in", "paid in", "cr"],
    "balance": ["balance", "running balance", "ledger balance", "available balance"],
    "category": ["category", "type", "transaction type", "classification"],
}

# Try rapidfuzz; fall back to stdlib difflib so the engine still works without it.
try:  # pragma: no cover - import guard
    from rapidfuzz import fuzz as _rf_fuzz

    def _similarity(a: str, b: str) -> float:
        return float(_rf_fuzz.token_set_ratio(a, b))

    FUZZY_BACKEND = "rapidfuzz"
except Exception:  # pragma: no cover - import guard
    from difflib import SequenceMatcher

    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio() * 100.0

    FUZZY_BACKEND = "difflib"


def _norm(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def match_field(column_name: str, field: str, threshold: float = 80.0) -> float:
    """Return the best similarity score (0-100) of ``column_name`` to ``field``."""
    col = _norm(column_name)
    if not col:
        return 0.0
    best = 0.0
    for syn in FIELD_SYNONYMS.get(field, [field]):
        syn_n = _norm(syn)
        # exact / substring shortcut beats fuzzy noise
        if col == syn_n:
            return 100.0
        if syn_n in col or col in syn_n:
            best = max(best, 92.0)
        best = max(best, _similarity(col, syn_n))
    return best if best >= threshold else best


def best_column_for(columns: List[str], field: str, threshold: float = 80.0,
                    exclude: Optional[List[str]] = None) -> Optional[str]:
    """Pick the most likely source column for a normalized ``field``.

    Returns ``None`` if no column clears ``threshold``. No positional
    assumptions are made; selection is purely score based.
    """
    exclude = set(exclude or [])
    scored = [
        (c, match_field(c, field, threshold))
        for c in columns
        if c not in exclude
    ]
    scored = [s for s in scored if s[1] >= threshold]
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]
