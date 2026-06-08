"""Rule-based merchant categorization with confidence scores.

This is intentionally a thin, swappable layer: ``Categorizer.predict`` returns
(label, confidence) and could be replaced by an ML model implementing the same
interface without touching callers.

Category labels are aligned with the categories the existing dashboard and its
insights/mock data already use (Groceries, Dining, ... Income, Other).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

CATEGORIES: List[str] = [
    "Groceries", "Dining", "Transportation", "Shopping", "Rent",
    "Utilities", "Healthcare", "Travel", "Entertainment", "Income",
    "Investment", "Other",
]

# Keyword -> category. Lowercase, matched as word-ish substrings.
_RULES: Dict[str, List[str]] = {
    "Groceries": ["whole foods", "trader joe", "safeway", "kroger", "aldi", "costco",
                  "grocery", "supermarket", "wholefds", "walmart"],
    "Dining": ["starbucks", "chipotle", "mcdonald", "doordash", "uber eats", "ubereats",
               "grubhub", "restaurant", "cafe", "coffee", "pizza", "sushi", "thai",
               "bar ", "dunkin", "panera", "taco"],
    "Transportation": ["uber", "lyft", "shell", "chevron", "exxon", "bp ", "gas",
                       "parking", "metro", "transit", "toyota service", "auto", "fuel"],
    "Shopping": ["amazon", "target", "gap", "best buy", "ebay", "etsy", "nike",
                 "macy", "nordstrom", "store", "shop"],
    "Rent": ["rent", "landlord", "property mgmt", "apartment", "leasing", "zillow rent"],
    "Utilities": ["electric", "water bill", "gas bill", "internet", "comcast", "xfinity",
                  "at&t", "verizon", "t-mobile", "utility", "power", "pg&e", "con ed"],
    "Healthcare": ["pharmacy", "cvs", "walgreens", "clinic", "hospital", "dental",
                   "doctor", "medical", "health", "kaiser"],
    "Travel": ["airline", "delta", "united air", "american air", "hotel", "marriott",
               "hilton", "airbnb", "expedia", "booking.com", "flight"],
    "Entertainment": ["netflix", "spotify", "hulu", "disney", "cinema", "amc", "concert",
                      "steam", "playstation", "xbox", "hbo", "cinemark", "movie"],
    "Income": ["payroll", "salary", "direct deposit", "employer", "ach credit",
               "interest paid", "refund", "reimbursement"],
    "Investment": ["vanguard", "fidelity", "schwab", "robinhood", "coinbase", "etrade",
                   "brokerage", "401k", "ira ", "dividend"],
}


@dataclass
class Prediction:
    category: str
    confidence: float


class Categorizer:
    """Keyword + rule engine. Replaceable by an ML model with the same API."""

    def __init__(self) -> None:
        # Precompile one regex per category for speed on large frames.
        self._patterns: Dict[str, re.Pattern] = {
            cat: re.compile("|".join(re.escape(k) for k in kws), re.IGNORECASE)
            for cat, kws in _RULES.items()
        }

    def predict(self, description: str, txn_type: str = "Debit") -> Prediction:
        text = str(description or "")
        # Credits without a merchant signal default to Income, not Other.
        hits: List[Tuple[str, int]] = []
        for cat, pat in self._patterns.items():
            found = pat.findall(text)
            if found:
                hits.append((cat, len(found)))
        if hits:
            hits.sort(key=lambda x: x[1], reverse=True)
            top_cat, n = hits[0]
            confidence = min(99.0, 80.0 + 6.0 * n + (5.0 if len(hits) == 1 else 0.0))
            return Prediction(top_cat, round(confidence, 1))
        if txn_type == "Credit":
            return Prediction("Income", 70.0)
        return Prediction("Other", 40.0)

    def categorize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing/Uncategorized categories from descriptions.

        Existing categories from the source file are respected; only blanks,
        NaN and "Uncategorized" are (re)classified, so files that already carry
        good categories are left alone.
        """
        if "category" not in df.columns:
            df["category"] = pd.NA
        existing = df["category"].astype("string").str.strip()
        needs = existing.isna() | (existing == "") | existing.str.lower().eq("uncategorized")

        if needs.any():
            descs = df.loc[needs, "description"]
            types = df.loc[needs, "transaction_type"] if "transaction_type" in df.columns else "Debit"
            if isinstance(types, str):
                preds = [self.predict(d, "Debit") for d in descs]
            else:
                preds = [self.predict(d, t) for d, t in zip(descs, types)]
            df.loc[needs, "category"] = [p.category for p in preds]
            df.loc[needs, "category_confidence"] = [p.confidence for p in preds]
        df["category"] = df["category"].astype("string").fillna("Other")
        return df
