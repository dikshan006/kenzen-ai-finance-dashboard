"""Production merchant categorization engine.

Three-tier classification:
  1. Exact merchant lookup (120+ mappings, highest confidence)
  2. Keyword/regex fallback (category rules)
  3. Fuzzy merchant matching via rapidfuzz/difflib (lowest tier)

Every prediction carries a confidence score. The ``Categorizer`` API is
unchanged: ``predict(description, txn_type)`` → ``Prediction``, and
``categorize_frame(df)`` fills blanks vectorized.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Try rapidfuzz; fallback to difflib
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz as _rf
    def _fuzzy_score(a: str, b: str) -> float:
        return float(_rf.token_set_ratio(a.lower(), b.lower()))
    _FUZZY_BACKEND = "rapidfuzz"
except Exception:
    from difflib import SequenceMatcher
    def _fuzzy_score(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100.0
    _FUZZY_BACKEND = "difflib"

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
CATEGORIES: List[str] = [
    "Groceries", "Dining", "Transportation", "Shopping", "Rent",
    "Utilities", "Healthcare", "Travel", "Entertainment", "Subscriptions",
    "Education", "Gas & Convenience", "Income", "Investment",
    "Fees & Charges", "Personal Care", "Gifts & Donations", "Other",
]

# ---------------------------------------------------------------------------
# Tier 1 — Exact merchant lookup (120+ entries)
# Key: lowercased merchant fragment → (category, confidence)
# ---------------------------------------------------------------------------
_MERCHANT_DB: Dict[str, Tuple[str, float]] = {
    # ---- Groceries ----
    "whole foods":        ("Groceries", 99), "trader joe":         ("Groceries", 99),
    "safeway":            ("Groceries", 98), "kroger":             ("Groceries", 98),
    "aldi":               ("Groceries", 98), "costco":             ("Groceries", 95),
    "sam's club":         ("Groceries", 95), "publix":             ("Groceries", 98),
    "wegmans":            ("Groceries", 98), "heb":                ("Groceries", 97),
    "food lion":          ("Groceries", 98), "piggly wiggly":      ("Groceries", 97),
    "sprouts":            ("Groceries", 97), "meijer":             ("Groceries", 96),
    "winco":              ("Groceries", 97), "hy-vee":             ("Groceries", 97),
    "giant eagle":        ("Groceries", 97), "stop & shop":        ("Groceries", 97),
    "albertsons":         ("Groceries", 97), "shoprite":           ("Groceries", 97),
    "grocery":            ("Groceries", 90), "supermarket":        ("Groceries", 90),
    "wholefds":           ("Groceries", 97), "fresh market":       ("Groceries", 96),
    "food":               ("Groceries", 70),

    # ---- Dining ----
    "starbucks":          ("Dining", 99),  "chipotle":           ("Dining", 99),
    "mcdonald":           ("Dining", 99),  "burger king":        ("Dining", 99),
    "wendy's":            ("Dining", 99),  "wendys":             ("Dining", 99),
    "taco bell":          ("Dining", 99),  "subway":             ("Dining", 98),
    "domino":             ("Dining", 98),  "papa john":          ("Dining", 98),
    "pizza hut":          ("Dining", 98),  "panera":             ("Dining", 98),
    "chick-fil-a":        ("Dining", 99),  "chick fil a":        ("Dining", 99),
    "popeyes":            ("Dining", 98),  "five guys":          ("Dining", 98),
    "panda express":      ("Dining", 98),  "olive garden":       ("Dining", 98),
    "applebee":           ("Dining", 98),  "ihop":               ("Dining", 98),
    "waffle house":       ("Dining", 98),  "denny":              ("Dining", 97),
    "dunkin":             ("Dining", 98),  "doordash":           ("Dining", 97),
    "uber eats":          ("Dining", 97),  "ubereats":           ("Dining", 97),
    "grubhub":            ("Dining", 97),  "postmates":          ("Dining", 97),
    "instacart":          ("Dining", 85),  "restaurant":         ("Dining", 90),
    "cafe":               ("Dining", 88),  "coffee":             ("Dining", 85),
    "pizza":              ("Dining", 88),  "sushi":              ("Dining", 90),
    "thai":               ("Dining", 88),  "chinese":            ("Dining", 85),
    "bakery":             ("Dining", 85),  "diner":              ("Dining", 88),
    "grill":              ("Dining", 85),  "buffet":             ("Dining", 88),
    "noodle":             ("Dining", 85),

    # ---- Gas & Convenience ----
    "kwik star":          ("Gas & Convenience", 97),
    "kwikstar":           ("Gas & Convenience", 97),
    "casey":              ("Gas & Convenience", 96),
    "kum & go":           ("Gas & Convenience", 96),
    "circle k":           ("Gas & Convenience", 96),
    "7-eleven":           ("Gas & Convenience", 96),
    "7 eleven":           ("Gas & Convenience", 96),
    "wawa":               ("Gas & Convenience", 96),
    "sheetz":             ("Gas & Convenience", 96),
    "quiktrip":           ("Gas & Convenience", 96),
    "pilot":              ("Gas & Convenience", 94),
    "loves travel":       ("Gas & Convenience", 94),
    "flying j":           ("Gas & Convenience", 94),
    "racetrac":           ("Gas & Convenience", 95),
    "speedway":           ("Gas & Convenience", 95),
    "bp ":                ("Gas & Convenience", 90),
    "decorah mart":       ("Gas & Convenience", 95),
    "convenience":        ("Gas & Convenience", 85),
    "c-store":            ("Gas & Convenience", 85),

    # ---- Transportation ----
    "uber":               ("Transportation", 92),
    "lyft":               ("Transportation", 97),
    "shell":              ("Transportation", 92),
    "chevron":            ("Transportation", 95),
    "exxon":              ("Transportation", 95),
    "mobil":              ("Transportation", 94),
    "sunoco":             ("Transportation", 94),
    "valero":             ("Transportation", 94),
    "marathon":           ("Transportation", 90),
    "gas":                ("Transportation", 80),
    "parking":            ("Transportation", 90),
    "metro":              ("Transportation", 80),
    "transit":            ("Transportation", 85),
    "fuel":               ("Transportation", 85),
    "toyota service":     ("Transportation", 90),
    "jiffy lube":         ("Transportation", 92),
    "autozone":           ("Transportation", 90),
    "tire":               ("Transportation", 85),

    # ---- Shopping ----
    "amazon":             ("Shopping", 92),
    "walmart":            ("Shopping", 88),
    "target":             ("Shopping", 94),
    "best buy":           ("Shopping", 97),
    "ebay":               ("Shopping", 95),
    "etsy":               ("Shopping", 95),
    "nike":               ("Shopping", 95),
    "adidas":             ("Shopping", 95),
    "macy":               ("Shopping", 95),
    "nordstrom":          ("Shopping", 95),
    "gap":                ("Shopping", 95),
    "old navy":           ("Shopping", 95),
    "h&m":                ("Shopping", 95),
    "zara":               ("Shopping", 95),
    "ikea":               ("Shopping", 95),
    "home depot":         ("Shopping", 95),
    "lowe's":             ("Shopping", 95),
    "lowes":              ("Shopping", 95),
    "dollar tree":        ("Shopping", 92),
    "dollar general":     ("Shopping", 92),
    "tj maxx":            ("Shopping", 94),
    "marshalls":          ("Shopping", 94),
    "ross":               ("Shopping", 93),
    "bed bath":           ("Shopping", 93),
    "store":              ("Shopping", 65),

    # ---- Subscriptions ----
    "apple.com":          ("Subscriptions", 97),
    "apple.com/bill":     ("Subscriptions", 99),
    "openai":             ("Subscriptions", 97),
    "chatgpt":            ("Subscriptions", 97),
    "google storage":     ("Subscriptions", 96),
    "google one":         ("Subscriptions", 96),
    "microsoft 365":      ("Subscriptions", 96),
    "dropbox":            ("Subscriptions", 96),
    "adobe":              ("Subscriptions", 95),
    "icloud":             ("Subscriptions", 96),
    "youtube premium":    ("Subscriptions", 97),
    "patreon":            ("Subscriptions", 94),
    "substack":           ("Subscriptions", 94),

    # ---- Entertainment ----
    "netflix":            ("Entertainment", 99),
    "spotify":            ("Entertainment", 99),
    "hulu":               ("Entertainment", 99),
    "disney":             ("Entertainment", 97),
    "hbo":                ("Entertainment", 98),
    "paramount":          ("Entertainment", 96),
    "peacock":            ("Entertainment", 96),
    "cinema":             ("Entertainment", 95),
    "amc":                ("Entertainment", 95),
    "regal":              ("Entertainment", 95),
    "cinemark":           ("Entertainment", 95),
    "steam":              ("Entertainment", 94),
    "playstation":        ("Entertainment", 95),
    "xbox":               ("Entertainment", 95),
    "nintendo":           ("Entertainment", 95),
    "twitch":             ("Entertainment", 93),
    "concert":            ("Entertainment", 92),
    "movie":              ("Entertainment", 88),
    "ticket":             ("Entertainment", 80),

    # ---- Rent ----
    "rent":               ("Rent", 92),
    "landlord":           ("Rent", 92),
    "property mgmt":      ("Rent", 92),
    "apartment":          ("Rent", 88),
    "leasing":            ("Rent", 85),
    "zillow rent":        ("Rent", 94),

    # ---- Utilities ----
    "electric":           ("Utilities", 92),
    "water bill":         ("Utilities", 95),
    "gas bill":           ("Utilities", 92),
    "internet":           ("Utilities", 90),
    "comcast":            ("Utilities", 96),
    "xfinity":            ("Utilities", 96),
    "at&t":               ("Utilities", 92),
    "verizon":            ("Utilities", 93),
    "t-mobile":           ("Utilities", 93),
    "sprint":             ("Utilities", 93),
    "utility":            ("Utilities", 90),
    "power":              ("Utilities", 82),
    "pg&e":               ("Utilities", 96),
    "con ed":             ("Utilities", 95),

    # ---- Healthcare ----
    "pharmacy":           ("Healthcare", 92),
    "cvs":                ("Healthcare", 95),
    "walgreens":          ("Healthcare", 93),
    "rite aid":           ("Healthcare", 93),
    "clinic":             ("Healthcare", 90),
    "hospital":           ("Healthcare", 95),
    "dental":             ("Healthcare", 92),
    "doctor":             ("Healthcare", 90),
    "medical":            ("Healthcare", 88),
    "health":             ("Healthcare", 75),
    "kaiser":             ("Healthcare", 95),
    "optometrist":        ("Healthcare", 92),
    "urgent care":        ("Healthcare", 95),

    # ---- Travel ----
    "airline":            ("Travel", 92),
    "delta":              ("Travel", 90),
    "united air":         ("Travel", 95),
    "american air":       ("Travel", 95),
    "southwest":          ("Travel", 94),
    "jetblue":            ("Travel", 95),
    "hotel":              ("Travel", 90),
    "marriott":           ("Travel", 97),
    "hilton":             ("Travel", 97),
    "hyatt":              ("Travel", 97),
    "airbnb":             ("Travel", 96),
    "expedia":            ("Travel", 95),
    "booking.com":        ("Travel", 95),
    "flight":             ("Travel", 85),
    "tsa":                ("Travel", 92),

    # ---- Education ----
    "luther college":     ("Education", 99),
    "college":            ("Education", 88),
    "university":         ("Education", 90),
    "tuition":            ("Education", 95),
    "school":             ("Education", 82),
    "textbook":           ("Education", 90),
    "chegg":              ("Education", 92),
    "coursera":           ("Education", 92),
    "udemy":              ("Education", 92),

    # ---- Income ----
    "payroll":            ("Income", 97),
    "salary":             ("Income", 97),
    "direct deposit":     ("Income", 96),
    "employer":           ("Income", 95),
    "ach credit":         ("Income", 90),
    "interest paid":      ("Income", 88),
    "refund":             ("Income", 80),
    "reimbursement":      ("Income", 85),
    "zelle":              ("Income", 70),
    "venmo":              ("Income", 65),

    # ---- Investment ----
    "vanguard":           ("Investment", 97),
    "fidelity":           ("Investment", 97),
    "schwab":             ("Investment", 97),
    "robinhood":          ("Investment", 97),
    "coinbase":           ("Investment", 96),
    "etrade":             ("Investment", 96),
    "brokerage":          ("Investment", 90),
    "401k":               ("Investment", 95),
    "dividend":           ("Investment", 88),

    # ---- Fees & Charges ----
    "overdraft":          ("Fees & Charges", 97),
    "monthly fee":        ("Fees & Charges", 96),
    "service charge":     ("Fees & Charges", 95),
    "atm fee":            ("Fees & Charges", 96),
    "late fee":           ("Fees & Charges", 97),
    "nsf fee":            ("Fees & Charges", 97),
    "annual fee":         ("Fees & Charges", 96),
    "interest charge":    ("Fees & Charges", 92),
    "finance charge":     ("Fees & Charges", 95),

    # ---- Personal Care ----
    "salon":              ("Personal Care", 92),
    "barber":             ("Personal Care", 92),
    "spa":                ("Personal Care", 90),
    "nail":               ("Personal Care", 85),
    "beauty":             ("Personal Care", 85),
    "sephora":            ("Personal Care", 95),
    "ulta":               ("Personal Care", 95),
    "gym":                ("Personal Care", 88),
    "fitness":            ("Personal Care", 88),
    "planet fitness":     ("Personal Care", 96),

    # ---- Gifts & Donations ----
    "charity":            ("Gifts & Donations", 92),
    "donation":           ("Gifts & Donations", 92),
    "gofundme":           ("Gifts & Donations", 94),
    "church":             ("Gifts & Donations", 80),
    "tithe":              ("Gifts & Donations", 92),
}


# ---------------------------------------------------------------------------
# Tier 2 — keyword fallback (merged from Tier 1 keys, used when DB misses)
# ---------------------------------------------------------------------------
def _build_keyword_rules() -> Dict[str, re.Pattern]:
    """Group Tier-1 keys by category into one compiled regex each."""
    buckets: Dict[str, List[str]] = {}
    for kw, (cat, _) in _MERCHANT_DB.items():
        buckets.setdefault(cat, []).append(re.escape(kw.strip()))
    return {cat: re.compile("|".join(sorted(kws, key=len, reverse=True)), re.IGNORECASE)
            for cat, kws in buckets.items()}

_KW_PATTERNS = _build_keyword_rules()


# ---------------------------------------------------------------------------
# Tier 3 — fuzzy merchant matching (top 5 DB keys by similarity)
# ---------------------------------------------------------------------------
_FUZZY_THRESHOLD = 78.0
_FUZZY_KEYS: List[Tuple[str, str, float]] = [
    (k, cat, conf) for k, (cat, conf) in _MERCHANT_DB.items() if len(k) > 3
]


@dataclass
class Prediction:
    category: str
    confidence: float
    tier: str = "unknown"  # "exact", "keyword", "fuzzy", "default"


class Categorizer:
    """Three-tier merchant categorization engine.

    Tier 1: exact substring match in merchant DB (highest confidence).
    Tier 2: regex keyword fallback by category.
    Tier 3: fuzzy string matching against DB (slowest, lowest confidence).
    """

    def __init__(self) -> None:
        self._db = _MERCHANT_DB
        self._kw = _KW_PATTERNS

    def predict(self, description: str, txn_type: str = "Debit") -> Prediction:
        text = str(description or "").strip()
        text_lower = text.lower()
        if not text_lower:
            if txn_type == "Credit":
                return Prediction("Income", 70.0, "default")
            return Prediction("Other", 40.0, "default")

        # --- Tier 1: exact merchant lookup (longest-match-first) ---
        best_exact: Optional[Tuple[str, float]] = None
        best_len = 0
        for key, (cat, conf) in self._db.items():
            if key in text_lower and len(key) > best_len:
                best_exact = (cat, conf)
                best_len = len(key)
        if best_exact:
            return Prediction(best_exact[0], best_exact[1], "exact")

        # --- Tier 2: keyword regex per category ---
        hits: List[Tuple[str, int]] = []
        for cat, pat in self._kw.items():
            found = pat.findall(text)
            if found:
                hits.append((cat, sum(len(m) for m in found)))
        if hits:
            hits.sort(key=lambda x: x[1], reverse=True)
            return Prediction(hits[0][0], min(88.0, 70.0 + hits[0][1]), "keyword")

        # --- Tier 3: fuzzy match ---
        best_fuzzy: Optional[Tuple[str, float, float]] = None
        for key, cat, db_conf in _FUZZY_KEYS:
            sc = _fuzzy_score(text_lower, key)
            if sc >= _FUZZY_THRESHOLD:
                eff = sc * 0.6 + db_conf * 0.4  # blend string sim with DB conf
                if best_fuzzy is None or eff > best_fuzzy[2]:
                    best_fuzzy = (cat, sc, eff)
        if best_fuzzy:
            return Prediction(best_fuzzy[0], round(min(85.0, best_fuzzy[2]), 1), "fuzzy")

        # --- Tier 4: default ---
        if txn_type == "Credit":
            return Prediction("Income", 70.0, "default")
        return Prediction("Other", 40.0, "default")

    def categorize_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing / 'Uncategorized' categories from descriptions.

        Existing categories from the source file are respected; only blanks,
        NaN and 'Uncategorized' are (re)classified.
        """
        df = df.copy()
        if "category" not in df.columns:
            df["category"] = pd.NA
        existing = df["category"].astype("string").str.strip()
        needs = existing.isna() | (existing == "") | existing.str.lower().eq("uncategorized")

        if needs.any():
            descs = df.loc[needs, "description"]
            types = df.loc[needs, "transaction_type"] if "transaction_type" in df.columns else pd.Series("Debit", index=descs.index)
            if isinstance(types, str):
                types = pd.Series(types, index=descs.index)
            preds = [self.predict(d, t) for d, t in zip(descs, types)]
            df.loc[needs, "category"] = [p.category for p in preds]
            df.loc[needs, "category_confidence"] = [p.confidence for p in preds]
        df["category"] = df["category"].astype("string").fillna("Other")
        return df
