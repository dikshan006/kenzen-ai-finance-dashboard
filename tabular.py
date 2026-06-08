"""CSV parser: robust to messy real-world bank exports.

Tries multiple encodings, separators and preamble offsets, then hands the
best raw frame to the shared tabular normalizer.
"""
from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import pandas as pd

from kenzen.parsers.tabular import normalize_table
from kenzen.utils.logging_config import get_logger
from kenzen.utils import schema

log = get_logger("parsers.csv")

_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]
_SEPARATORS = [",", ";", "\t", "|", None]  # None => python engine sniffs


def _score_raw(df: Optional[pd.DataFrame]) -> int:
    """Cheap heuristic: how many of our key fields are present in the header."""
    if df is None or len(df.columns) < 2:
        return -1
    cols = [str(c).strip() for c in df.columns]
    found = 0
    for field in ("date", "amount", "debit", "credit", "description"):
        if schema.best_column_for(cols, field):
            found += 1
    return found * 100 + len(df.columns)


def parse_csv(
    content: bytes,
    source_file: str = "upload.csv",
    spending_is_negative: bool = True,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Parse raw CSV bytes into the standard schema."""
    best_raw: Optional[pd.DataFrame] = None
    best_score = -1

    for enc in _ENCODINGS:
        for sep in _SEPARATORS:
            for skip in range(5):  # tolerate up to 4 preamble lines
                try:
                    kwargs = dict(
                        encoding=enc,
                        skip_blank_lines=True,
                        skiprows=skip,
                        dtype=str,
                        on_bad_lines="skip",
                    )
                    if sep is None:
                        kwargs.update(sep=None, engine="python")
                    else:
                        kwargs["sep"] = sep
                    candidate = pd.read_csv(io.BytesIO(content), **kwargs)
                except Exception:
                    continue
                sc = _score_raw(candidate)
                if sc > best_score:
                    best_score, best_raw = sc, candidate
                    if best_score >= 300:  # date+amount+desc all found: good enough
                        break
            if best_score >= 300:
                break
        if best_score >= 300:
            break

    if best_raw is None or best_score < 0:
        return None, "Could not parse CSV with any encoding/separator combination."
    return normalize_table(best_raw, source_file, spending_is_negative)
