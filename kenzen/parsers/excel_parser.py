"""Excel parser (.xlsx / .xls).

Bank Excel exports often bury the real header a few rows down and may carry
the data on a non-first sheet. We sniff the best sheet + header row, then
normalize via the shared tabular path.
"""
from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import pandas as pd

from kenzen.parsers.tabular import normalize_table
from kenzen.utils import schema
from kenzen.utils.logging_config import get_logger

log = get_logger("parsers.excel")


def _score_raw(df: Optional[pd.DataFrame]) -> int:
    if df is None or len(df.columns) < 2:
        return -1
    cols = [str(c).strip() for c in df.columns]
    found = sum(
        1 for f in ("date", "amount", "debit", "credit", "description")
        if schema.best_column_for(cols, f)
    )
    return found * 100 + len(df.columns)


def parse_excel(
    content: bytes,
    source_file: str = "upload.xlsx",
    spending_is_negative: bool = True,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Parse raw Excel bytes into the standard schema."""
    try:
        engine = "xlrd" if source_file.lower().endswith(".xls") else "openpyxl"
        xls = pd.ExcelFile(io.BytesIO(content), engine=engine)
    except Exception as exc:  # pragma: no cover - depends on user file
        return None, f"Could not open Excel workbook: {exc}"

    best_raw: Optional[pd.DataFrame] = None
    best_score = -1
    for sheet in xls.sheet_names:
        for header_row in range(6):  # real header is usually within first 6 rows
            try:
                candidate = xls.parse(sheet, header=header_row, dtype=str)
            except Exception:
                continue
            sc = _score_raw(candidate)
            if sc > best_score:
                best_score, best_raw = sc, candidate

    if best_raw is None or best_score < 0:
        return None, "No usable transaction table found in the workbook."
    return normalize_table(best_raw, source_file, spending_is_negative)
