"""Multi-stage PDF bank/credit-card statement extraction.

Stages are attempted in order and each candidate is normalized and scored;
the highest-confidence candidate wins:

    1. pdfplumber  (tables + text)
    2. Camelot     (lattice/stream)
    3. Tabula      (java)
    4. OCR         (pytesseract over rendered pages)

Every heavy dependency is imported lazily and wrapped, so a missing library
or system binary simply skips that stage instead of crashing the app. If no
stage clears the confidence floor, the caller falls back to demo data.
"""
from __future__ import annotations

import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import pandas as pd

from kenzen.analytics.confidence import score_frame
from kenzen.parsers.tabular import normalize_table
from kenzen.utils import schema
from kenzen.utils.logging_config import get_logger

log = get_logger("parsers.pdf")

# Confidence floor below which we treat extraction as failed.
MIN_CONFIDENCE = 55.0

_DATE_RE = re.compile(
    r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"          # 03/14/2025, 3-14-25
    r"|\d{4}-\d{2}-\d{2}"                          # 2025-03-14
    r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}"         # 14 Mar 2025
    r"|[A-Za-z]{3,9}\s+\d{1,2},?\s*\d{0,4})"      # Mar 14, 2025
)
_AMOUNT_RE = re.compile(r"\(?-?\$?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})\)?(?:\s?(?:CR|DR))?", re.IGNORECASE)
_CREDIT_HINT = re.compile(r"\b(cr|credit|deposit|refund|payment received|reversal)\b", re.IGNORECASE)
_NOISE = re.compile(
    r"(opening balance|closing balance|beginning balance|ending balance|"
    r"statement period|page \d+ of \d+|available credit|minimum payment|"
    r"total (?:debits|credits|fees)|balance forward)",
    re.IGNORECASE,
)


def _frame_from_text(text: str) -> Optional[pd.DataFrame]:
    """Regex a block of statement text into Date/Description/Debit/Credit rows.

    Heuristics (documented because statement layouts vary wildly):
      * a line must contain both a date and at least one money amount to count;
      * if a line has 2+ amounts, the last is treated as a running balance and
        the prior amount as the transaction amount;
      * credits are detected via CR / credit / deposit / refund markers or
        parenthesized amounts.
    """
    rows: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _NOISE.search(line):
            continue
        dm = _DATE_RE.search(line)
        amts = _AMOUNT_RE.findall(line)
        if not dm or not amts:
            continue
        date_str = dm.group(0)
        txn_amt = amts[-2] if len(amts) >= 2 else amts[-1]
        # description = text between the date and the first amount
        first_amt_pos = line.find(amts[0])
        desc = line[dm.end():first_amt_pos].strip(" -\t") or "Transaction"

        is_credit = bool(_CREDIT_HINT.search(line)) or (txn_amt.strip().startswith("(") and txn_amt.strip().endswith(")"))
        rows.append({
            "Date": date_str,
            "Description": desc,
            "Debit": "" if is_credit else txn_amt,
            "Credit": txn_amt if is_credit else "",
        })
    if not rows:
        return None
    return pd.DataFrame(rows)


def _frame_from_table_rows(rows: List[List]) -> Optional[pd.DataFrame]:
    """Build a frame from extracted table rows, sniffing the header row."""
    if not rows or len(rows) < 2:
        return None
    best_idx, best_hits = 0, -1
    for i, r in enumerate(rows[:5]):
        cells = [str(c or "") for c in r]
        hits = sum(
            1 for f in ("date", "amount", "debit", "credit", "description", "balance")
            if any(schema.match_field(c, f) >= 88 for c in cells)
        )
        if hits > best_hits:
            best_hits, best_idx = hits, i
    if best_hits <= 0:
        return None
    header = [str(c or f"col{j}").strip() for j, c in enumerate(rows[best_idx])]
    data = rows[best_idx + 1:]
    data = [r for r in data if any(str(c or "").strip() for c in r)]
    if not data:
        return None
    width = len(header)
    data = [(r + [""] * width)[:width] for r in data]
    return pd.DataFrame(data, columns=header)


# ----- extraction stages (each returns a list of raw candidate frames) -----

def _stage_pdfplumber(content: bytes) -> List[pd.DataFrame]:
    try:
        import pdfplumber  # lazy
    except Exception:
        log.info("pdfplumber unavailable; skipping")
        return []
    out: List[pd.DataFrame] = []
    try:
        import io
        text_chunks: List[str] = []
        all_table_rows: List[List] = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text_chunks.append(page.extract_text() or "")
                for tbl in (page.extract_tables() or []):
                    all_table_rows.extend(tbl)
        tf = _frame_from_table_rows(all_table_rows)
        if tf is not None:
            out.append(tf)
        xf = _frame_from_text("\n".join(text_chunks))
        if xf is not None:
            out.append(xf)
    except Exception as exc:
        log.warning("pdfplumber stage failed: %s", exc)
    return out


def _stage_camelot(path: str) -> List[pd.DataFrame]:
    try:
        import camelot  # lazy; needs ghostscript
    except Exception:
        log.info("camelot unavailable; skipping")
        return []
    out: List[pd.DataFrame] = []
    for flavor in ("lattice", "stream"):
        try:
            tables = camelot.read_pdf(path, pages="all", flavor=flavor)
            for t in tables:
                rows = [list(t.df.columns)] + t.df.values.tolist()
                f = _frame_from_table_rows(rows)
                if f is not None:
                    out.append(f)
        except Exception as exc:
            log.info("camelot %s failed: %s", flavor, exc)
    return out


def _stage_tabula(path: str) -> List[pd.DataFrame]:
    try:
        import tabula  # lazy; needs java
    except Exception:
        log.info("tabula unavailable; skipping")
        return []
    out: List[pd.DataFrame] = []
    try:
        dfs = tabula.read_pdf(path, pages="all", multiple_tables=True, pandas_options={"dtype": str})
        for df in dfs or []:
            rows = [list(df.columns)] + df.astype(str).values.tolist()
            f = _frame_from_table_rows(rows)
            if f is not None:
                out.append(f)
    except Exception as exc:
        log.info("tabula failed: %s", exc)
    return out


def _stage_ocr(content: bytes) -> List[pd.DataFrame]:
    try:
        import pytesseract  # lazy; needs tesseract binary
        from pdf2image import convert_from_bytes  # needs poppler
    except Exception:
        log.info("OCR stack unavailable; skipping")
        return []
    try:
        images = convert_from_bytes(content, dpi=200)
        text = "\n".join(pytesseract.image_to_string(img) for img in images)
        f = _frame_from_text(text)
        return [f] if f is not None else []
    except Exception as exc:
        log.info("OCR failed: %s", exc)
        return []


def parse_pdf(
    content: bytes,
    source_file: str = "upload.pdf",
    spending_is_negative: bool = True,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Run the multi-stage pipeline and return the best normalized result."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as fh:
            fh.write(content)
            tmp_path = fh.name

        best_df: Optional[pd.DataFrame] = None
        best_info: Dict = {}
        best_score = -1.0
        best_engine = None

        # Stages are ordered cheap->expensive; stop as soon as one clears a
        # strong bar so we don't pay for OCR when pdfplumber already nailed it.
        stages = [
            ("pdfplumber", lambda: _stage_pdfplumber(content)),
            ("camelot", lambda: _stage_camelot(tmp_path)),
            ("tabula", lambda: _stage_tabula(tmp_path)),
            ("ocr", lambda: _stage_ocr(content)),
        ]
        for engine, run in stages:
            for raw in run():
                norm, info = normalize_table(raw, source_file, spending_is_negative)
                if not isinstance(norm, pd.DataFrame):
                    continue
                rep = score_frame(norm)
                if rep.score > best_score:
                    best_score, best_df, best_engine = rep.score, norm, engine
                    best_info = dict(info)
                    best_info["confidence"] = rep.score
                    best_info["confidence_factors"] = rep.factors
                    best_info["extraction_engine"] = engine
            if best_score >= 85.0:
                break

        if best_df is None:
            return None, "Could not extract any transaction table from the PDF."
        if best_score < MIN_CONFIDENCE:
            return None, (
                f"Low extraction confidence ({best_score:.0f}%) via {best_engine}; "
                "PDF layout not reliably parseable."
            )
        log.info("PDF parsed via %s at %.0f%% confidence", best_engine, best_score)
        return best_df, best_info
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
