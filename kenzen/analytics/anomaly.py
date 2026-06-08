"""Advanced anomaly detection engine — vectorized for 25k+ rows.

Six detection methods with human-readable explanations.
Group-level iteration is limited to merchants meeting filter criteria.
Row-level iteration is eliminated. Results are capped to keep the UI clean.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Maximum anomalies to surface so the UI stays usable.
_MAX_RESULTS = 25


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Run all detectors; return consolidated, deduped, capped DataFrame."""
    debits = df[df["Type"] == "Debit"].copy()
    if debits.empty:
        return pd.DataFrame(columns=["Date", "Merchant", "Category", "Amount", "Type", "Explanation"])

    parts: List[pd.DataFrame] = [
        _merchant_outliers(debits),
        _large_purchases(debits),
        _repeated_merchant_bursts(debits),
        _subscription_detection(debits),
        _category_spikes(debits),
        _multi_charge_detection(debits),
    ]
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=["Date", "Merchant", "Category", "Amount", "Type", "Explanation"])

    # Dedupe: one anomaly per (Date, Merchant, raw_amt), keep first (most informative)
    if "raw_amt" in out.columns:
        out = out.drop_duplicates(subset=["Date", "Merchant", "raw_amt"], keep="first")
        out = out.drop(columns=["raw_amt"], errors="ignore")
    out = out.sort_values("Date", ascending=False).head(_MAX_RESULTS).reset_index(drop=True)
    return out


def _fmt_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 1. Merchant-specific outliers — vectorized groupby + merge
# ---------------------------------------------------------------------------
def _merchant_outliers(debits: pd.DataFrame) -> pd.DataFrame:
    # Pre-compute group stats (no per-group lambda)
    gstats = debits.groupby("Merchant")["Amount"].agg(
        _count="count", _mean="mean", _q1=lambda x: x.quantile(0.25),
        _q3=lambda x: x.quantile(0.75)
    )
    gstats = gstats[gstats["_count"] >= 3].copy()
    if gstats.empty:
        return pd.DataFrame()
    gstats["_iqr"] = gstats["_q3"] - gstats["_q1"]
    gstats["_upper"] = gstats["_q3"] + 2.0 * gstats["_iqr"]
    gstats["_upper"] = gstats[["_upper", "_mean"]].max(axis=1) * 1.0
    # Where IQR is too tight, use 2.5× mean
    tight = gstats["_upper"] <= gstats["_mean"]
    gstats.loc[tight, "_upper"] = gstats.loc[tight, "_mean"] * 2.5

    sub = debits[debits["Merchant"].isin(gstats.index)].merge(
        gstats[["_mean", "_upper"]], left_on="Merchant", right_index=True
    )
    outliers = sub[sub["Amount"] > sub["_upper"]].copy()
    if outliers.empty:
        return pd.DataFrame()

    ratio = (outliers["Amount"] / outliers["_mean"]).round(1)
    outliers["Explanation"] = (
        outliers["Merchant"] + " charge of $" + outliers["Amount"].apply(lambda x: f"{x:,.2f}")
        + " is " + ratio.astype(str) + "x your average "
        + outliers["Merchant"] + " transaction ($"
        + outliers["_mean"].apply(lambda x: f"{x:,.2f}") + ")."
    )
    return _package(outliers, "Merchant Outlier")


# ---------------------------------------------------------------------------
# 2. Unusually large purchases — vectorized
# ---------------------------------------------------------------------------
def _large_purchases(debits: pd.DataFrame) -> pd.DataFrame:
    mean = debits["Amount"].mean()
    std = debits["Amount"].std()
    if std == 0 or pd.isna(std):
        return pd.DataFrame()
    threshold = mean + 2.0 * std
    big = debits[debits["Amount"] > threshold].copy()
    if big.empty:
        return pd.DataFrame()
    big["Explanation"] = (
        "$" + big["Amount"].apply(lambda x: f"{x:,.2f}") + " at " + big["Merchant"]
        + f" is significantly above your overall average transaction of ${mean:,.2f}."
    )
    return _package(big, "Large Purchase")


# ---------------------------------------------------------------------------
# 3. Repeated merchant bursts — optimized group-level, not row-level
# ---------------------------------------------------------------------------
def _repeated_merchant_bursts(debits: pd.DataFrame) -> pd.DataFrame:
    # Only check merchants with ≥ 3 transactions
    counts = debits.groupby("Merchant").size()
    candidates = counts[counts >= 3].index
    if len(candidates) == 0:
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []
    for merchant in candidates:
        grp = debits[debits["Merchant"] == merchant].sort_values("Date")
        dates = grp["Date"].values  # numpy datetime64 for speed
        n = len(dates)
        # Sliding window: find max count in any 10-day window
        best_count, best_start, best_end = 0, 0, 0
        j = 0
        for i in range(n):
            while j < n and (dates[j] - dates[i]) <= np.timedelta64(10, "D"):
                j += 1
            if (j - i) > best_count:
                best_count = j - i
                best_start, best_end = i, j
            if best_count >= 3:
                break  # good enough, stop scanning this merchant
        if best_count >= 3:
            window_rows = grp.iloc[best_start:best_end]
            total = window_rows["Amount"].sum()
            results.append({
                "Date": grp["Date"].iloc[best_start].strftime("%Y-%m-%d"),
                "Merchant": merchant,
                "Category": grp["Category"].iloc[0],
                "Amount": f"${total:,.2f} total",
                "raw_amt": total,
                "Type": "Repeated Visits",
                "Explanation": (
                    f"{merchant} was visited {best_count} times within 10 days "
                    f"(${total:,.2f} total)."
                ),
            })
        if len(results) >= 10:
            break  # cap to keep detection fast
    return pd.DataFrame(results) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. Subscription / recurring detection — vectorized
# ---------------------------------------------------------------------------
def _subscription_detection(debits: pd.DataFrame) -> pd.DataFrame:
    grp = debits.groupby("Merchant")["Amount"].agg(["count", "mean", "std"])
    grp["std"] = grp["std"].fillna(0)
    grp["cv"] = grp["std"] / grp["mean"].replace(0, np.nan)
    recurring = grp[(grp["count"] >= 2) & (grp["cv"] < 0.10)].copy()
    if recurring.empty:
        return pd.DataFrame()

    # One row per merchant: latest date
    latest = debits[debits["Merchant"].isin(recurring.index)].sort_values("Date").groupby("Merchant").tail(1).copy()
    latest = latest.merge(recurring[["count", "mean"]], left_on="Merchant", right_index=True, suffixes=("", "_grp"))
    latest["Explanation"] = (
        latest["Merchant"] + " charged " + latest["count"].astype(int).astype(str)
        + " times at ~$" + latest["mean"].apply(lambda x: f"{x:,.2f}")
        + " each — likely a recurring payment."
    )
    latest["Amount"] = latest["mean"].apply(lambda x: f"${x:,.2f} each")
    return _package(latest, "Recurring/Subscription")


# ---------------------------------------------------------------------------
# 5. Category spikes — vectorized
# ---------------------------------------------------------------------------
def _category_spikes(debits: pd.DataFrame) -> pd.DataFrame:
    today = debits["Date"].max()
    recent_start = today - timedelta(days=13)
    prior_start = recent_start - timedelta(days=14)

    recent = debits[debits["Date"] >= recent_start].groupby("Category")["Amount"].sum()
    prior = debits[(debits["Date"] >= prior_start) & (debits["Date"] < recent_start)].groupby("Category")["Amount"].sum()

    if recent.empty or prior.empty:
        return pd.DataFrame()

    combined = pd.DataFrame({"recent": recent, "prior": prior}).dropna()
    combined = combined[combined["prior"] > 0]
    combined["pct"] = ((combined["recent"] - combined["prior"]) / combined["prior"] * 100).round(0)
    spikes = combined[combined["pct"] >= 50].sort_values("pct", ascending=False).head(5)
    if spikes.empty:
        return pd.DataFrame()

    rows = [{
        "Date": today.strftime("%Y-%m-%d"), "Merchant": f"[{cat}]",
        "Category": cat, "Amount": f"${r['recent']:,.2f}", "raw_amt": r["recent"],
        "Type": "Category Spike",
        "Explanation": f"{cat} spending increased {r['pct']:.0f}% vs the prior two weeks (${r['prior']:,.2f} → ${r['recent']:,.2f}).",
    } for cat, r in spikes.iterrows()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Multi-charge detection — fully vectorized
# ---------------------------------------------------------------------------
def _multi_charge_detection(debits: pd.DataFrame) -> pd.DataFrame:
    grp = debits.groupby("Merchant").agg(
        n=("Amount", "size"), total=("Amount", "sum"),
        category=("Category", "first"), last_date=("Date", "max"),
    )
    multi = grp[grp["n"] >= 3].sort_values("total", ascending=False).head(10)
    if multi.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        "Date": multi["last_date"].dt.strftime("%Y-%m-%d"),
        "Merchant": multi.index,
        "Category": multi["category"],
        "Amount": multi["total"].apply(lambda x: f"${x:,.2f} total"),
        "raw_amt": multi["total"],
        "Type": "Frequent Charges",
        "Explanation": multi.apply(
            lambda r: f"{r.name} charged {r['n']} times this statement period (${r['total']:,.2f} total).",
            axis=1
        ),
    }).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _package(src: pd.DataFrame, anom_type: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Date": _fmt_date(src["Date"]),
        "Merchant": src["Merchant"].values,
        "Category": src["Category"].values,
        "Amount": src["Amount"].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float, np.floating)) else x).values,
        "raw_amt": src["Amount"].apply(lambda x: float(x) if isinstance(x, (int, float, np.floating)) else 0).values,
        "Type": anom_type,
        "Explanation": src["Explanation"].values,
    })
