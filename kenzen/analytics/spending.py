"""Spending analytics engine — all vectorized pandas.

Computes every KPI, merchant breakdown, and category breakdown the dashboard
needs in a single pass over the data. No row-by-row loops.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def compute_advanced_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Full KPI suite computed from actual transaction data only.

    Returns a dict consumable by the dashboard's metric cards and new sections.
    """
    debits = df.loc[df["Type"] == "Debit", "Amount"]
    credits = df.loc[df["Type"] == "Credit", "Amount"]

    total_spending = float(debits.sum())
    total_income = float(credits.sum())
    net_cash_flow = total_income - total_spending
    tx_count = len(df)
    debit_count = int(len(debits))

    # Date range for burn-rate calculation
    date_min, date_max = df["Date"].min(), df["Date"].max()
    span_days = max(1, (date_max - date_min).days)
    months = max(1.0, span_days / 30.0)
    monthly_burn = total_spending / months

    avg_txn = float(debits.mean()) if debit_count > 0 else 0.0

    # Largest single transaction
    if debit_count > 0:
        idx_max = debits.idxmax()
        largest_amount = float(debits.loc[idx_max])
        largest_merchant = str(df.loc[idx_max, "Merchant"])
    else:
        largest_amount, largest_merchant = 0.0, ""

    # Income-to-expense ratio
    ie_ratio = (total_income / total_spending) if total_spending > 0 else float("inf")

    # Merchant aggregations (vectorized)
    debit_df = df[df["Type"] == "Debit"]
    merchant_stats = (
        debit_df.groupby("Merchant")
        .agg(
            txn_count=("Amount", "size"),
            total_spend=("Amount", "sum"),
            avg_spend=("Amount", "mean"),
        )
        .sort_values("total_spend", ascending=False)
    )
    top_merchant = merchant_stats.index[0] if len(merchant_stats) > 0 else ""
    top_merchant_spend = float(merchant_stats["total_spend"].iloc[0]) if len(merchant_stats) > 0 else 0.0
    most_visited = merchant_stats["txn_count"].idxmax() if len(merchant_stats) > 0 else ""
    most_visited_count = int(merchant_stats["txn_count"].max()) if len(merchant_stats) > 0 else 0

    # Small-purchase analysis
    small_mask = debits < 5.0
    small_count = int(small_mask.sum())
    small_total = float(debits[small_mask].sum())

    # Month-over-month (last 30 vs prior 30)
    today = date_max
    last30_start = today - timedelta(days=29)
    prev30_start = last30_start - timedelta(days=30)
    last30_spend = float(debit_df[debit_df["Date"] >= last30_start]["Amount"].sum())
    prev30_spend = float(debit_df[(debit_df["Date"] >= prev30_start) & (debit_df["Date"] < last30_start)]["Amount"].sum())
    monthly_delta = ((last30_spend - prev30_spend) / prev30_spend * 100) if prev30_spend > 0 else 0.0

    return {
        "total_spending": total_spending,
        "total_income": total_income,
        "net_cash_flow": net_cash_flow,
        "tx_count": tx_count,
        "debit_count": debit_count,
        "monthly_burn": monthly_burn,
        "avg_txn": avg_txn,
        "largest_amount": largest_amount,
        "largest_merchant": largest_merchant,
        "ie_ratio": ie_ratio,
        "top_merchant": top_merchant,
        "top_merchant_spend": top_merchant_spend,
        "most_visited": most_visited,
        "most_visited_count": most_visited_count,
        "small_count": small_count,
        "small_total": small_total,
        "last30_spend": last30_spend,
        "prev30_spend": prev30_spend,
        "monthly_delta": monthly_delta,
        "span_days": span_days,
    }


def merchant_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Merchant Analytics table: Merchant | Transactions | Total | Avg | Category | % of Budget."""
    debit_df = df[df["Type"] == "Debit"]
    if debit_df.empty:
        return pd.DataFrame()
    total = debit_df["Amount"].sum()

    # Pick the most-common category per merchant (vectorized)
    cat_mode = (
        debit_df.groupby("Merchant")["Category"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Other")
    )

    stats = (
        debit_df.groupby("Merchant")["Amount"]
        .agg(Transactions="count", Total_Spend="sum", Avg_Spend="mean")
        .sort_values("Total_Spend", ascending=False)
    )
    stats["Category"] = cat_mode
    stats["Pct_of_Budget"] = (stats["Total_Spend"] / total * 100).round(1)
    stats = stats.reset_index()
    stats.columns = ["Merchant", "Transactions", "Total Spend", "Avg Spend", "Category", "% of Budget"]
    stats["Total Spend"] = stats["Total Spend"].round(2)
    stats["Avg Spend"] = stats["Avg Spend"].round(2)
    return stats


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Category Breakdown table: Category | Transactions | Total | Avg | % of Spending."""
    debit_df = df[df["Type"] == "Debit"]
    if debit_df.empty:
        return pd.DataFrame()
    total = debit_df["Amount"].sum()

    stats = (
        debit_df.groupby("Category")["Amount"]
        .agg(Transactions="count", Total_Spend="sum", Avg_Spend="mean")
        .sort_values("Total_Spend", ascending=False)
    )
    stats["Pct"] = (stats["Total_Spend"] / total * 100).round(1)
    stats = stats.reset_index()
    stats.columns = ["Category", "Transactions", "Total Spend", "Avg Spend", "% of Spending"]
    stats["Total Spend"] = stats["Total Spend"].round(2)
    stats["Avg Spend"] = stats["Avg Spend"].round(2)
    return stats
