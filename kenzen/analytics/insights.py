"""Personalized financial insight generation.

Every insight cites actual dollar values and percentages from the parsed data.
All values are computed dynamically from the transaction DataFrame.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def generate_insights(df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
    """Return HTML-formatted insight strings, most important first.

    Every insight is backed by real numbers from the dataframe.
    """
    insights: List[str] = []
    debit_df = df[df["Type"] == "Debit"]
    if debit_df.empty:
        return ["<strong>No spending transactions found.</strong> Upload a statement to see insights."]

    total_spend = kpis["total_spending"]
    total_income = kpis["total_income"]

    # --- Cash flow summary ---
    if total_income > 0 and total_spend > 0:
        if kpis["net_cash_flow"] >= 0:
            insights.append(
                f"You received <strong>${total_income:,.2f}</strong> in deposits and spent "
                f"<strong>${total_spend:,.2f}</strong>, resulting in a "
                f"<strong>positive cash flow of ${kpis['net_cash_flow']:,.2f}</strong>."
            )
        else:
            insights.append(
                f"You received <strong>${total_income:,.2f}</strong> in deposits and spent "
                f"<strong>${total_spend:,.2f}</strong>, resulting in a "
                f"<strong>deficit of ${abs(kpis['net_cash_flow']):,.2f}</strong>."
            )
    elif total_spend > 0 and total_income == 0:
        insights.append(
            f"You spent <strong>${total_spend:,.2f}</strong> across "
            f"<strong>{kpis['debit_count']}</strong> transactions. "
            f"No income deposits were found in this statement."
        )

    # --- Top spending category with % ---
    cat_spend = debit_df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    if len(cat_spend) > 0:
        top_cat = cat_spend.index[0]
        top_pct = cat_spend.iloc[0] / total_spend * 100
        insights.append(
            f"<strong>{top_cat}</strong> was your largest spending category at "
            f"<strong>${cat_spend.iloc[0]:,.2f}</strong> ({top_pct:.0f}% of all expenses)."
        )

    # --- Top merchant: "You spent $X at Y across N transactions." ---
    merch_spend = debit_df.groupby("Merchant")["Amount"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    if len(merch_spend) > 0:
        top_m = merch_spend.index[0]
        top_m_total = merch_spend.iloc[0]["sum"]
        top_m_count = int(merch_spend.iloc[0]["count"])
        insights.append(
            f"You spent <strong>${top_m_total:,.2f}</strong> at "
            f"<strong>{top_m}</strong> across {top_m_count} transaction(s)."
        )

    # --- Second merchant share: "X accounted for Y% of total spending." ---
    if len(merch_spend) >= 2:
        m2 = merch_spend.index[1]
        m2_pct = merch_spend.iloc[1]["sum"] / total_spend * 100
        if m2_pct >= 5:
            insights.append(
                f"<strong>{m2}</strong> accounted for <strong>{m2_pct:.0f}%</strong> "
                f"of total spending (${merch_spend.iloc[1]['sum']:,.2f})."
            )

    # --- Non-top categories > 10% share (max 3): "X represented Y% of all expenses." ---
    cat_share_count = 0
    for cat, val in cat_spend.items():
        pct = val / total_spend * 100
        if cat != cat_spend.index[0] and pct >= 10 and cat_share_count < 3:
            insights.append(
                f"<strong>{cat}</strong> represented <strong>{pct:.0f}%</strong> "
                f"of all expenses (${val:,.2f})."
            )
            cat_share_count += 1

    # --- Highest-share category explicit %: "X% of spending occurred at Y." ---
    if len(cat_spend) > 0:
        top_pct = cat_spend.iloc[0] / total_spend * 100
        if top_pct >= 25:
            insights.append(
                f"<strong>{top_pct:.0f}%</strong> of spending occurred in "
                f"<strong>{cat_spend.index[0]}</strong>."
            )

    # --- Dining + food total ---
    food_cats = {"Dining", "Groceries", "Gas & Convenience"}
    food_spend = debit_df[debit_df["Category"].isin(food_cats)]["Amount"].sum()
    if food_spend > 0:
        insights.append(
            f"You spent <strong>${food_spend:,.2f}</strong> on food and dining "
            f"(Dining, Groceries, Convenience combined)."
        )

    # --- Convenience vs dining comparison ---
    conv = cat_spend.get("Gas & Convenience", 0)
    dining = cat_spend.get("Dining", 0)
    if conv > 0 and dining > 0:
        if conv > dining:
            insights.append(
                f"You spent more at convenience stores (<strong>${conv:,.2f}</strong>) "
                f"than on dining (<strong>${dining:,.2f}</strong>)."
            )
        elif dining > conv:
            insights.append(
                f"Dining spending (<strong>${dining:,.2f}</strong>) exceeded "
                f"convenience-store spending (<strong>${conv:,.2f}</strong>)."
            )

    # --- Most-visited merchant ---
    if kpis["most_visited"] and kpis["most_visited_count"] >= 3:
        insights.append(
            f"Your most-visited merchant was <strong>{kpis['most_visited']}</strong> "
            f"with <strong>{kpis['most_visited_count']}</strong> visits."
        )

    # --- Small purchases ---
    if kpis["small_count"] >= 3:
        insights.append(
            f"You made <strong>{kpis['small_count']}</strong> transactions under $5. "
            f"Small purchases totaled <strong>${kpis['small_total']:,.2f}</strong>."
        )

    # --- Subscription spending (top 5 merchants) ---
    sub_df = debit_df[debit_df["Category"] == "Subscriptions"]
    if not sub_df.empty:
        sub_total = sub_df["Amount"].sum()
        sub_merchants = sub_df.groupby("Merchant")["Amount"].sum().sort_values(ascending=False)
        for m, amt in sub_merchants.head(5).items():
            insights.append(
                f"<strong>{m}</strong> subscriptions accounted for <strong>${amt:,.2f}</strong>."
            )
        if sub_total > 0:
            insights.append(
                f"Total subscription spending: <strong>${sub_total:,.2f}</strong>."
            )

    # --- Largest single transaction ---
    if kpis["largest_amount"] > 0:
        insights.append(
            f"Your largest expense was <strong>{kpis['largest_merchant']}</strong> "
            f"(<strong>${kpis['largest_amount']:,.2f}</strong>)."
        )

    return insights[:18]  # Cap for clean UI; most important insights are generated first


def generate_savings_tips(df: pd.DataFrame, kpis: Dict[str, Any]) -> List[str]:
    """Actionable savings recommendations from actual spending patterns."""
    tips: List[str] = []
    debit_df = df[df["Type"] == "Debit"]
    if debit_df.empty:
        return tips
    total = kpis["total_spending"]
    cat_spend = debit_df.groupby("Category")["Amount"].sum()

    # Convenience store reduction
    conv = cat_spend.get("Gas & Convenience", 0)
    if conv > 0:
        save25 = conv * 0.25
        pct = conv / total * 100
        tips.append(
            f"Reducing convenience-store spending by 25% would save approximately "
            f"<strong>${save25:,.2f}</strong>. Convenience stores are "
            f"<strong>{pct:.0f}%</strong> of your total spend."
        )

    # Dining vs groceries
    dining = cat_spend.get("Dining", 0)
    grocery = cat_spend.get("Groceries", 0)
    if dining > 0 and grocery >= 0 and dining > grocery:
        tips.append(
            f"Dining spending (<strong>${dining:,.2f}</strong>) exceeds grocery spending "
            f"(<strong>${grocery:,.2f}</strong>). Cooking more could save significantly."
        )

    # Subscription audit
    sub_df = debit_df[debit_df["Category"] == "Subscriptions"]
    if not sub_df.empty:
        sub_total = sub_df["Amount"].sum()
        tips.append(
            f"Subscription spending totals <strong>${sub_total:,.2f}</strong>. "
            f"Review active subscriptions for unused services."
        )

    # Small-purchase awareness
    if kpis["small_count"] >= 5:
        tips.append(
            f"You made {kpis['small_count']} transactions under $5 totaling "
            f"<strong>${kpis['small_total']:,.2f}</strong>. "
            f"Batching small purchases could reduce impulse spending."
        )

    # Top category reduction (skip Income/Rent/Utilities)
    cat_sorted = cat_spend.sort_values(ascending=False)
    if len(cat_sorted) > 0:
        top = cat_sorted.index[0]
        top_val = cat_sorted.iloc[0]
        pct = top_val / total * 100
        if pct >= 20 and top not in ("Income", "Rent", "Utilities"):
            save10 = top_val * 0.10
            tips.append(
                f"<strong>{top}</strong> is {pct:.0f}% of your spending. "
                f"A 10% reduction would save <strong>${save10:,.2f}</strong>."
            )

    # Cash flow warning
    if kpis["net_cash_flow"] < 0:
        tips.append(
            f"You spent <strong>${abs(kpis['net_cash_flow']):,.2f}</strong> more than you earned. "
            f"Look for non-essential categories to trim."
        )
    elif kpis["net_cash_flow"] > 0 and kpis["total_income"] > 0:
        tips.append(
            f"You have a <strong>${kpis['net_cash_flow']:,.2f}</strong> surplus. "
            f"Consider moving it to a savings or investment account."
        )

    return tips
