"""
Financial Insights Module
Generates 4 key insights from transaction data:
1. Monthly cashflow (income vs spend)
2. Category breakdown
3. Top merchants
4. Recurring transactions detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


def monthly_cashflow(df: pd.DataFrame) -> Dict:
    """
    Calculate monthly income vs spending.
    Returns: {month: {income: float, spend: float, net: float}}
    """
    df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
    
    monthly = {}
    for month, group in df.groupby('month'):
        income = group[group['amount_signed'] > 0]['amount_signed'].sum()
        spend = abs(group[group['amount_signed'] < 0]['amount_signed'].sum())
        net = income - spend
        monthly[str(month)] = {
            'income': round(income, 2),
            'spend': round(spend, 2),
            'net': round(net, 2),
        }
    
    return monthly


def category_breakdown(df: pd.DataFrame) -> Dict:
    """
    Calculate spending by category.
    Returns: {category: {total: float, percentage: float, count: int}}
    """
    # Only count spending (negative amounts)
    spending_df = df[df['amount_signed'] < 0].copy()
    spending_df['amount_abs'] = spending_df['amount_signed'].abs()
    
    if len(spending_df) == 0:
        return {}
    
    total_spend = spending_df['amount_abs'].sum()
    
    breakdown = {}
    for category, group in spending_df.groupby('category'):
        cat_total = group['amount_abs'].sum()
        percentage = (cat_total / total_spend * 100) if total_spend > 0 else 0
        breakdown[category] = {
            'total': round(cat_total, 2),
            'percentage': round(percentage, 1),
            'count': len(group),
        }
    
    # Sort by total descending
    breakdown = dict(sorted(breakdown.items(), key=lambda x: x[1]['total'], reverse=True))
    return breakdown


def top_merchants(df: pd.DataFrame, n: int = 10) -> List[Dict]:
    """
    Get top N merchants by total spending.
    Returns: [{merchant: str, total: float, count: int, avg: float}, ...]
    """
    spending_df = df[df['amount_signed'] < 0].copy()
    spending_df['amount_abs'] = spending_df['amount_signed'].abs()
    
    if len(spending_df) == 0:
        return []
    
    merchant_stats = []
    for merchant, group in spending_df.groupby('description'):
        total = group['amount_abs'].sum()
        count = len(group)
        avg = total / count if count > 0 else 0
        merchant_stats.append({
            'merchant': merchant,
            'total': round(total, 2),
            'count': count,
            'avg': round(avg, 2),
        })
    
    # Sort by total descending
    merchant_stats.sort(key=lambda x: x['total'], reverse=True)
    return merchant_stats[:n]


def recurring_transactions(df: pd.DataFrame, min_occurrences: int = 2) -> List[Dict]:
    """
    Detect recurring transactions (same description + similar amount across months).
    Returns: [{description: str, frequency: str, amount: float, months: int, confidence: float}, ...]
    """
    if len(df) == 0:
        return []
    
    df_copy = df.copy()
    df_copy['month'] = pd.to_datetime(df_copy['date']).dt.to_period('M')
    df_copy['description_norm'] = df_copy['description'].str.upper().str.strip()
    
    recurring = []
    
    for desc, group in df_copy.groupby('description_norm'):
        # Get unique months
        months = group['month'].unique()
        if len(months) < min_occurrences:
            continue
        
        # Get all amounts for this description
        amounts = group['amount_signed'].abs().values
        
        # Check if amounts are similar (within 20% standard deviation)
        amount_mean = np.mean(amounts)
        amount_std = np.std(amounts)
        
        # If std is very small relative to mean, it's recurring
        if amount_mean > 0:
            cv = amount_std / amount_mean  # Coefficient of variation
            if cv < 0.2:  # Low variance = recurring
                confidence = 1.0 - min(cv, 0.2) / 0.2  # 0.0 to 1.0
                recurring.append({
                    'description': desc,
                    'frequency': f"~{len(months)}x/month" if len(months) > 1 else "Monthly",
                    'amount': round(amount_mean, 2),
                    'months': len(months),
                    'confidence': round(confidence, 2),
                })
    
    # Sort by confidence descending
    recurring.sort(key=lambda x: x['confidence'], reverse=True)
    return recurring


def spending_summary(df: pd.DataFrame) -> Dict:
    """
    Quick summary stats.
    """
    if len(df) == 0:
        return {
            'total_transactions': 0,
            'total_income': 0.0,
            'total_spend': 0.0,
            'net_cashflow': 0.0,
            'avg_transaction': 0.0,
            'date_range': '',
        }
    
    income = df[df['amount_signed'] > 0]['amount_signed'].sum()
    spend = abs(df[df['amount_signed'] < 0]['amount_signed'].sum())
    
    date_min = df['date'].min()
    date_max = df['date'].max()
    days = (date_max - date_min).days + 1
    
    return {
        'total_transactions': len(df),
        'total_income': round(income, 2),
        'total_spend': round(spend, 2),
        'net_cashflow': round(income - spend, 2),
        'avg_transaction': round((income + spend) / len(df), 2) if len(df) > 0 else 0.0,
        'date_range': f"{date_min.date()} to {date_max.date()} ({days} days)",
    }


def generate_all_insights(df: pd.DataFrame) -> Dict:
    """
    Generate all insights at once.
    """
    return {
        'summary': spending_summary(df),
        'monthly_cashflow': monthly_cashflow(df),
        'category_breakdown': category_breakdown(df),
        'top_merchants': top_merchants(df, n=10),
        'recurring_transactions': recurring_transactions(df),
    }
