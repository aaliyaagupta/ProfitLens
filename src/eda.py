"""
eda.py — Exploratory Data Analysis functions for ProfitLens
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os

# ── Styling ──────────────────────────────────────────────────────────────────
PALETTE   = "Set2"
FIG_SIZE  = (12, 5)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  💾 Saved → outputs/{name}")


# ── 1. Regional Profitability ────────────────────────────────────────────────
def plot_regional_profitability(df: pd.DataFrame):
    """Bar chart — total profit & avg margin by region."""
    reg = (df.groupby('region')
             .agg(total_profit=('profit', 'sum'),
                  avg_margin=('profit_margin_pct', 'mean'))
             .sort_values('total_profit', ascending=False)
             .reset_index())

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
    fig.suptitle('Regional Profitability', fontsize=14, fontweight='bold')

    # Total profit
    sns.barplot(data=reg, x='region', y='total_profit', palette=PALETTE, ax=axes[0])
    axes[0].set_title('Total Profit by Region')
    axes[0].set_xlabel('')
    axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    for bar in axes[0].patches:
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 200,
                     f'${bar.get_height():,.0f}', ha='center', fontsize=9)

    # Avg margin
    sns.barplot(data=reg, x='region', y='avg_margin', palette=PALETTE, ax=axes[1])
    axes[1].set_title('Avg Profit Margin % by Region')
    axes[1].set_xlabel('')
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    _save(fig, '01_regional_profitability.png')
    plt.show()

    # Print insight
    best  = reg.iloc[0]
    worst = reg.iloc[-1]
    print(f"\n🔍 Insight: '{best['region']}' is the most profitable region "
          f"(${best['total_profit']:,.0f}), "
          f"{best['total_profit']/worst['total_profit']:.1f}x more than "
          f"'{worst['region']}' (${worst['total_profit']:,.0f})")
    return reg


# ── 2. Discount Impact on Profit ─────────────────────────────────────────────
def plot_discount_impact(df: pd.DataFrame):
    """Scatter + bucket bar showing how discount erodes profit."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
    fig.suptitle('Discount Impact on Profit Margins', fontsize=14, fontweight='bold')

    # Scatter
    sample = df.sample(min(3000, len(df)), random_state=42)
    axes[0].scatter(sample['discount'] * 100, sample['profit'],
                    alpha=0.3, s=15, color='steelblue')
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Discount %')
    axes[0].set_ylabel('Profit ($)')
    axes[0].set_title('Discount vs Profit (per order)')

    # Bucket bar
    bucket = (df.groupby('discount_bucket', observed=True)
                .agg(avg_profit=('profit', 'mean'),
                     order_count=('profit', 'count'))
                .reset_index())
    colors = ['#2ecc71' if x >= 0 else '#e74c3c' for x in bucket['avg_profit']]
    axes[1].bar(bucket['discount_bucket'], bucket['avg_profit'], color=colors)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Avg Profit by Discount Bucket')
    axes[1].set_xlabel('Discount Range')
    axes[1].set_ylabel('Avg Profit ($)')
    for i, row in bucket.iterrows():
        axes[1].text(i, row['avg_profit'] + (2 if row['avg_profit'] >= 0 else -8),
                     f"n={row['order_count']:,}", ha='center', fontsize=8)

    plt.tight_layout()
    _save(fig, '02_discount_impact.png')
    plt.show()

    # Find breakeven discount
    loss_buckets = bucket[bucket['avg_profit'] < 0]['discount_bucket'].tolist()
    high_disc_orders = (df['discount'] > 0.20).mean() * 100
    print(f"\n🔍 Insight: Average profit turns negative at discount buckets: {loss_buckets}")
    print(f"   {high_disc_orders:.1f}% of orders carry discounts above 20%")
    return bucket


# ── 3. Category & Sub-category Performance ───────────────────────────────────
def plot_category_performance(df: pd.DataFrame):
    """Horizontal bar — profit by category and sub-category."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Category & Sub-Category Profitability', fontsize=14, fontweight='bold')

    # Category
    cat = (df.groupby('category')['profit']
             .sum()
             .sort_values()
             .reset_index())
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in cat['profit']]
    axes[0].barh(cat['category'], cat['profit'], color=colors)
    axes[0].axvline(0, color='black', linewidth=0.8)
    axes[0].set_title('Total Profit by Category')
    axes[0].set_xlabel('Total Profit ($)')

    # Sub-category
    subcat = (df.groupby('sub_category')['profit']
                .sum()
                .sort_values()
                .reset_index())
    colors2 = ['#e74c3c' if x < 0 else '#2ecc71' for x in subcat['profit']]
    axes[1].barh(subcat['sub_category'], subcat['profit'], color=colors2)
    axes[1].axvline(0, color='black', linewidth=0.8)
    axes[1].set_title('Total Profit by Sub-Category')
    axes[1].set_xlabel('Total Profit ($)')

    plt.tight_layout()
    _save(fig, '03_category_performance.png')
    plt.show()

    loss_subcats = subcat[subcat['profit'] < 0]['sub_category'].tolist()
    top_subcat   = subcat.iloc[-1]
    print(f"\n🔍 Insight: Loss-making sub-categories: {loss_subcats}")
    print(f"   Top performer: '{top_subcat['sub_category']}' "
          f"(${top_subcat['profit']:,.0f} profit)")
    return subcat


# ── 4. Customer Segment Analysis ─────────────────────────────────────────────
def plot_segment_analysis(df: pd.DataFrame):
    """Compare profit and margin across customer segments."""
    seg = (df.groupby('segment')
             .agg(total_profit=('profit', 'sum'),
                  avg_profit_per_order=('profit', 'mean'),
                  avg_margin=('profit_margin_pct', 'mean'),
                  order_count=('profit', 'count'))
             .sort_values('total_profit', ascending=False)
             .reset_index())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Customer Segment Analysis', fontsize=14, fontweight='bold')

    metrics = [
        ('total_profit',        'Total Profit ($)',        axes[0]),
        ('avg_profit_per_order','Avg Profit per Order ($)', axes[1]),
        ('avg_margin',          'Avg Profit Margin (%)',    axes[2]),
    ]
    for col, title, ax in metrics:
        sns.barplot(data=seg, x='segment', y=col, palette=PALETTE, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('')

    plt.tight_layout()
    _save(fig, '04_segment_analysis.png')
    plt.show()

    top = seg.iloc[0]
    bot = seg.iloc[-1]
    diff = ((top['avg_profit_per_order'] - bot['avg_profit_per_order'])
            / abs(bot['avg_profit_per_order'])) * 100
    print(f"\n🔍 Insight: '{top['segment']}' generates "
          f"${top['avg_profit_per_order']:.1f} avg profit/order, "
          f"{diff:.0f}% more than '{bot['segment']}'")
    return seg


# ── 5. Monthly Profit Trend ───────────────────────────────────────────────────
def plot_monthly_trend(df: pd.DataFrame):
    """Line chart of monthly profit over time."""
    monthly = (df.groupby('year_month')['profit']
                 .sum()
                 .reset_index())
    monthly['year_month_dt'] = monthly['year_month'].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(monthly['year_month_dt'], monthly['profit'],
            marker='o', markersize=4, linewidth=1.5, color='steelblue')
    ax.axhline(monthly['profit'].mean(), color='orange',
               linestyle='--', linewidth=1, label='Mean profit')
    ax.fill_between(monthly['year_month_dt'], monthly['profit'],
                    alpha=0.15, color='steelblue')
    ax.set_title('Monthly Profit Trend', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Profit ($)')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save(fig, '05_monthly_trend.png')
    plt.show()
    return monthly
