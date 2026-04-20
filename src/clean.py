"""
clean.py — Data loading and cleaning utilities for ProfitLens
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """Load superstore CSV with encoding fallback."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    print(f"✅ Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    - Fix column names
    - Parse dates
    - Drop duplicates
    - Handle nulls
    - Add engineered features
    """
    df = df.copy()

    # Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')

    # Parse date columns
    for col in ['order_date', 'ship_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], dayfirst=False, errors='coerce')

    # Drop exact duplicate rows
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"🗑  Dropped {before - len(df)} duplicate rows")

    # Report & drop rows where critical columns are null
    critical = ['sales', 'profit', 'discount', 'order_date']
    for col in critical:
        if col in df.columns:
            nulls = df[col].isna().sum()
            if nulls:
                print(f"⚠️  {nulls} nulls in '{col}' — dropping those rows")
            df = df[df[col].notna()]

    # --- Feature Engineering ---

    # Profit margin %
    df['profit_margin_pct'] = (df['profit'] / df['sales'].replace(0, np.nan)) * 100

    # Discount bucket
    df['discount_bucket'] = pd.cut(
        df['discount'],
        bins=[-0.01, 0.0, 0.10, 0.20, 0.30, 0.50, 1.01],
        labels=['0%', '1–10%', '11–20%', '21–30%', '31–50%', '51%+']
    )

    # Time features
    df['year']          = df['order_date'].dt.year
    df['month']         = df['order_date'].dt.month
    df['month_name']    = df['order_date'].dt.strftime('%b')
    df['quarter']       = df['order_date'].dt.quarter
    df['year_month']    = df['order_date'].dt.to_period('M')

    # Profit flag
    df['is_profitable'] = df['profit'] > 0

    print(f"✅ Clean dataset: {len(df):,} rows | date range: "
          f"{df['order_date'].min().date()} → {df['order_date'].max().date()}")
    return df


def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Print high-level KPIs."""
    stats = {
        'Total Orders':       df['order_id'].nunique() if 'order_id' in df.columns else len(df),
        'Total Sales ($)':    f"${df['sales'].sum():,.0f}",
        'Total Profit ($)':   f"${df['profit'].sum():,.0f}",
        'Avg Profit Margin':  f"{df['profit_margin_pct'].mean():.1f}%",
        'Loss-making Orders': f"{(~df['is_profitable']).sum():,} ({(~df['is_profitable']).mean()*100:.1f}%)",
        'Avg Discount':       f"{df['discount'].mean()*100:.1f}%",
    }
    print("\n📊 Dataset KPIs")
    print("-" * 35)
    for k, v in stats.items():
        print(f"  {k:<25} {v}")
    return pd.DataFrame(stats, index=[0])
