"""
forecast.py — 30-day profit forecasting using Facebook Prophet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  💾 Saved → outputs/{name}")


def prepare_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily profit and return Prophet-ready df with columns [ds, y].
    Also fills missing dates with 0 profit (weekends/holidays).
    """
    daily = (df.groupby('order_date')['profit']
               .sum()
               .reset_index()
               .rename(columns={'order_date': 'ds', 'profit': 'y'}))

    # Fill gaps in date range
    full_range = pd.date_range(daily['ds'].min(), daily['ds'].max(), freq='D')
    daily = (daily.set_index('ds')
                  .reindex(full_range, fill_value=0)
                  .reset_index()
                  .rename(columns={'index': 'ds'}))

    print(f"✅ Time series: {len(daily)} daily observations | "
          f"{daily['ds'].min().date()} → {daily['ds'].max().date()}")
    return daily


def train_prophet(daily_df: pd.DataFrame, forecast_days: int = 30):
    """
    Fit Prophet model, forecast next `forecast_days` days.
    Returns (model, forecast_df).
    """
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError("Run: pip install prophet")

    # Hold-out last 30 days for evaluation
    cutoff     = daily_df['ds'].max() - pd.Timedelta(days=forecast_days)
    train_df   = daily_df[daily_df['ds'] <= cutoff]
    test_df    = daily_df[daily_df['ds'] > cutoff]

    print(f"📅 Training on {len(train_df)} days | Evaluating on {len(test_df)} days")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode='multiplicative'
    )
    model.fit(train_df)

    # Forecast over test period + 30 future days
    future   = model.make_future_dataframe(periods=forecast_days + len(test_df))
    forecast = model.predict(future)

    # Evaluation on hold-out
    merged = test_df.merge(
        forecast[['ds', 'yhat']].rename(columns={'yhat': 'predicted'}),
        on='ds', how='inner'
    )
    mae  = mean_absolute_error(merged['y'], merged['predicted'])
    rmse = np.sqrt(mean_squared_error(merged['y'], merged['predicted']))
    mape = (np.abs((merged['y'] - merged['predicted'])
                   / merged['y'].replace(0, np.nan)).mean()) * 100

    print(f"\n📈 Forecast Evaluation (last {forecast_days} days hold-out):")
    print(f"   MAE  : ${mae:,.2f}")
    print(f"   RMSE : ${rmse:,.2f}")
    print(f"   MAPE : {mape:.1f}%")

    return model, forecast, {'mae': mae, 'rmse': rmse, 'mape': mape}


def plot_forecast(daily_df: pd.DataFrame, forecast: pd.DataFrame,
                  forecast_days: int = 30):
    """Plot actuals vs predicted with confidence interval."""
    cutoff = daily_df['ds'].max() - pd.Timedelta(days=forecast_days)

    fig, ax = plt.subplots(figsize=(14, 5))

    # Actuals
    ax.plot(daily_df['ds'], daily_df['y'],
            color='steelblue', linewidth=1, alpha=0.8, label='Actual Profit')

    # Forecast line
    ax.plot(forecast['ds'], forecast['yhat'],
            color='darkorange', linewidth=1.5, linestyle='--', label='Forecast')

    # Confidence interval
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    alpha=0.2, color='darkorange', label='95% Confidence Interval')

    # Mark forecast start
    ax.axvline(cutoff, color='red', linestyle=':', linewidth=1.2,
               label='Forecast Start')

    ax.set_title('30-Day Profit Forecast (Prophet)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Profit ($)')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    _save(fig, '06_profit_forecast.png')
    plt.show()


def plot_components(model, forecast):
    """Prophet trend + seasonality decomposition."""
    fig = model.plot_components(forecast)
    fig.suptitle('Forecast Components — Trend & Seasonality',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save(fig, '07_forecast_components.png')
    plt.show()


def save_forecast_csv(forecast: pd.DataFrame, daily_df: pd.DataFrame,
                      forecast_days: int = 30):
    """Save the next 30 future days forecast to CSV."""
    future_start = daily_df['ds'].max() + pd.Timedelta(days=1)
    future_fc = forecast[forecast['ds'] >= future_start][
        ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    ].head(forecast_days).copy()
    future_fc.columns = ['date', 'predicted_profit', 'lower_bound', 'upper_bound']
    future_fc['date'] = future_fc['date'].dt.date

    path = os.path.join(OUTPUT_DIR, 'forecast_30days.csv')
    future_fc.to_csv(path, index=False)
    print(f"\n💾 30-day forecast saved → outputs/forecast_30days.csv")
    print(future_fc.to_string(index=False))
    return future_fc
