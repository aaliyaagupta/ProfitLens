# 🛒 ProfitLens — Retail Analytics & Forecasting

A Python project analyzing 10,000+ e-commerce transactions to uncover
regional profitability trends, discount impact on margins, and 30-day profit forecasting.

---

## 📁 Folder Structure

```
ProfitLens/
├── data/                        ← Put your downloaded CSV here
│   └── Sample - Superstore.csv
├── notebooks/
│   └── profitlens.ipynb         ← Main notebook (run this)
├── outputs/                     ← Charts are auto-saved here
└── README.md
```

---

## 🚀 How to Run

### 1. Download the Dataset
Go to: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final
Download `Sample - Superstore.csv` and place it in the `data/` folder.

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn prophet scikit-learn jupyter
```

### 3. Open the Notebook
```bash
jupyter notebook notebooks/profitlens.ipynb
```

### 4. Run All Cells
In Jupyter: Kernel → Restart & Run All

---

## 📊 What the Notebook Does

| Step | What it does | Key output |
|------|-------------|------------|
| Step 1 | Load data | See all columns and first few rows |
| Step 2 | Clean data | Fix dates, remove duplicates, check nulls |
| Step 3 | Regional analysis | Which region makes most profit? |
| Step 4 | Discount analysis | How badly do discounts hurt margins? |
| Step 5 | 30-day forecast | Predict future profit using Prophet |

---

## 🔮 About Prophet (the forecasting model)

Prophet is a library built by Meta (Facebook).
- You feed it: dates + daily profit numbers
- It learns: weekly patterns, yearly patterns, overall trend
- It predicts: the next 30 days of profit

Think of it as a pattern recognizer for time-based data.
No deep math required — just model.fit() and model.predict().

---

## 📦 Libraries Used

| Library | Purpose |
|---------|---------|
| pandas | Load and clean data |
| numpy | Math operations |
| matplotlib | Plotting |
| seaborn | Nicer plots |
| prophet | 30-day forecasting |
| scikit-learn | Forecast accuracy (MAE, RMSE) |
