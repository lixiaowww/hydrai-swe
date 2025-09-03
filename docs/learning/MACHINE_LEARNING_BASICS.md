# 🤖 Machine Learning Basics for HydrAI-SWE

This guide gives you a practical introduction to machine learning focused on hydrology and Snow Water Equivalent (SWE) use cases in this project. Read alongside the Python Basics guide before diving into model code.

## What is Machine Learning?

- **Goal**: Learn patterns from real data to make predictions or decisions.
- **Key tasks**: prediction (regression), classification, clustering, anomaly detection, and time-series forecasting.

## Problem Types

- **Regression**: Predict a continuous value (e.g., future SWE in mm).
- **Classification**: Predict a category (e.g., flood risk level: LOW/MEDIUM/HIGH).
- **Clustering**: Group similar items without labels (e.g., station similarity by seasonal profiles).
- **Anomaly Detection**: Identify unusual behavior (e.g., sensor spikes or flow anomalies).
- **Time Series Forecasting**: Predict future values from timestamped data (SWE, temperature, streamflow).

## Data Workflow (End-to-End)

1) Ingest real data → 2) Clean & validate → 3) Feature engineering → 4) Split train/validation/test → 5) Train → 6) Evaluate → 7) Serve via API → 8) Monitor & improve.

Important principles for this project:
- Use only authentic data. If data is unavailable, return N/A rather than simulate.
- Keep file paths centralized via constants (e.g., `DATA_PATHS`).
- Be explicit about data provenance and time ranges.

## Core Concepts

### Train/Validation/Test Split

- Avoid leakage by splitting by time for time series: train on the past, validate on recent past, test on the most recent segment.
- Typical ratios: 70/15/15 or 80/10/10 depending on dataset size.

### Scaling and Normalization

- Many models benefit from standardized inputs (zero mean, unit variance). Use `StandardScaler` fit only on the training set; apply the transform to validation/test.

### Cross-Validation (CV)

- For time series, prefer time-aware CV (e.g., expanding window). Avoid shuffling timestamps.

### Metrics

- Regression: MAE, RMSE, MAPE, R².
- Classification: Accuracy, Precision/Recall/F1, ROC-AUC.
- Forecasting diagnostics: residual plots, seasonal error analysis.

## Working With Time Series

Common engineered features:
- Calendar: year, month, day-of-year, week-of-year.
- Seasonality encodings: `sin/cos(2π * month/12)` to capture annual cycles.
- Lags: values from previous k steps (e.g., SWE_lag_1, SWE_lag_7, SWE_lag_30).
- Rolling statistics: moving averages and standard deviations (e.g., 7-day, 30-day).

Basic example (Pandas):

```python
df = df.sort_values('date').reset_index(drop=True)
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['dayofyear'] = df['date'].dt.dayofyear
import numpy as np
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
for lag in [1, 3, 7, 14, 30]:
    df[f'swe_lag_{lag}'] = df['snow_water_equivalent_mm'].shift(lag)
df['swe_ma_7'] = df['snow_water_equivalent_mm'].rolling(7).mean()
```

## Quickstart: Regression with scikit-learn

```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load real dataset (ensure the path points to actual data)
df = pd.read_csv('/home/sean/hydrai_swe/data/processed/eccc_manitoba_snow_processed.csv', parse_dates=['date'])
df = df.sort_values('date')

# Select features/target (adjust column names to real ones)
target_col = 'snow_water_equivalent_mm'
feature_cols = [
    'sin_month', 'cos_month', 'swe_lag_1', 'swe_lag_7', 'swe_ma_7'
]

df = df.dropna(subset=[target_col] + feature_cols)
X = df[feature_cols]
y = df[target_col]

pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=True, with_std=True)),
    ('model', RandomForestRegressor(n_estimators=300))
])

tscv = TimeSeriesSplit(n_splits=5)
maes, r2s = [], []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    maes.append(mean_absolute_error(y_test, y_pred))
    r2s.append(r2_score(y_test, y_pred))

print({'MAE': sum(maes)/len(maes), 'R2': sum(r2s)/len(r2s)})
```

Notes:
- Do not shuffle time series.
- Validate column availability; use a mapping function to reconcile column name differences across datasets (see `_find_matching_column` pattern in the project).

## Anomaly Detection (IsolationForest)

Useful for detecting sensor glitches or unexpected hydrological behavior.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv('/home/sean/hydrai_swe/data/processed/comprehensive_training_dataset.csv', parse_dates=['date'])
df = df.sort_values('date')

features = ['snow_water_equivalent_mm', 'temperature_c', 'precip_mm']
df = df.dropna(subset=features)
X = df[features]

iso = IsolationForest(n_estimators=300, contamination=0.02)
labels = iso.fit_predict(X)  # -1 = anomaly, 1 = normal
df['anomaly'] = (labels == -1)

anomalies = df[df['anomaly']]
print(f"Anomalies: {len(anomalies)} rows")
```

Tips:
- Keep `contamination` realistic based on domain knowledge.
- Inspect anomalies manually; do not auto-delete without review.

## Deep Learning for Time Series (Keras)

GRU/LSTM can capture seasonality and long-term dependencies in SWE and streamflow.

```python
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import StandardScaler

def make_sequences(values: np.ndarray, seq_len: int = 30):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X), np.array(y)

df = pd.read_csv('/home/sean/hydrai_swe/data/processed/eccc_manitoba_snow_processed.csv', parse_dates=['date'])
df = df.sort_values('date')
series = df['snow_water_equivalent_mm'].dropna().values.reshape(-1, 1)

scaler = StandardScaler()
series_scaled = scaler.fit_transform(series)

X, y = make_sequences(series_scaled, seq_len=30)

model = Sequential([
    GRU(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    GRU(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X[:-200], y[:-200], validation_data=(X[-200:], y[-200:]), epochs=20, batch_size=32)

# Predict next step
pred_scaled = model.predict(X[-1:])
pred = scaler.inverse_transform(pred_scaled)[0, 0]
print({'next_swe_prediction': float(pred)})
```

Guidelines:
- Always inverse-transform predictions to the original units.
- Track validation loss to avoid overfitting; consider early stopping.

## Hydrology-Specific Best Practices

- Respect physical plausibility (e.g., SWE cannot be negative).
- Compare against historical percentiles (e.g., 10th/50th/90th) to interpret context.
- Align with regional calendars (freeze-up, melt, peak flow).
- Use deterministic transformations when necessary; never fabricate data.

## Common Pitfalls (and Fixes)

- Data leakage: do not let future information into training features.
- Misaligned timestamps: always sort and align before joining datasets.
- Inconsistent units: normalize units (e.g., mm vs cm) and document conversions.
- Over-smoothing: rolling windows can hide extremes; evaluate both raw and smoothed series.

## How This Maps to the Codebase

- Column reconciliation: see `_find_matching_column` in `data_science` components.
- Centralized paths: use `DATA_PATHS` where applicable.
- No random or mock generators: return N/A when real data is missing.
- APIs expose analysis endpoints (SWE, statistics, anomalies) consumed by the frontend.

## Next Steps

- Read the SWE Analysis System docs and trace an end-to-end request from the UI to the API to the model.
- Reproduce one analysis locally using the real datasets in `/home/sean/hydrai_swe/data/processed`.
- Extend features with an additional lag or seasonal indicator and measure impact.

## References

- scikit-learn User Guide (`https://scikit-learn.org/stable/user_guide.html`)
- TensorFlow Keras API (`https://www.tensorflow.org/api_docs/python/tf/keras`)
- Time Series Cross-Validation (`https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split`)


