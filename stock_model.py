"""Shared data-fetching, feature engineering, and prediction logic
for both the CLI script (stock_predictor.py) and the live dashboard (app.py).
"""
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURE_COLUMNS = [
    "lag_1", "lag_2", "lag_3", "lag_5", "lag_10",
    "ma_5", "ma_10", "ma_20", "rsi_14",
]


def fetch_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'")
    df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for lag in (1, 2, 3, 5, 10):
        data[f"lag_{lag}"] = data["Close"].shift(lag)
    data["ma_5"] = data["Close"].rolling(5).mean()
    data["ma_10"] = data["Close"].rolling(10).mean()
    data["ma_20"] = data["Close"].rolling(20).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    data["rsi_14"] = 100 - (100 / (1 + rs))

    data["target"] = data["Close"].shift(-1)
    return data.dropna()


def train_and_predict(data: pd.DataFrame):
    """Train on an 80/20 time-ordered split and predict the next close.

    Returns a dict with test dates/actual/predicted, RMSE/MAE, last close,
    and the predicted next close.
    """
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split], data.iloc[split:]

    X_train, y_train = train[FEATURE_COLUMNS], train["target"]
    X_test, y_test = test[FEATURE_COLUMNS], test["target"]

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)

    latest_features = data[FEATURE_COLUMNS].iloc[[-1]]
    next_close_pred = model.predict(latest_features)[0]
    last_close = data["Close"].iloc[-1]

    return {
        "test_dates": test.index,
        "actual": y_test,
        "predicted": preds,
        "rmse": rmse,
        "mae": mae,
        "last_close": last_close,
        "next_close_pred": next_close_pred,
    }


def predict_ticker(ticker: str, period: str = "2y", interval: str = "1d") -> dict:
    """Convenience wrapper: fetch, build features, train, and predict for one ticker."""
    df = fetch_history(ticker, period=period, interval=interval)
    data = build_features(df)
    result = train_and_predict(data)
    result["history"] = data
    return result
