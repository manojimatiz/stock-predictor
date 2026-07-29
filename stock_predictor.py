"""Predict next-day stock closing price from historical data.

Usage:
    python stock_predictor.py TICKER [--period 5y] [--out chart.png]
"""
import argparse

import matplotlib.pyplot as plt

from stock_model import build_features, fetch_history, train_and_predict


def plot_results(ticker: str, dates, actual, predicted, out_path: str | None):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, predicted, label="Predicted")
    plt.title(f"{ticker} — Actual vs Predicted Close")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Chart saved to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL or TATASTEEL.NS")
    parser.add_argument("--period", default="5y", help="History period to fetch (default: 5y)")
    parser.add_argument("--out", default=None, help="Save chart to this file instead of opening a window")
    args = parser.parse_args()

    df = fetch_history(args.ticker, args.period)
    data = build_features(df)
    result = train_and_predict(data)

    print(f"Test RMSE: {result['rmse']:.2f}  MAE: {result['mae']:.2f}")
    print(f"Last close: {result['last_close']:.2f}  ->  Predicted next close: {result['next_close_pred']:.2f}")

    plot_results(args.ticker, result["test_dates"], result["actual"], result["predicted"], args.out)


if __name__ == "__main__":
    main()
