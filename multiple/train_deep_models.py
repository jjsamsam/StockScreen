#!/usr/bin/env python3
"""
CLI to train and persist deep-learning (.h5) models for a given ticker.

- Uses existing LSTMPredictor and TransformerPredictor from stock_prediction.py
- Saves to models/<TICKER>/ via model_persistence.py with versioned filenames
  e.g., models/AAPL/lstm_YYYYMMDD_HHMMSS.h5

Examples:
  py train_deep_models.py --ticker AAPL --period 5y --models lstm transformer --force
  py train_deep_models.py -t AAPL -p 10y --models lstm

Note:
  This script requires TensorFlow installed in the active Python env.
  Install:  py -m pip install --upgrade tensorflow
"""

import argparse
import sys
from typing import List

import numpy as np
import pandas as pd

from logger_config import get_logger

logger = get_logger(__name__)


def ensure_tensorflow():
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except Exception as e:
        logger.error(f"TensorFlow not available: {e}")
        logger.info("Install with: py -m pip install tensorflow")
        return False


def load_prices(ticker: str, period: str) -> np.ndarray:
    from cache_manager import get_stock_data

    df = get_stock_data(ticker, period=period)
    if df is None or len(df) < 100:
        raise RuntimeError("Not enough data fetched. Try increasing period or check connectivity.")
    return df['Close'].values


def train_lstm(ticker: str, prices: np.ndarray, forecast_days: int, force: bool):
    from stock_prediction import LSTMPredictor

    lstm = LSTMPredictor(ticker=ticker, auto_load=True)
    result = lstm.fit_predict(prices, forecast_days=forecast_days, force_retrain=force)
    if 'error' in result:
        raise RuntimeError(f"LSTM training failed: {result['error']}")
    return result


def train_transformer(ticker: str, prices: np.ndarray, forecast_days: int, force: bool):
    from stock_prediction import TransformerPredictor

    tr = TransformerPredictor(ticker=ticker, auto_load=True)
    result = tr.fit_predict(prices, forecast_days=forecast_days, force_retrain=force)
    if 'error' in result:
        raise RuntimeError(f"Transformer training failed: {result['error']}")
    return result


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and persist LSTM/Transformer models (.h5)")
    p.add_argument('-t', '--ticker', required=True, help='Ticker symbol, e.g., AAPL or 005930.KS')
    p.add_argument('-p', '--period', default='5y', help='History period (e.g., 2y, 5y, 10y, max)')
    p.add_argument('--models', nargs='+', choices=['lstm', 'transformer'], default=['lstm', 'transformer'],
                   help='Which deep models to train')
    p.add_argument('--forecast-days', type=int, default=5, help='Days to forecast when training')
    p.add_argument('--force', action='store_true', help='Force retrain even if a model exists')
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if not ensure_tensorflow():
        return 2

    ticker = args.ticker.upper()
    logger.info(f"Training deep models for {ticker} (period={args.period}, models={args.models})")

    try:
        prices = load_prices(ticker, args.period)
    except Exception as e:
        logger.error(f"Failed to load data for {ticker}: {e}")
        return 1

    try:
        if 'lstm' in args.models:
            logger.info("Starting LSTM training…")
            train_lstm(ticker, prices, args.forecast_days, args.force)
            logger.info("LSTM training complete and saved.")

        if 'transformer' in args.models:
            logger.info("Starting Transformer training…")
            train_transformer(ticker, prices, args.forecast_days, args.force)
            logger.info("Transformer training complete and saved.")

        logger.info("All requested models trained.")
        logger.info(f"Check models/{ticker} for .h5 and metadata files.")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))


#py train_deep_models.py --ticker AAPL --period 5y --models lstm transformer --force