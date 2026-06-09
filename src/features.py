import os
import numpy as np
import pandas as pd


def calculate_log_returns(price_df):
    """Computes daily log returns for the entire price matrix."""
    print("Calculating daily log returns...")
    # ln(P_t / P_{t-1})
    log_returns = np.log(price_df / price_df.shift(1))
    return log_returns


def calculate_rsi(price_df, period=14):
    """
    Computes the standard 14-day Relative Strength Index (RSI) for each asset.
    Purely historical—no forward-looking data points are utilized.
    """
    print(f"Calculating {period}-day RSI scores...")
    rsi_df = pd.DataFrame(index=price_df.index, columns=price_df.columns)

    for ticker in price_df.columns:
        delta = price_df[ticker].diff()

        # Isolate gains and losses
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Calculate exponential moving averages
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

        # Avoid division by zero bugs if an asset experiences absolute flatlines
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Whenever avg_loss was 0, RSI is structurally 100
        rsi_df[ticker] = rsi.fillna(100)

    return rsi_df


def calculate_momentum(price_df, window=20):
    """
    Computes rolling price momentum (rate of change over the specified window).
    Using a 20-day trading window (~1 calendar month).
    """
    print(f"Calculating {window}-day price momentum...")
    # P_t / P_{t-window} - 1
    momentum = price_df.pct_change(periods=window)
    return momentum


def build_feature_space():
    """Loads cleaned prices, builds features, and exports clean independent state matrices."""

    # 1. Load data
    if not os.path.exists("data/cleaned_prices.csv"):
        raise FileNotFoundError(
            "Cleaned prices not found. Please run src/dataset.py first."
        )

    prices = pd.read_csv("data/cleaned_prices.csv", index_col=0, parse_dates=True)

    # 2. Extract Features
    log_returns = calculate_log_returns(prices)
    rsi_14 = calculate_rsi(prices, period=14)
    momentum_20 = calculate_momentum(prices, window=20)

    # 3. Save feature matrices to disk
    log_returns.to_csv("data/log_returns.csv")
    rsi_14.to_csv("data/feature_rsi.csv")
    momentum_20.to_csv("data/feature_momentum.csv")

    print(f" - data/log_returns.csv       Shape: {log_returns.shape}")
    print(f" - data/feature_rsi.csv       Shape: {rsi_14.shape}")
    print(f" - data/feature_momentum.csv  Shape: {momentum_20.shape}\n")


def generate_walk_forward_splits():
    """
    Loads features, cuts them into chronological Train/Val/Test segments,
    and applies Z-score normalization fitted strictly on the Train window.
    """
    print("=== STARTING PHASE 3: CHRONOLOGICAL SPLITS & SCALING ===")
    
    # 1. Load the matrices generated in Phase 2
    log_returns = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
    rsi = pd.read_csv("data/feature_rsi.csv", index_col=0, parse_dates=True)
    momentum = pd.read_csv("data/feature_momentum.csv", index_col=0, parse_dates=True)
    
    # 2. Define strict chronological boundaries
    train_start, train_end = "2010-01-01", "2018-12-31"
    val_start, val_end     = "2019-01-01", "2020-12-31"
    test_start, test_end   = "2021-01-01", "2024-12-31"
    
    # 3. Cut Log Returns (Keep them unscaled for final portfolio optimization engines)
    train_returns = log_returns.loc[train_start:train_end]
    val_returns   = log_returns.loc[val_start:val_end]
    test_returns  = log_returns.loc[test_start:test_end]
    
    # 4. Concatenate state features (RSI + Momentum) into a single structural asset feature dataframe
    # To pass features to our VAE, we stack them cleanly
    train_raw_rsi = rsi.loc[train_start:train_end]
    train_raw_mom = momentum.loc[train_start:train_end]
    
    val_raw_rsi = rsi.loc[val_start:val_end]
    val_raw_mom = momentum.loc[val_start:val_end]
    
    test_raw_rsi = rsi.loc[test_start:test_end]
    test_raw_mom = momentum.loc[test_start:test_end]
    
    # 5. DEFENSIVE Z-SCORE SCALING ENGINE (Fitted exclusively on TRAIN)
    # Calculate historical means and standard deviations asset-by-asset
    rsi_mean, rsi_std = train_raw_rsi.mean(), train_raw_rsi.std()
    mom_mean, mom_std = train_raw_mom.mean(), train_raw_mom.std()
    
    # Apply transformation: (X - Mean) / Std
    # Replace zeros in std to prevent division by zero runtime errors
    train_scaled_rsi = (train_raw_rsi - rsi_mean) / rsi_std.replace(0, 1)
    train_scaled_mom = (train_raw_mom - mom_mean) / mom_std.replace(0, 1)
    
    val_scaled_rsi = (val_raw_rsi - rsi_mean) / rsi_std.replace(0, 1)
    val_scaled_mom = (val_raw_mom - mom_mean) / mom_std.replace(0, 1)
    
    test_scaled_rsi = (test_raw_rsi - rsi_mean) / rsi_std.replace(0, 1)
    test_scaled_mom = (test_raw_mom - mom_mean) / mom_std.replace(0, 1)
    
    # 6. Save split matrices to disk
    os.makedirs("data/splits", exist_ok=True)
    
    # Save target return arrays
    train_returns.to_csv("data/splits/train_returns.csv")
    val_returns.to_csv("data/splits/val_returns.csv")
    test_returns.to_csv("data/splits/test_returns.csv")
    
    # Save scaled features
    train_scaled_rsi.to_csv("data/splits/train_scaled_rsi.csv")
    train_scaled_mom.to_csv("data/splits/train_scaled_mom.csv")
    
    val_scaled_rsi.to_csv("data/splits/val_scaled_rsi.csv")
    val_scaled_mom.to_csv("data/splits/val_scaled_mom.csv")
    
    test_scaled_rsi.to_csv("data/splits/test_scaled_rsi.csv")
    test_scaled_mom.to_csv("data/splits/test_scaled_mom.csv")
    
    print("\nPhase 3 Complete! Data segmented into 'data/splits/'.")
    print(f" - Train Returns Size:      {train_returns.shape}")
    print(f" - Validation Returns Size: {val_returns.shape}")
    print(f" - Test Out-of-Sample Size: {test_returns.shape}\n")


if __name__ == "__main__":
    build_feature_space()
    generate_walk_forward_splits()