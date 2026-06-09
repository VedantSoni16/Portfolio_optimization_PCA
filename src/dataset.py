import os
import time
import pandas as pd
import yfinance as yf

# ==========================================
# 1. DEFINE INVESTMENT UNIVERSE
# ==========================================
NIFTY_SUPERSET = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", 
    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LTIM.NS", "HINDUNILVR.NS",
    "BAJAJFINSV.NS", "BAJFINANCE.NS", "MARUTI.NS", "HCLTECH.NS", "AXISBANK.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "ASIANPAINT.NS", "NESTLEIND.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "GRASIM.NS", "POWERGRID.NS", "NTPC.NS", "INDUSINDBK.NS", "KOTAKBANK.NS",
    
    "YESBANK.NS", "ZEEL.NS", "VEDL.NS", "GAIL.NS", "WIPRO.NS",
    "ONGC.NS", "COALINDIA.NS", "HINDALCO.NS", "TATAMOTORS.NS", "BPCL.NS",
    "IOC.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "DIVISLAB.NS",
    "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "BRITANNIA.NS", "SHRIRAMFIN.NS",
    "BEL.NS", "HAL.NS", "TRENT.NS", "SBILIFE.NS", "HDFCLIFE.NS"
]

# ==========================================
# 2. DEFENSIVE LOOP INGESTION ENGINE
# ==========================================
def download_raw_data(tickers, start_date="2010-01-01", end_date="2024-12-31"):
    """
    Downloads historical data asset-by-asset.
    Prevents individual API drops from corrupting the collective dataset.
    """
    print("=== STEP 1: DEFENSIVE INGESTION LOOP ===")
    all_series = {}
    failed_tickers = []
    
    total_tickers = len(tickers)
    for index, ticker in enumerate(tickers, start=1):
        print(f"[{index}/{total_tickers}] Processing: {ticker}...")
        try:
            # Download individually without multi-thread assembly failure risk
            # Force auto_adjust=False to isolate pure retroactive Adjusted Close values
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
            
            if not data.empty:
                # Defensive check for yfinance multi-index column layouts
                if 'Adj Close' in data.columns:
                    target_series = data['Adj Close']
                    
                    # If returned as a multi-index DataFrame, isolate the first column vector
                    if isinstance(target_series, pd.DataFrame):
                        target_series = target_series.iloc[:, 0]
                        
                    all_series[ticker] = target_series
                else:
                    print(f"--> Warning: 'Adj Close' missing for {ticker}")
                    failed_tickers.append(ticker)
            else:
                print(f"--> Warning: Empty response for {ticker}")
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"--> Critical Exception encountered on {ticker}: {e}")
            failed_tickers.append(ticker)
            
        # Standard API safety cooldown delay to block remote IP rate limiting
        time.sleep(0.3)
        
    if not all_series:
        raise ValueError("Fatal Error: Ingestion yielded zero functional data frames.")
        
    # Line up all individual asset vectors cleanly into a unified chronological matrix
    adj_close_matrix = pd.DataFrame(all_series)
    
    # Commit raw structural data to data directory
    os.makedirs("data", exist_ok=True)
    raw_path = "data/raw_prices.csv"
    adj_close_matrix.to_csv(raw_path)
    
    print(f"\nIngestion Complete! Failed Tickers logged: {failed_tickers}")
    print(f"Raw Matrix Saved to '{raw_path}' with Shape: {adj_close_matrix.shape}\n")
    return adj_close_matrix

def clean_portfolio_data(df):
    """
    Cleans raw price data while adhering to Strategy A guidelines:
    - Retains structural NaNs for companies before their official IPO date.
    - Forward-fills brief transactional gaps up to a strict 5-day limit.
    - Evicts assets completely if they fail the minimum 5-year data footprint.
    """
    cleaned_df = df.copy()
    
    # A. Forward-fill micro-gaps (holidays, processing anomalies, brief trading halts)
    cleaned_df = cleaned_df.ffill(limit=5)
    
    # B. Enforce data longevity constraint (Min 5 Years ~ 1250 trading days)
    min_trading_days = 1250
    valid_data_counts = cleaned_df.notna().sum()
    
    insufficient_assets = valid_data_counts[valid_data_counts < min_trading_days].index.tolist()
    
    if insufficient_assets:
        print(f">> Evicting tickers with insufficient history (< 5 years): {insufficient_assets}")
        cleaned_df = cleaned_df.drop(columns=insufficient_assets)
    else:
        print("All tracked tickers satisfy the 5-year data longevity threshold.")
        
    # C. Commit sanitized matrix to storage
    clean_path = "data/cleaned_prices.csv"
    cleaned_df.to_csv(clean_path)
    
    print(f"Filtered Matrix Shape: {cleaned_df.shape}")
    return cleaned_df

if __name__ == "__main__":
    raw_prices = download_raw_data(NIFTY_SUPERSET)
    clean_prices = clean_portfolio_data(raw_prices)