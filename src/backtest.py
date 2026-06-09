import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Guarantee clean local lookups for models.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from models import ClassicalQuantModels, StatisticalQuantModels


def run_walk_forward_backtest(start_test="2021-01-01", end_test="2024-12-31"):
    print("=== INITIALIZING OUT-OF-SAMPLE BACKTESTING ENGINE ===")

    # 1. Load the underlying log return matrix
    if not os.path.exists("data/log_returns.csv"):
        raise FileNotFoundError("Log returns file missing. Run src/features.py first.")
    
    log_returns = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
    tickers = log_returns.columns.tolist()
    
    # Isolate the explicit trading horizon for simulation evaluations
    test_returns = log_returns.loc[start_test:end_test]
    if test_returns.empty:
        raise ValueError("Target test index boundaries returned an empty DataFrame slice.")

    # 2. Extract monthly rebalancing anchors (First trading day of each month)
    rebalance_dates = test_returns.groupby([test_returns.index.year, test_returns.index.month]).apply(lambda x: x.index[0]).values
    rebalance_dates = sorted(list(set(rebalance_dates)))
    print(f"Generated {len(rebalance_dates)} monthly rebalancing dates across the testing horizon.\n")

    # Initialize model tracking instances
    classical_engine = ClassicalQuantModels(tickers)
    statistical_engine = StatisticalQuantModels(tickers)

    strategies = ["Equal_Weight", "Max_Sharpe", "Min_Variance", "PCA_Eigen"]
    
    # Storage structures for daily portfolio returns and historical allocation logs
    portfolio_daily_returns = {strat: [] for strat in strategies}
    historical_weights = {strat: pd.DataFrame(index=rebalance_dates, columns=tickers, data=0.0) for strat in strategies}
    
    # Initialize baseline tracking metrics
    last_weights = {strat: pd.Series(0.0, index=tickers) for strat in strategies}

    # 3. CORE SIMULATION LOOP
    for idx, raw_rebalance_date in enumerate(rebalance_dates):
        current_rebalance_date = pd.Timestamp(raw_rebalance_date)
        print(f"[{idx+1}/{len(rebalance_dates)}] Rebalancing on: {current_rebalance_date.strftime('%Y-%m-%d')}")
        
        # Define expanding window (all historical records prior to current rebalance date)
        historical_slice = log_returns.loc[:current_rebalance_date - pd.Timedelta(days=1)]
        
        # Define forward investment window
        if idx < len(rebalance_dates) - 1:
            next_rebalance_date = pd.Timestamp(rebalance_dates[idx + 1])
            forward_window_returns = test_returns.loc[current_rebalance_date:next_rebalance_date - pd.Timedelta(days=1)]
        else:
            forward_window_returns = test_returns.loc[current_rebalance_date:]

        if forward_window_returns.empty:
            continue

        # Generate allocations for each strategy using the historical window slice
        current_weights = {
            "Equal_Weight": classical_engine.allocate_equal_weight(),
            "Max_Sharpe": classical_engine.allocate_max_sharpe(historical_slice),
            "Min_Variance": classical_engine.allocate_min_variance(historical_slice),
            "PCA_Eigen": statistical_engine.allocate_pca_eigenportfolio(historical_slice)
        }

        # 4. APPLY TRANSACTION COSTS & TRACK DAILY TIMELINE PERFORMANCE
        for strat in strategies:
            w_new = current_weights[strat]
            w_old = last_weights[strat]
            
            # Log weights to our historical dataframe matrix
            historical_weights[strat].loc[raw_rebalance_date] = w_new
            
            # Calculate turnover volume: sum(abs(W_new - W_old))
            turnover = np.sum(np.abs(w_new - w_old)) if idx > 0 else 1.0
            transaction_penalty = turnover * 0.001  # 0.1% brokerage penalty constraint
            
            # Project daily asset returns into portfolio space: R_p = sum(w_i * R_i)
            strat_daily_series = forward_window_returns.dot(w_new)
            
            # Deduct the transaction fee from the first trading day of the rebalance period
            strat_daily_series.iloc[0] -= transaction_penalty
            
            # Append performance log
            portfolio_daily_returns[strat].append(strat_daily_series)
            
            # Update history tracker for the next iteration
            last_weights[strat] = w_new

    # 5. AGGREGATE RAW LOGS INTO UNIFIED DATAFRAMES
    summary_returns_df = pd.DataFrame({strat: pd.concat(portfolio_daily_returns[strat]) for strat in strategies})
    
    # Save raw performance arrays to disk
    os.makedirs("data/results", exist_ok=True)
    summary_returns_df.to_csv("data/results/backtest_daily_returns.csv")
    
    return summary_returns_df


def generate_backtest_plots(returns_df):
    """Generates and saves professional performance charts into a plots/ directory."""
    print("\nGenerating visual performance charts...")
    os.makedirs("plots", exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 6]
    
    colors = {
        "Equal_Weight": "#7f8c8d",  # Grey Baseline
        "Max_Sharpe": "#e74c3c",    # Red Aggressive
        "Min_Variance": "#2ecc71",  # Green Defensive
        "PCA_Eigen": "#3498db"      # Blue Statistical
    }
    
    # 1. CHART 1: CUMULATIVE EQUITY CURVES
    plt.figure()
    for strat in returns_df.columns:
        cumulative_equity = np.exp(returns_df[strat].cumsum())
        plt.plot(cumulative_equity, label=strat, color=colors.get(strat), linewidth=2)
    plt.title("Out-of-Sample Cumulative Equity Curves (2021-2024)\n[Adjusted for 0.1% Rebalancing Turnover Cost]", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Growth of Capital (Base 1.0)", fontsize=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.savefig("plots/cumulative_equity_curves.png", dpi=300)
    plt.close()
    print("  -> Saved Cumulative Growth Chart to: 'plots/cumulative_equity_curves.png'")
    
    # 2. CHART 2: UNDERWATER DRAWDOWN MAPS
    plt.figure()
    for strat in returns_df.columns:
        cumulative_equity = np.exp(returns_df[strat].cumsum())
        running_max = cumulative_equity.cummax()
        drawdown_series = (cumulative_equity - running_max) / running_max
        plt.fill_between(drawdown_series.index, drawdown_series, 0, label=strat, color=colors.get(strat), alpha=0.15)
        plt.plot(drawdown_series, color=colors.get(strat), linewidth=1.5)
    plt.title("Out-of-Sample Underwater Drawdown Profiles (2021-2024)", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Drawdown Magnitude (%)", fontsize=12)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig("plots/underwater_drawdowns.png", dpi=300)
    plt.close()
    print("  -> Saved Underwater Drawdown Map to: 'plots/underwater_drawdowns.png'")

    # 3. NEW UPGRADE - CHART 3: CUMULATIVE CAPM ALPHA GENERATION
    plt.figure()
    benchmark = returns_df["Equal_Weight"]
    benchmark_var = benchmark.var()
    
    for strat in returns_df.columns:
        if strat == "Equal_Weight":
            # Baseline benchmark alpha is structurally flat zero line
            plt.plot(returns_df[strat].index, np.zeros(len(returns_df)), label=f"{strat} (Benchmark)", color=colors.get(strat), linestyle="--")
            continue
            
        # Calculate systematic beta relative to Equal Weight index benchmark
        beta = returns_df[strat].cov(benchmark) / benchmark_var if benchmark_var > 0 else 1.0
        
        # Isolate daily Alpha: R_p - Beta * R_b
        daily_alpha = returns_df[strat] - beta * benchmark
        cumulative_alpha = daily_alpha.cumsum() * 100  # Scale to percentage format
        
        plt.plot(cumulative_alpha, label=f"{strat} (β: {beta:.2f})", color=colors.get(strat), linewidth=2)
        
    plt.title("Out-of-Sample Cumulative Alpha Generation (2021-2024)\n[Risk-Adjusted Excess Returns vs Equal Weight Baseline]", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline", fontsize=12)
    plt.ylabel("Cumulative Alpha (%)", fontsize=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.tight_layout()
    plt.savefig("plots/cumulative_alpha_curves.png", dpi=300)
    plt.close()
    print("  -> Saved Cumulative Alpha Curve to: 'plots/cumulative_alpha_curves.png'")


def compute_performance_metrics(returns_df):
    """Computes comprehensive risk and alpha performance metrics (Phase 6)."""
    print("\n" + "="*65 + "\nQUANTITATIVE PERFORMANCE & RISK RATIO SCORECARD\n" + "="*65)
    scorecard = {}
    
    benchmark = returns_df["Equal_Weight"]
    benchmark_var = benchmark.var()
    
    for strat in returns_df.columns:
        series = returns_df[strat]
        
        # Absolute Return metrics
        cum_return = np.exp(series.sum()) - 1
        n_days = len(series)
        annualized_return = (cum_return + 1) ** (252 / n_days) - 1
        annualized_vol = series.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Drawdown calculation
        cumulative_equity = np.exp(series.cumsum())
        running_max = cumulative_equity.cummax()
        drawdown = (cumulative_equity - running_max) / running_max
        max_dd = drawdown.min()
        
        # FACTOR ANALYSIS LOGIC (Alpha & Beta)
        if strat == "Equal_Weight":
            beta = 1.0
            annualized_alpha = 0.0
        else:
            beta = series.cov(benchmark) / benchmark_var if benchmark_var > 0 else 1.0
            daily_alpha = series - beta * benchmark
            annualized_alpha = daily_alpha.mean() * 252

        scorecard[strat] = {
            "Total Return": f"{cum_return * 100:.2f}%",
            "Annualized Return": f"{annualized_return * 100:.2f}%",
            "Annualized Volatility": f"{annualized_vol * 100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.4f}",
            "Max Drawdown": f"{max_dd * 100:.2f}%",
            "Systematic Beta (β)": f"{beta:.2f}",
            "Annualized Alpha (α)": f"{annualized_alpha * 100:.2f}%" if strat != "Equal_Weight" else "0.00% (Base)"
        }
        
    score_df = pd.DataFrame(scorecard)
    print(score_df.to_string())
    score_df.to_csv("data/results/performance_scorecard.csv")
    print("\nScorecard matrix written to 'data/results/performance_scorecard.csv'")


if __name__ == "__main__":
    daily_returns = run_walk_forward_backtest()
    generate_backtest_plots(daily_returns)
    compute_performance_metrics(daily_returns)