import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf


class ClassicalQuantModels:
    def __init__(self, asset_tickers):
        self.tickers = asset_tickers
        self.num_assets = len(asset_tickers)

    def allocate_equal_weight(self):
        """Implements the simple 1/N Benchmark portfolio baseline."""
        weights = np.ones(self.num_assets) / self.num_assets
        return pd.Series(weights, index=self.tickers)

    def allocate_max_sharpe(self, historical_returns, risk_free_rate=0.06, min_periods=252):
        """
        Executes Markowitz Mean-Variance Optimization to maximize the Sharpe Ratio.
        Uses Ledoit-Wolf shrinkage on covariance and James-Stein shrinkage on returns 
        to stabilize allocations against historical noise.
        """
        # Filter out assets without enough history in this specific window slice
        valid_counts = historical_returns.notna().sum()
        has_variance = historical_returns.std() > 1e-8
        active_mask = (valid_counts >= min_periods) & has_variance
        active_tickers = historical_returns.columns[active_mask].tolist()
        n_active = len(active_tickers)

        print(f"  [Max Sharpe] Active tickers after filter: {n_active}")

        if n_active < 2:
            print(">> Max Sharpe: fewer than 2 active assets. Returning equal weight.")
            return self.allocate_equal_weight()

        # Handle Strategy A NaNs: forward fill and drop leading non-existent blocks
        active_returns = (historical_returns[active_tickers]
                          .ffill()
                          .dropna(how='any'))

        if active_returns.shape[0] < min_periods:
            print(f">> Max Sharpe: only {active_returns.shape[0]} clean rows after fill. Equal weight fallback.")
            return self.allocate_equal_weight()

        # Ledoit-Wolf covariance shrinkage (annualized)
        lw = LedoitWolf().fit(active_returns)
        sigma_active = lw.covariance_ * 252

        # James-Stein shrinkage on expected return vectors
        raw_means = active_returns.mean() * 252
        grand_mean = raw_means.mean()
        r_active = 0.5 * raw_means.values + 0.5 * grand_mean

        def objective(weights):
            p_return = np.dot(r_active, weights)
            p_vol = np.sqrt(weights @ sigma_active @ weights)
            if p_vol < 1e-10:
                return 0.0
            return -(p_return - risk_free_rate) / p_vol

        # Constraints & Boundaries: Fully invested, Long-only, Max 10% per stock
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 0.10) for _ in range(n_active))
        initial_guess = np.ones(n_active) / n_active

        result = minimize(
            objective, initial_guess,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )

        final_weights = pd.Series(0.0, index=self.tickers)
        if result.success:
            final_weights[active_tickers] = result.x
            print(f"  [Max Sharpe] Optimization succeeded.")
        else:
            print(f">> Max Sharpe FAILED: {result.message}. Equal weight fallback.")
            final_weights[active_tickers] = initial_guess

        return final_weights

    def allocate_min_variance(self, historical_returns, min_periods=252):
        """
        Minimum Variance portfolio — ignores expected returns entirely.
        Optimizes strictly to minimize global portfolio volatility.
        Highly stable out-of-sample due to zero return-estimation errors.
        """
        valid_counts = historical_returns.notna().sum()
        has_variance = historical_returns.std() > 1e-8
        active_mask = (valid_counts >= min_periods) & has_variance
        active_tickers = historical_returns.columns[active_mask].tolist()
        n_active = len(active_tickers)

        print(f"  [Min Var] Active tickers after filter: {n_active}")

        if n_active < 2:
            print(">> Min Var: fewer than 2 active assets. Returning equal weight.")
            return self.allocate_equal_weight()

        # Handle Strategy A NaNs: forward fill and drop leading non-existent blocks
        active_returns = (historical_returns[active_tickers]
                          .ffill()
                          .dropna(how='any'))

        if active_returns.shape[0] < min_periods:
            print(f">> Min Var: only {active_returns.shape[0]} clean rows after fill. Equal weight fallback.")
            return self.allocate_equal_weight()

        # Ledoit-Wolf covariance shrinkage (annualized)
        lw = LedoitWolf().fit(active_returns)
        sigma_active = lw.covariance_ * 252

        # Quadratic Objective Function: Minimize total portfolio variance
        def objective(weights):
            return weights @ sigma_active @ weights

        # Constraints & Boundaries: Fully invested, Long-only, Max 10% per stock
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 0.10) for _ in range(n_active))
        initial_guess = np.ones(n_active) / n_active

        result = minimize(
            objective, initial_guess,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )

        final_weights = pd.Series(0.0, index=self.tickers)
        if result.success:
            final_weights[active_tickers] = result.x
            print(f"  [Min Var] Optimization succeeded.")
        else:
            print(f">> Min Var FAILED: {result.message}. Equal weight fallback.")
            final_weights[active_tickers] = initial_guess

        return final_weights


class StatisticalQuantModels:
    def __init__(self, asset_tickers):
        self.tickers = asset_tickers
        self.num_assets = len(asset_tickers)

    def allocate_pca_eigenportfolio(self, historical_returns, min_periods=252):
        """
        Decomposes asset returns into principal components using the covariance matrix.
        Combines components up to 90% cumulative variance to build an eigen-portfolio.
        """
        valid_counts = historical_returns.notna().sum()
        active_tickers = historical_returns.columns[valid_counts >= min_periods].tolist()
        n_active = len(active_tickers)

        print(f"  [PCA] Active tickers after filter: {n_active}")

        if n_active == 0:
            return pd.Series(1.0 / self.num_assets, index=self.tickers)

        active_returns = (historical_returns[active_tickers]
                          .ffill()
                          .dropna(how='any'))

        n_components = min(12, n_active)
        pca = PCA(n_components=n_components)
        pca.fit(active_returns)

        print(f"  [PCA] Explained variance ratio (top {n_components} PCs):")
        for i, v in enumerate(pca.explained_variance_ratio_):
            cumv = np.cumsum(pca.explained_variance_ratio_)[i]
            print(f"    PC{i+1}: {v:.4f}  (cumulative: {cumv:.4f})")

        # Retain components up to a 80% cumulative variance floor
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, 0.80)) + 1
        n_keep = min(n_keep, n_components)
        print(f"  [PCA] Using {n_keep} components")

        kept_var = pca.explained_variance_ratio_[:n_keep]
        kept_var = kept_var / kept_var.sum()

        composite = np.zeros(n_active)
        for comp, var_ratio in zip(pca.components_[:n_keep], kept_var):
            composite += var_ratio * np.abs(comp)

        weights = composite / composite.sum()

        print(f"  [PCA] Raw weight stats — min: {weights.min():.4f}  max: {weights.max():.4f}  std: {weights.std():.4f}")

        # Clip individual exposures to enforce the 10% ceiling, then renormalize
        weights = np.clip(weights, 0.0, 0.10)
        weights = weights / weights.sum()

        final_weights = pd.Series(0.0, index=self.tickers)
        final_weights[active_tickers] = weights
        return final_weights


if __name__ == "__main__":


    
    train_df = pd.read_csv("data/splits/train_returns.csv", index_col=0, parse_dates=True)
    tickers = train_df.columns.tolist()
    print(f"Loaded train_df: {train_df.shape} | "
            f"NaN count per ticker (sample): "
            f"{train_df.isna().sum().sort_values(ascending=False).head(5).to_dict()}")

    classical_engine = ClassicalQuantModels(tickers)
    statistical_engine = StatisticalQuantModels(tickers)

    print("\n--- EQUAL WEIGHT ---")
    ew = classical_engine.allocate_equal_weight()
    print(f"  N: {len(ew)} | Each: {ew.iloc[0]:.6f} | Sum: {ew.sum():.4f}")

    print("\n--- MAX SHARPE ---")
    mvo = classical_engine.allocate_max_sharpe(train_df)
    nz = mvo[mvo > 0.001].sort_values(ascending=False)
    print(nz.head(10).to_string())
    print(f"  Selected: {len(nz)} | Sum: {mvo.sum():.4f} | Max: {mvo.max():.4f}")

    print("\n--- GLOBAL MINIMUM VARIANCE ---")
    min_var = classical_engine.allocate_min_variance(train_df)
    nz_minvar = min_var[min_var > 0.001].sort_values(ascending=False)
    print(nz_minvar.head(10).to_string())
    print(f"  Selected: {len(nz_minvar)} | Sum: {min_var.sum():.4f} | Max: {min_var.max():.4f}")

    print("\n--- PCA EIGEN-PORTFOLIO ---")
    pca_w = statistical_engine.allocate_pca_eigenportfolio(train_df)
    nz_pca = pca_w[pca_w > 0.001].sort_values(ascending=False)
    print(nz_pca.head(15).to_string())
    print(f"  Selected: {len(nz_pca)} | Sum: {pca_w.sum():.4f} | Max: {pca_w.max():.4f}")
    


