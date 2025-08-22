import numpy as np
import pandas as pd

class PortfolioVaR:
    def __init__(self, price_data: pd.DataFrame, tickers: list, weights: np.ndarray, initial_portfolio_value: float = 1000000):
        """
        Initializes the VaR calculator.
        
        Args:
            price_data (pd.DataFrame): DataFrame with dates as index and tickers as columns.
            tickers (list): List of stock tickers to use for the portfolio.
            weights (np.ndarray): Array of portfolio weights.
        """
        self.portfolio_prices = price_data[tickers].copy()
        self.weights = weights
        self.initial_value = initial_portfolio_value 
        
        # Internal parameters estimated from data
        self.mu = None
        self.Sigma = None
        self._estimate_parameters()

    def _estimate_parameters(self):
        """Private method to calculate drift and covariance from price data."""
        log_returns = np.log(self.portfolio_prices).diff().dropna()
        self.mu = log_returns.mean()
        self.Sigma = log_returns.cov()

    def run_simulation(self, T_days: int = 1, n_sims: int = 10000) -> np.ndarray:
        """
        Runs the multi-asset Monte Carlo simulation and returns the P&L distribution.
        """
        num_assets = len(self.portfolio_prices.columns)
        S0 = self.portfolio_prices.iloc[-1].values
        
        # Calculate the number of shares based on initial value and weights
        num_shares = (self.initial_value * self.weights) / S0
        
        # Run the simulation engine
        L = np.linalg.cholesky(self.Sigma)
        Z = np.random.standard_normal((num_assets, n_sims))
        
        drift = self.mu.values.reshape(num_assets, 1) * T_days
        diffusion = L @ Z * np.sqrt(T_days)
        simulated_log_returns = drift + diffusion
        
        S0_reshaped = S0.reshape(num_assets, 1)
        simulated_terminal_prices = S0_reshaped * np.exp(simulated_log_returns)
        
        # Calculate final portfolio values and P&L
        simulated_portfolio_values = np.dot(num_shares, simulated_terminal_prices)
        pnl = simulated_portfolio_values - self.initial_value
        
        return pnl

    @staticmethod
    def calculate_var(pnl: np.ndarray, confidence_level: float = 0.99) -> float:
        """Calculates Value at Risk from a P&L distribution."""
        percentile = 100 * (1 - confidence_level)
        return np.percentile(pnl, percentile)

    @staticmethod
    def calculate_cvar(pnl: np.ndarray, confidence_level: float = 0.99) -> float:
        """Calculates Conditional Value at Risk (Expected Shortfall)."""
        var = PortfolioVaR.calculate_var(pnl, confidence_level)
        shortfall_losses = pnl[pnl <= var]
        return shortfall_losses.mean()

if __name__ == '__main__':
    test_data = {
        'STOCK_A': [100, 101, 102, 103],
        'STOCK_B': [50, 50.5, 49.5, 50]
    }
    test_dates = pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04'])
    test_df = pd.DataFrame(test_data, index=test_dates)
    
    tickers = ['STOCK_A', 'STOCK_B']
    weights = np.array([0.5, 0.5])
    
    test_model = PortfolioVaR(price_data=test_df, tickers=tickers, weights=weights)
    pnl = test_model.run_simulation(n_sims=100) # Quick simulation
    var = test_model.calculate_var(pnl)
    
    print("--- SELF-TEST ---")
    print(f"Test VaR (99%): ${abs(var):,.2f}")

