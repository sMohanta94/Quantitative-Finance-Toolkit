import numpy as np
from scipy.optimize import minimize
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
sns.set_theme(style="whitegrid", palette="viridis")

class FFTOptionPricer:
    """
    A flexible option pricer and calibrator using the Carr-Madan FFT method.
    Supports GBM, Heston, Merton (Jump-Diffusion), Bates, and Variance Gamma models.
    Includes a calibration method to fit model parameters to market prices.
    """
    def __init__(self, model_name: str, params: list = None):
        """
        Initializes the pricer with a specific model and its parameters.

        Args:
            model_name (str): The name of the model ('GBM', 'Heston', 'Merton', 'Bates', 'VG').
            params (list): A list of parameters corresponding to the chosen model.
        """
        self.model_name = model_name.upper()
        if params:
            self.params = params
            self._validate_params()

    def _validate_params(self):
        """Checks if the correct number of parameters is provided for the selected model."""
        param_counts = {'GBM': 1, 'HESTON': 5, 'MERTON': 4, 'BATES': 8, 'VG': 3}
        if self.model_name not in param_counts:
            raise ValueError(f"Model '{self.model_name}' is not supported.")
        if len(self.params) != param_counts[self.model_name]:
            raise ValueError(f"Incorrect number of parameters for {self.model_name}. "
                             f"Expected {param_counts[self.model_name]}, got {len(self.params)}.")

    def _param_mapping(self, x, c, d):
        """Maps a parameter x into a valid range [c, d] using periodic reflection."""
        if c <= x <= d:
            return x
        range_val = d - c
        n = np.floor((x - c) / range_val)
        if n % 2 == 0:
            return x - n * range_val
        else:
            return d + n * range_val - (x - c)

    def _characteristic_function(self, u, S0, r, q, T):
        """Calculates the characteristic function for the selected model."""
        current_params = self.params
        
        if self.model_name in ['HESTON', 'BATES']:
            if self.model_name == 'HESTON':
                kappa, theta, sigma, rho, v0 = self.params
            else: # BATES
                kappa, theta, sigma, rho, v0, _, _, _ = self.params
            
            kappa = self._param_mapping(kappa, 0.1, 20)
            theta = self._param_mapping(theta, 0.001, 0.4)
            sigma = self._param_mapping(sigma, 0.01, 0.8)
            rho   = self._param_mapping(rho, -0.99, 0.99)
            v0    = self._param_mapping(v0, 0.005, 0.4)
            
            if self.model_name == 'HESTON':
                current_params = [kappa, theta, sigma, rho, v0]
            else:
                current_params = [kappa, theta, sigma, rho, v0] + self.params[5:]

        if self.model_name == 'GBM':
            sig = current_params[0]
            mu = np.log(S0) + (r - q - sig**2 / 2) * T
            a = sig * np.sqrt(T)
            return np.exp(1j * mu * u - (a * u)**2 / 2)

        elif self.model_name == 'HESTON':
            kappa, theta, sigma, rho, v0 = current_params
            tmp = (kappa - 1j * rho * sigma * u)
            g = np.sqrt((sigma**2) * (u**2 + 1j * u) + tmp**2)
            pow1 = 2 * kappa * theta / (sigma**2)
            numer1 = (kappa * theta * T * tmp) / (sigma**2) + 1j * u * T * r + 1j * u * np.log(S0)
            log_denum1 = pow1 * np.log(np.cosh(g * T / 2) + (tmp / g) * np.sinh(g * T / 2))
            tmp2 = ((u**2 + 1j * u) * v0) / (g / np.tanh(g * T / 2) + tmp)
            log_phi = numer1 - log_denum1 - tmp2
            return np.exp(log_phi)

        elif self.model_name == 'MERTON':
            sig, lam, mu_j, sig_j = current_params
            m = np.exp(mu_j + 0.5 * sig_j**2) - 1
            mu_gbm = np.log(S0) + (r - q - sig**2 / 2 - lam * m) * T
            a = sig * np.sqrt(T)
            phi_gbm = np.exp(1j * mu_gbm * u - (a * u)**2 / 2)
            phi_jump = np.exp(lam * T * (np.exp(1j * u * mu_j - 0.5 * (u * sig_j)**2) - 1))
            return phi_gbm * phi_jump

        elif self.model_name == 'BATES':
            kappa, theta, sigma, rho, v0, lam, mu_j, sig_j = current_params
            heston_params = [kappa, theta, sigma, rho, v0]
            m = np.exp(mu_j + 0.5 * sig_j**2) - 1
            pricer_heston = FFTOptionPricer('Heston', heston_params)
            phi_heston = pricer_heston._characteristic_function(u, S0, r - lam * m, q, T)
            phi_jump = np.exp(lam * T * (np.exp(1j * u * mu_j - 0.5 * (u * sig_j)**2) - 1))
            return phi_heston * phi_jump
            
        elif self.model_name == 'VG':
            sigma_vg, nu_vg, theta_vg = current_params
            omega = (1/nu_vg) * np.log(1 - theta_vg * nu_vg - 0.5 * sigma_vg**2 * nu_vg)
            mu = np.log(S0) + (r - q + omega) * T
            phi = np.exp(1j * u * mu) * ((1 - 1j * nu_vg * theta_vg * u + 0.5 * nu_vg * sigma_vg**2 * u**2)**(-T/nu_vg))
            return phi

    def price_call_fft(self, S0, K, r, q, T, alpha=1.5, eta=0.25, n=12):
        N = 2**n
        lda = (2 * np.pi / N) / eta
        beta = np.log(S0) - N * lda / 2
        
        km = np.zeros(N)
        xX = np.zeros(N, dtype=complex)
        df = np.exp(-r * T)
        nuJ = np.arange(N) * eta
        
        psi_nuJ = self._characteristic_function(nuJ - (alpha + 1) * 1j, S0, r, q, T) / \
                  ((alpha + 1j * nuJ) * (alpha + 1 + 1j * nuJ))
        
        for j in range(N):
            km[j] = beta + j * lda
            wJ = eta / 2 if j == 0 else eta
            xX[j] = np.exp(-1j * beta * nuJ[j]) * df * psi_nuJ[j] * wJ
            
        yY = np.fft.fft(xX)
        cT_km = np.zeros(N)
        
        for i in range(N):
            multiplier = np.exp(-alpha * km[i]) / np.pi
            cT_km[i] = multiplier * np.real(yY[i])
            
        return np.exp(km), cT_km

    def price(self, S0, K, r, q, T, option_type='call', **fft_params):
        strikes, call_prices = self.price_call_fft(S0, K, r, q, T, **fft_params)
        call_price_at_K = np.interp(K, strikes, call_prices)
        
        if option_type.lower() == 'call':
            return call_price_at_K
        elif option_type.lower() == 'put':
            return call_price_at_K - S0 * np.exp(-q * T) + K * np.exp(-r * T)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def calculate_rmse(self, params, market_prices, maturities, strikes, S0, r, q, **fft_params):
        """
        Calculates the Root Mean Squared Error for a given set of parameters.
        This is the objective function for calibration.
        """
        self.params = list(params)
        error = 0
        
        if not isinstance(market_prices[0], (list, np.ndarray)):
            market_prices = [market_prices]
            strikes = [strikes]

        for i, T_val in enumerate(maturities):
            k_range, c_range = self.price_call_fft(S0, S0, r, q, T_val, **fft_params)
            model_prices_T = np.interp(strikes[i], k_range, c_range)
            error += np.sum((market_prices[i] - model_prices_T)**2)
            
        return np.sqrt(error / sum(len(s) for s in strikes))

    def calibrate(self, market_prices, maturities, strikes, S0, r, q, initial_guess, **fft_params):
        """
        Calibrates the model parameters to fit market prices using an optimizer.
        """
        objective_function = lambda params: self.calculate_rmse(
            params, market_prices, maturities, strikes, S0, r, q, **fft_params
        )

        result = minimize(objective_function, initial_guess, method='Nelder-Mead', options={'maxiter': 2000})
        self.params = list(result.x)
        return result

if __name__ == '__main__':
    # --- Dummy Market Data for Calibration Example ---
    S0, r, q = 100.0, 0.05, 0.02
    maturities = [0.5, 1.0]
    strikes = [np.array([90, 100, 110]), np.array([90, 100, 110, 120])]
    true_heston_params = [3.0, 0.05, 0.2, -0.6, 0.04]
    true_pricer = FFTOptionPricer('Heston', true_heston_params)
    market_prices = [
        true_pricer.price(S0, strikes[0], r, q, maturities[0], 'call'),
        true_pricer.price(S0, strikes[1], r, q, maturities[1], 'call')
    ]

    # --- Calibration Example for Heston Model ---
    print("--- Heston Model Calibration ---")
    heston_calibrator = FFTOptionPricer('Heston')

    # 1. EXPLORATORY STEP: Visualize the error surface between two plausible parameter sets
    params1 = np.array([2.0, 0.04, 0.25, -0.7, 0.04])
    params2 = np.array([4.0, 0.06, 0.15, -0.5, 0.05])
    
    mix_factors = np.linspace(0, 1, 21)
    rmse_values = []
    param_sets = []

    print("\n1. Exploring parameter space to find a good initial guess...")
    for mix in mix_factors:
        interp_params = mix * params1 + (1.0 - mix) * params2
        param_sets.append(interp_params)
        rmse = heston_calibrator.calculate_rmse(interp_params, market_prices, maturities, strikes, S0, r, q)
        rmse_values.append(rmse)

    # Find the best guess from the exploratory search
    min_rmse_index = np.argmin(rmse_values)
    initial_guess = param_sets[min_rmse_index]
    
    plt.figure(figsize=(7, 5))
    plt.plot(mix_factors, rmse_values, 'o--')
    plt.scatter(mix_factors[min_rmse_index], rmse_values[min_rmse_index], color='red', s=100, zorder=5, label=f'Best Guess (RMSE: {rmse_values[min_rmse_index]:.4f})')
    plt.xlabel('Mixing Factor')
    plt.ylabel('RMSE')
    plt.title('RMSE as a function of Parameter Interpolation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # 2. OPTIMIZATION STEP: Use the best guess to start the optimizer
    print(f"\n2. Starting optimization with the best guess found: {np.round(initial_guess, 4)}")
    
    calibration_result = heston_calibrator.calibrate(
        market_prices, maturities, strikes, S0, r, q, initial_guess
    )
    
    print("\nCalibration Complete.")
    print(f"Final RMSE: {calibration_result.fun:.6f}")
    print(f"Calibrated Parameters: {np.round(heston_calibrator.params, 4)}")
    print(f"True Parameters:       {true_heston_params}")
