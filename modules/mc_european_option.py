import numpy as np
from scipy.stats import norm
from typing import Protocol, Optional, Literal, Tuple 
# import matplotlib.pyplot as plt
# import seaborn as sns

class RandomNumberGenerator(Protocol):
    def get_rvs(self, n_steps:int, n_paths:int) -> np.ndarray:
        ...

class StandardNormalGenerator:
    def __init__(self, seed:int = None, moment_matching: bool = True):  # Set to False when n_paths ~ 1e6 for better performance
        self.rng = np.random.default_rng(seed)
        self.moment_matching = moment_matching
    def get_rvs(self, n_steps:int, n_paths:int) -> np.ndarray:    
        Z = self.rng.standard_normal((n_steps, n_paths))
        if n_paths > 1 and self.moment_matching:
            mu = np.mean(Z, axis=1, keepdims=True)
            std = np.std(Z, axis=1, keepdims=True)
            np.divide(Z - mu, std, out=Z) # per time step
            # Z = (Z - Z.mean()) / Z.std()  # global  
        return Z

class AntitheticGenerator:
    def __init__(self, seed:int = None, moment_matching: bool = True):  # Set to False when n_paths ~ 1e6 for better performance
        self.rng = np.random.default_rng(seed)
        self.moment_matching = moment_matching
    def get_rvs(self, n_steps:int, n_paths:int) -> np.ndarray:
        half_paths = n_paths // 2
        Z_half = self.rng.standard_normal((n_steps, half_paths))
        Z = np.concatenate([Z_half, - Z_half], axis = 1)
        if n_paths % 2:
            Z = np.concatenate([Z, self.rng.standard_normal((n_steps, 1))], axis = 1)
        if n_paths > 1 and n_paths % 2 == 0 and self.moment_matching:
            mu = np.mean(Z, axis=1, keepdims=True)
            std = np.std(Z, axis=1, keepdims=True)
            np.divide(Z - mu, std, out=Z) # per time step
            # Z = (Z - Z.mean()) / Z.std()  # global
        return Z


class MonteCarloEuropeanOption:
    def __init__(self, 
                 S0: float,
                 K: float,
                 T: float,
                 r: float,
                 sigma: float,
                 option_type : Literal['call', 'put', 'digital-call', 'digital-put'],
                 generator: Optional[RandomNumberGenerator] = None,
                 n_steps: int = 252,
                 n_paths: int = 10000,
                 q: float = 0,
                 t: float = 0,
                 tol: float = 1e-12):
        self.S0, self.K, self.T, self.r, self.sigma, self.t, self.q = float(S0), float(K), float(T), float(r), float(sigma), float(t), float(q)
        self.generator = generator if generator is not None else AntitheticGenerator()
        self.n_steps, self.n_paths = int(n_steps), int(n_paths)
        self.option_type = option_type.lower()
        self.tol = float(tol)
        if self.S0 <= 0:
            raise ValueError("Spot price (S0) must be positive.")
        if self.K <= 0:
            raise ValueError("Strike price (K) must be positive.")
        if self.T <= 0:
            raise ValueError("Total time to maturity (T) must be positive.")
        if self.t < 0:
            raise ValueError("Current time (t) cannot be negative.")
        if self.r < 0:
            raise ValueError("Interest rate (r) cannot be negative for standard BSM.")
        if self.sigma < 0:
            raise ValueError("Volatility (sigma) cannot be negative.")
        if self.q < 0:
            raise ValueError("Dividend yield (q) cannot be negative.")
        if self.option_type not in ['call', 'put', 'digital-call', 'digital-put']:
            raise ValueError("Option type must be 'call', 'put', 'digital-call', or 'digital-put'.")
        if self.t > self.T:
            raise ValueError(f"Current time (t={self.t}) cannot be greater than total time to maturity (T={self.T}). Option already expired.")

        self.time_to_maturity = self.T - self.t
        self.variance = self.sigma ** 2 * self.time_to_maturity
        self.discount = np.exp(-self.r * self.time_to_maturity)
        self.dividend_discount = np.exp(-self.q * self.time_to_maturity)

        dt = self.time_to_maturity / self.n_steps
        self.diffusion = self.sigma * np.sqrt(dt)
        self.drift = (self.r - self.q - 0.5 * self.sigma**2) * dt

    def _calculate_d1_d2(self):       
        if self.time_to_maturity < self.tol or self.sigma < self.tol:
            return np.nan, np.nan
        else:
            d1 = (np.log(self.S0 / self.K) + (self.r - self.q) * self.time_to_maturity + 0.5 * self.variance) / np.sqrt(self.variance)
            d2 = d1 - np.sqrt(self.variance)
            return d1, d2

    def _gbmSimulator_exact(self) -> Tuple[np.ndarray, np.ndarray]:
        Z = self.generator.get_rvs(self.n_steps, self.n_paths)
        sum_shocks = np.sum(Z, axis = 0)
        W_T = np.sqrt(self.time_to_maturity / self.n_steps) * sum_shocks
        log_returns = self.drift + self.diffusion * Z
        cumulative_log_returns = np.cumsum(log_returns, axis=0)
        price_paths = np.vstack([np.full(self.n_paths, self.S0), self.S0 * np.exp(cumulative_log_returns)]) # Shape (n_steps + 1, n_paths)
        return price_paths[-1,:], W_T # only terminal prices are needed

    # def _gbmSimulator_exact(self) -> Tuple[np.ndarray, np.ndarray]:
    #     Z = self.generator.get_rvs(self.n_steps, self.n_paths)
    #     sum_shocks = np.sum(Z, axis = 0)
        
    #     price_paths = np.zeros((self.n_steps + 1, self.n_paths))
    #     price_paths[0] = self.S0

    #     for t in range(1, self.n_steps + 1):
    #         price_paths[t] = price_paths[t - 1] * np.exp(self.drift + self.diffusion * Z[t-1,:])
            
    #     return price_paths, sum_shocks


    def BSprice(self) -> float:
        
        if self.time_to_maturity < self.tol:
            if self.option_type == 'call': return max(self.S0 - self.K, 0)
            if self.option_type == 'put': return max(self.K - self.S0, 0)
            if self.option_type == 'digital-call': return 1.0 if self.S0 > self.K else 0.0
            if self.option_type == 'digital-put': return 1.0 if self.S0 < self.K else 0.0
    
        
        if self.sigma < self.tol:
            S_T = self.S0 * np.exp((self.r - self.q) * self.time_to_maturity)
        
            if self.option_type == 'call': return self.discount * max(S_T - self.K, 0)
            if self.option_type == 'put': return self.discount * max(self.K - S_T, 0)
            if self.option_type == 'digital-call': return self.discount * 1.0 if S_T > self.K else 0.0
            if self.option_type == 'digital-put': return self.discount * 1.0 if S_T < self.K else 0.0

        d1, d2 = self._calculate_d1_d2()
    
        if self.option_type == 'call': return (self.S0 * self.dividend_discount * norm.cdf(d1) - self.K * self.discount * norm.cdf(d2))
        if self.option_type == 'put': return (self.K * self.discount * norm.cdf(-d2) - self.S0 * self.dividend_discount * norm.cdf(-d1))
        if self.option_type == 'digital-call': return self.discount * norm.cdf(d2)
        if self.option_type == 'digital-put': return self.discount * norm.cdf(-d2)


    def BSdelta(self) -> float:
        if self.time_to_maturity < self.tol:
            if self.option_type == 'call': return 1.0 if self.S0 > self.K else (0.5 if self.S0 == self.K else 0.0)
            if self.option_type == 'put': return -1.0 if self.S0 < self.K else ( -0.5 if self.S0 == self.K else 0.0)
            if self.option_type in ['digital-call', 'digital-put']: return 0.0    
        
        if self.sigma < self.tol:
            S_T = self.S0 * np.exp((self.r - self.q) * self.time_to_maturity)

            if self.option_type == 'call': 
                return self.dividend_discount if S_T > self.K else (0.5 * self.dividend_discount if S_T == self.K else 0.0)
            if self.option_type == 'put': 
                return -self.dividend_discount if S_T < self.K else ( -0.5 * self.dividend_discount if S_T == self.K else 0.0)
            if self.option_type in ['digital-call', 'digital-put']: return 0.0

        d1, d2 = self._calculate_d1_d2()

        if self.option_type == 'call':
            return self.dividend_discount * norm.cdf(d1)
        if self.option_type == 'put':
            return self.dividend_discount * (norm.cdf(d1) - 1)
        if self.option_type == 'digital-call': 
            return self.discount * norm.pdf(d2) / (self.S0 * np.sqrt(self.variance))
        if self.option_type == 'digital-put': 
            return -self.discount * norm.pdf(d2) / (self.S0 * np.sqrt(self.variance))

    def BSgamma(self) -> float:
        if self.time_to_maturity < self.tol:
            if self.option_type in ['call', 'put']:
                return np.inf if self.S0 == self.K else 0.0
            elif self.option_type in ['digital-call', 'digital-put']:
                return 0.0
        
        if self.sigma < self.tol:
            S_T = self.S0 * np.exp((self.r - self.q) * self.time_to_maturity)
            if self.option_type in ['call', 'put']:
                return np.inf if S_T == self.K else 0.0
            elif self.option_type in ['digital-call', 'digital-put']:
                return 0.0

        d1, d2 = self._calculate_d1_d2()
        
        if self.option_type in ['call', 'put']:
            return self.dividend_discount * norm.pdf(d1) / (self.S0 * np.sqrt(self.variance))
        if self.option_type == 'digital-call':
            return -self.discount * d1 * norm.pdf(d2) / (self.S0**2 * self.variance)
        if self.option_type == 'digital-put':
            return self.discount * d1 * norm.pdf(d2) / (self.S0**2 * self.variance)

    def BSvega(self) -> float:

        if self.time_to_maturity < self.tol:
            return 0.0 
    
        if self.sigma < self.tol:
            S_T = self.S0 * np.exp((self.r - self.q) * self.time_to_maturity)
            if self.option_type in ['call', 'put']:
                return np.inf if S_T == self.K else 0.0
            elif self.option_type in ['digital-call', 'digital-put']:
                return 0.0

        d1, d2 = self._calculate_d1_d2()
        
        if self.option_type in ['call', 'put']: 
            return self.S0 * np.sqrt(self.time_to_maturity) * self.dividend_discount * norm.pdf(d1) 
        if self.option_type == 'digital-call': 
            return -self.discount * d1 * norm.pdf(d2) /  self.sigma
        if self.option_type == 'digital-put': 
            return self.discount * d1 * norm.pdf(d2) / self.sigma

    def BStheta(self) -> float:  # PAUL WILMOTT's sign convension: theta is the pure partial derivative of option price w.r.t. time

        if self.time_to_maturity < self.tol:
            if self.option_type in ['call', 'put']:
                return -np.inf if self.S0 == self.K else 0.0
            elif self.option_type in ['digital-call', 'digital-put']:
                return 0.0

        if self.sigma < self.tol:
            S_T = self.S0 * np.exp((self.r - self.q) * self.time_to_maturity)
            if self.option_type in ['call', 'put']:
                return -np.inf if S_T == self.K else 0.0
            elif self.option_type in ['digital-call', 'digital-put']:
                return 0.0        

        d1, d2 = self._calculate_d1_d2()

        if self.option_type == 'call':
            return (- self.sigma / (2*np.sqrt(self.time_to_maturity)) * self.S0 * self.dividend_discount *  norm.pdf(d1) 
                    + self.q * self.S0 * self.dividend_discount * norm.cdf(d1)
                    - self.r * self.K * self.discount * norm.cdf(d2)
                   )
        if self.option_type == 'put':
            return (- self.sigma / (2*np.sqrt(self.time_to_maturity)) * self.S0 * self.dividend_discount *  norm.pdf(d1) 
                    - self.q * self.S0 * self.dividend_discount * norm.cdf(-d1)
                    + self.r * self.K * self.discount * norm.cdf(-d2)
                   )
        if self.option_type == 'digital-call':
            return (self.r * self.discount * norm.cdf(d2) 
                    + self.discount * norm.pdf(d2) * (d1/(2*self.time_to_maturity) - (self.r - self.q)/np.sqrt(self.variance))
                   )        
        if self.option_type == 'digital-put':
            return (self.r * self.discount * norm.cdf(-d2) 
                    - self.discount * norm.pdf(d2) * (d1/(2*self.time_to_maturity) - (self.r - self.q)/np.sqrt(self.variance))
                   )

    def BSrho(self) -> float:

        if self.time_to_maturity < self.tol:
            return 0.0

        if self.sigma < self.tol:
            S_T = self.S0 * np.exp((self.r - self.q) * self.time_to_maturity)
            if self.option_type == 'call':
                return self.time_to_maturity * self.K * self.discount if S_T > self.K else 0.0
            if self.option_type == 'put':
                return - self.time_to_maturity * self.K * self.discount if S_T < self.K else 0.0    
            elif self.option_type in ['digital-call', 'digital-put']:
                return 0.0  


        d1, d2 = self._calculate_d1_d2()

        if self.option_type == 'call':
            return self.K * self.time_to_maturity * self.discount * norm.cdf(d2)
        if self.option_type == 'put':
            return - self.K * self.time_to_maturity * self.discount * norm.cdf(-d2)
        if self.option_type == 'digital-call':
            return (- self.time_to_maturity * self.discount * norm.cdf(d2)
                    + self.discount * norm.pdf(d2) * np.sqrt(self.time_to_maturity) / self.sigma
                   )       
        if self.option_type == 'digital-put':
            return (- self.time_to_maturity * self.discount * norm.cdf(-d2)
                    - self.discount * norm.pdf(d2) * np.sqrt(self.time_to_maturity) / self.sigma
                   )        
    

    def MCprice(self) -> Tuple[float, float]:
        if self.time_to_maturity < self.tol:
            if self.option_type == 'call': payoff = max(self.S0 - self.K, 0)
            elif self.option_type == 'put': payoff = max(self.K - self.S0, 0)
            elif self.option_type == 'digital-call': payoff = (self.S0 > self.K).astype(float)
            elif self.option_type == 'digital-put': payoff = (self.S0 < self.K).astype(float)

            return payoff, 0.0   # std = 0 (no uncertainty)
    
        
        if self.sigma < self.tol:
            S_T = self.S0 * np.exp((self.r - self.q) * self.time_to_maturity)
        
            if self.option_type == 'call': payoff = max(S_T - self.K, 0)
            elif self.option_type == 'put': payoff = max(self.K - S_T, 0)
            elif self.option_type == 'digital-call': payoff = (S_T > self.K).astype(float)
            elif self.option_type == 'digital-put': payoff = (S_T < self.K).astype(float)

            return payoff * self.discount, 0.0  # std = 0 (determinstic)

        S_T, _ = self._gbmSimulator_exact()

        if self.option_type == 'call':
            payoff = np.maximum(S_T - self.K, 0)
        elif self.option_type == 'put':
            payoff = np.maximum(self.K - S_T, 0)
        elif self.option_type == 'digital-call':
            payoff = (S_T > self.K).astype(float)
            # payoff = np.where(ST > self.K, 1, 0) 
        elif self.option_type == 'digital-put':
            payoff = (S_T < self.K).astype(float)
            # payoff = np.where(ST < self.K, 1, 0)

        price = self.discount * np.mean(payoff)
        std_err = self.discount * np.std(payoff) / np.sqrt(self.n_paths) 
        
        return price, std_err

    def MCdelta_vanilla_pathwise(self) -> Tuple[float, float]:
        if self.time_to_maturity < self.tol:
            if self.option_type == 'call': return (1.0 if self.S0 > self.K else (0.5 if self.S0 == self.K else 0.0)), 0.0
            if self.option_type == 'put': return (-1.0 if self.S0 < self.K else ( -0.5 if self.S0 == self.K else 0.0)), 0.0

        if self.sigma < self.tol:
            S_T = self.S0 * np.exp((self.r - self.q) * self.time_to_maturity)
            if self.option_type == 'call':
                return (self.dividend_discount if S_T > self.K else (0.5 * self.dividend_discount if S_T == self.K else 0.0)), 0.0
            if self.option_type == 'put': 
                return (-self.dividend_discount if S_T < self.K else ( -0.5 * self.dividend_discount if S_T == self.K else 0.0)), 0.0

        S_T, _ = self._gbmSimulator_exact()
        if self.option_type == 'call':
            delta_weights = np.where(S_T > self.K, S_T / self.S0, 0.0)
        elif self.option_type == 'put':
            delta_weights = np.where(S_T < self.K, -S_T / self.S0, 0.0)
        else:
            raise ValueError(f"Unsupported option type: {self.option_type}")
        return (np.mean(delta_weights) * self.discount,
                np.std(delta_weights) * self.discount / np.sqrt(self.n_paths)
               )

    def MCvega_vanilla_pathwise(self) -> Tuple[float, float]:

        if self.option_type not in ['call', 'put']:
            raise ValueError(f"Unsupported option type: {self.option_type}")
            
        if self.time_to_maturity < self.tol:
            return 0.0, 0.0 
    
        if self.sigma < self.tol:
            S_T = self.S0 * np.exp((self.r - self.q) * self.time_to_maturity)
            if self.option_type in ['call', 'put']:
                return (np.inf if S_T == self.K else 0.0), 0.0


        S_T, W_T = self._gbmSimulator_exact()
        vega_weights = np.where((self.option_type == 'call' and S_T > self.K) | (self.option_type == 'put' and S_T < self.K)
                               ,S_T * (W_T - self.sigma * self.time_to_maturity), 0.0)
        return (np.mean(vega_weights) * self.discount,
                np.std(vega_weights) * self.discount / np.sqrt(self.n_paths)
               )

    def MCgamma_vanilla_pathwise_FD(self, bump: float = 0.001) -> Tuple[float, float]:
        
        if self.time_to_maturity < self.tol:
            return (np.inf if self.S0 == self.K else 0.0), 0.0     
        if self.sigma < self.tol:
            return (np.inf if self.S0 * self.dividend_discount == self.K * self.discount else 0.0), 0.0

        h = self.S0 * bump 
        Z = self.generator.get_rvs(self.n_steps, self.n_paths)
        log_S_component = np.cumsum(self.drift + self.diffusion * Z, axis=0)[-1]

        S_T_up = (self.S0 + h) * np.exp(log_S_component)
        S_T_down = (self.S0 - h) * np.exp(log_S_component)
       
        if self.option_type == 'call':
            delta_weights_up = np.where(S_T_up > self.K, S_T_up / (self.S0 + h), 0.0)
            delta_weights_down = np.where(S_T_down > self.K, S_T_down / (self.S0 - h), 0.0)
        elif self.option_type == 'put':
            delta_weights_up = np.where(S_T_up < self.K, -S_T_up / (self.S0 + h), 0.0)
            delta_weights_down = np.where(S_T_down < self.K, -S_T_down / (self.S0 - h), 0.0)
        else:
            raise ValueError(f"Unsupported option type: {self.option_type}")

        gamma_weights = (delta_weights_up - delta_weights_down) / (2 * h)

        return (self.discount * np.mean(gamma_weights),
                 self.discount * np.std(gamma_weights) / np.sqrt(self.n_paths)
               )
    
    def MCdelta_digital_LRM_CV(self) -> Tuple[float, float]:
        if self.option_type not in ['digital-call', 'digital-put']:
            raise ValueError(f"Unsupported option type: {self.option_type}")

        vanilla_type = 'call' if 'call' in self.option_type else 'put'
    
        S_T, W_T = self._gbmSimulator_exact()
        lrm_weight = W_T / (self.S0 * self.sigma * self.time_to_maturity)
        payoff = (S_T > self.K).astype(float) if self.option_type == 'digital-call' else (S_T < self.K).astype(float)
        
        # Control variate: Vanilla delta weight (same LRM weight, different payoff)

        cv_payoff_vanilla = np.maximum(S_T - self.K, 0) if self.option_type == 'digital-call' else np.maximum(self.K - S_T, 0)
        base_params = {
            "K": self.K, "T": self.T, "r": self.r, "sigma": self.sigma,
            "option_type": vanilla_type, # Use the vanilla type here
            "n_steps": self.n_steps, "n_paths": self.n_paths, "q": self.q, 
            "t": self.t, "tol": self.tol,
            "generator": type(self.generator)(seed=getattr(self.generator, 'seed', None))
        }
        vanilla_pricer = MonteCarloEuropeanOption(S0=self.S0, **base_params)
        E_cv = vanilla_pricer.BSdelta()
        
        cov = np.cov(payoff * lrm_weight, cv_payoff_vanilla * lrm_weight)[0, 1]
        var_cv = np.var(cv_payoff_vanilla * lrm_weight, ddof = 1)
        lambda_opt = cov / var_cv if var_cv > 1e-10 else 0.0
        
        delta = np.mean(payoff * lrm_weight - lambda_opt * (cv_payoff_vanilla * lrm_weight - E_cv)) * self.discount
        std_err = np.std(payoff * lrm_weight - lambda_opt * (cv_payoff_vanilla * lrm_weight - E_cv)) * self.discount / np.sqrt(self.n_paths)
        return delta, std_err
    
    def MCvega_digital_LRM_CV(self) -> Tuple[float, float]:
        if self.option_type not in ['digital-call', 'digital-put']:
            raise ValueError(f"Unsupported option type: {self.option_type}")

        vanilla_type = 'call' if 'call' in self.option_type else 'put'

        
        S_T, W_T = self._gbmSimulator_exact()
        lrm_weight = (W_T**2 / self.time_to_maturity - 1) / self.sigma - W_T
        payoff = (S_T > self.K).astype(float) if self.option_type == 'digital-call' else (S_T < self.K).astype(float)
        
        # Control variate: Vanilla vega weight
        cv_payoff_vanilla = np.maximum(S_T - self.K, 0) if self.option_type == 'digital-call' else np.maximum(self.K - S_T, 0)
        base_params = {
            "K": self.K, "T": self.T, "r": self.r, "sigma": self.sigma,
            "option_type": vanilla_type, # Use the vanilla type here
            "n_steps": self.n_steps, "n_paths": self.n_paths, "q": self.q, 
            "t": self.t, "tol": self.tol,
            "generator": type(self.generator)(seed=getattr(self.generator, 'seed', None))
        }
        vanilla_pricer = MonteCarloEuropeanOption(S0=self.S0, **base_params)
        E_cv = vanilla_pricer.BSvega() 
        
        # Optimal lambda
        cov = np.cov(payoff * lrm_weight, cv_payoff_vanilla * lrm_weight)[0, 1]
        var_cv = np.var(cv_payoff_vanilla * lrm_weight, ddof=1)
        lambda_opt = cov / var_cv if var_cv > 1e-10 else 0.0
        
        vega = np.mean(payoff * lrm_weight - lambda_opt * (cv_payoff_vanilla * lrm_weight - E_cv)) * self.discount
        std_err = np.std(payoff * lrm_weight - lambda_opt * (cv_payoff_vanilla * lrm_weight - E_cv)) * self.discount / np.sqrt(self.n_paths)
        return vega, std_err


if __name__ == '__main__':
    
    params = {'S0': 100.0, 'K': 110.0, 'T': 1.0, 'r': 0.03, 'sigma': 0.2, 'q': 0.0}
    
    print("-" * 60)
    print("ANALYTICAL BLACK-SCHOLES RESULTS (VANILLA CALL)")
    print("-" * 60)
    
    bs_pricer = MonteCarloEuropeanOption(option_type = 'call', **params)
    
    print(f"Price: {bs_pricer.BSprice():.5f}")
    print(f"Delta: {bs_pricer.BSdelta():.5f}")
    print(f"Gamma: {bs_pricer.BSgamma():.5f}")
    print(f"Vega:  {bs_pricer.BSvega():.5f}")
    print("\n")
    
    print("-" * 60)
    print("MONTE CARLO RESULTS (VANILLA CALL)")
    print("-" * 60)
    
    mc_params = {**params, "n_paths": 100000}
    mc_pricer = MonteCarloEuropeanOption(option_type='call', **mc_params)
        
    mc_price, mc_price_std_err = mc_pricer.MCprice()
    print(f"MC Price:       {mc_price:.5f} (Std Err: {mc_price_std_err:.5f})")
        
    mc_delta, mc_delta_std_err = mc_pricer.MCdelta_vanilla_pathwise()
    print(f"MC Delta (PW):  {mc_delta:.5f} (Std Err: {mc_delta_std_err:.5f})")
    
    mc_vega, mc_vega_std_err = mc_pricer.MCvega_vanilla_pathwise()
    print(f"MC Vega (PW):   {mc_vega:.5f} (Std Err: {mc_vega_std_err:.5f})")
    
    mc_gamma, mc_gamma_std_err = mc_pricer.MCgamma_vanilla_pathwise_FD()
    print(f"MC Gamma (FD):  {mc_gamma:.5f} (Std Err: {mc_gamma_std_err:.5f})")
    print("\n")
    
    print("-" * 60)
    print("MONTE CARLO RESULTS (DIGITAL CALL)")
    print("-" * 60)
        
    digital_params = {**params, "n_paths": 100000} 
    digital_pricer = MonteCarloEuropeanOption(option_type='digital-call', **digital_params)
    
    bs_digital_price = digital_pricer.BSprice()
    bs_digital_delta = digital_pricer.BSdelta()
    bs_digital_vega = digital_pricer.BSvega()
    print("ANALYTICAL DIGITAL CALL:")
    print(f"BS Digital Price: {bs_digital_price:.5f}")
    print(f"BS Digital Delta: {bs_digital_delta:.5f}")
    print(f"BS Digital Vega: {bs_digital_vega:.5f}")
    print("-" * 60)
    
    mc_delta_d, mc_delta_d_std_err = digital_pricer.MCdelta_digital_LRM_CV()
    print(f"MC Delta (LRM+CV): {mc_delta_d:.5f} (Std Err: {mc_delta_d_std_err:.5f})")
    
    mc_vega_d, mc_vega_d_std_err = digital_pricer.MCvega_digital_LRM_CV()
    print(f"MC Vega (LRM+CV):  {mc_vega_d:.5f} (Std Err: {mc_vega_d_std_err:.5f})")
    print("-" * 60)
        
  
            