import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq # For implied volatility

def black_scholes(Spot, Strike, TimeToMaturity, InterestRate, Volatility, OptionType, quantity='price', Dividend=0):
    """
    black_scholes - Computes Black-Scholes option price or Greeks for European vanilla and digital options.

    This function is vectorized and can handle scalar or array inputs for all parameters.
    It returns a scalar float if the result is a single value, otherwise a NumPy array.

    Args:
        Spot (float or np.ndarray): Current price of the underlying asset.
        Strike (float or np.ndarray): Strike price of the option.
        TimeToMaturity (float or np.ndarray): Time to maturity in years.
        InterestRate (float or np.ndarray): Risk-free interest rate.
        Volatility (float or np.ndarray): Volatility of the underlying asset.
        OptionType (str or list/np.ndarray of str): Type of option: 'call', 'put', 'digicall', or 'digiput'.
        quantity (str, optional): Quantity to compute: 'price', 'delta', 'gamma', 'vega', 'theta', or 'rho'.
                                  Defaults to 'price'.
        Dividend (float or np.ndarray, optional): Dividend yield of the underlying asset. Defaults to 0.

    Returns:
        float or np.ndarray: Computed price or Greek. Returns a float if inputs are scalar,
                             otherwise returns a 1D NumPy array.

    Example Usage (demonstrated in __main__ block of this file):
        # Scalar inputs: black_scholes(100, 100, 1, 0.05, 0.2, 'call', 'price')
        # Vectorized inputs: black_scholes(np.array([100, 100]), 100, 1, 0.05, 0.2, ['call', 'put'], 'price')
    """

    # --- Input Validation and Preprocessing ---
    Spot = np.atleast_1d(Spot).astype(float)
    Strike = np.atleast_1d(Strike).astype(float)
    TimeToMaturity = np.atleast_1d(TimeToMaturity).astype(float)
    InterestRate = np.atleast_1d(InterestRate).astype(float)
    Volatility = np.atleast_1d(Volatility).astype(float)
    Dividend = np.atleast_1d(Dividend).astype(float)

    if not np.all(np.concatenate([Spot, Strike, TimeToMaturity, InterestRate, Volatility, Dividend]) >= 0):
        raise ValueError("Numeric inputs (Spot, Strike, TimeToMaturity, InterestRate, Volatility, Dividend) must be non-negative.")

    valid_option_types = ['call', 'put', 'digicall', 'digiput']
    OptionType = np.atleast_1d(OptionType).astype(str)
    OptionType_lower = np.char.lower(OptionType)

    if not np.all(np.isin(OptionType_lower, valid_option_types)):
        raise ValueError(f"Invalid option type. Use one of: {', '.join(valid_option_types)}.")

    valid_quantities = ['price', 'delta', 'gamma', 'vega', 'theta', 'rho']
    if quantity.lower() not in valid_quantities:
        raise ValueError(f"Invalid quantity. Use one of: {', '.join(valid_quantities)}.")
    quantity = quantity.lower()

    # --- Handle broadcasting / input shape consistency ---
    max_len = max(len(Spot), len(Strike), len(TimeToMaturity),
                  len(InterestRate), len(Volatility), len(Dividend),
                  len(OptionType_lower))

    if len(Spot) == 1 and max_len > 1: Spot = np.full(max_len, Spot[0])
    if len(Strike) == 1 and max_len > 1: Strike = np.full(max_len, Strike[0])
    if len(TimeToMaturity) == 1 and max_len > 1: TimeToMaturity = np.full(max_len, TimeToMaturity[0])
    if len(InterestRate) == 1 and max_len > 1: InterestRate = np.full(max_len, InterestRate[0])
    if len(Volatility) == 1 and max_len > 1: Volatility = np.full(max_len, Volatility[0])
    if len(Dividend) == 1 and max_len > 1: Dividend = np.full(max_len, Dividend[0])
    if len(OptionType_lower) == 1 and max_len > 1: OptionType_lower = np.full(max_len, OptionType_lower[0])

    is_call = (OptionType_lower == 'call')
    is_put = (OptionType_lower == 'put')
    is_digicall = (OptionType_lower == 'digicall')
    is_digiput = (OptionType_lower == 'digiput')

    result = np.zeros(max_len)

    # Avoid zero values in computations for numerical stability
    TimeToMaturity = np.maximum(1e-100, TimeToMaturity)
    Spot = np.maximum(1e-100, Spot)
    Strike = np.maximum(1e-100, Strike)
    Volatility = np.maximum(1e-100, Volatility)

    # Precompute common terms
    Spot_adj = Spot * np.exp(-Dividend * TimeToMaturity)
    discount = np.exp(-InterestRate * TimeToMaturity)
    Strike_adj = Strike * discount
    Sqrt_T = np.sqrt(TimeToMaturity)

    # Avoid division by zero for very small volatility or time to maturity in d1/d2 calculation
    vol_sqrt_t = Volatility * Sqrt_T
    vol_sqrt_t_safe = np.where(vol_sqrt_t < 1e-100, 1e-100, vol_sqrt_t)

    d1 = (np.log(Spot_adj / Strike_adj) + 0.5 * Volatility**2 * TimeToMaturity) / vol_sqrt_t_safe
    d2 = d1 - vol_sqrt_t

    norm_d1 = norm.cdf(d1)
    norm_d2 = norm.cdf(d2)
    normpdf_d1 = norm.pdf(d1)
    normpdf_d2 = norm.pdf(d2)

    # --- Compute result based on requested quantity ---
    if quantity == 'price':
        call_price = Spot_adj * norm_d1 - Strike_adj * norm_d2
        put_price = Strike_adj * norm.cdf(-d2) - Spot_adj * norm.cdf(-d1)
        result[is_call] = call_price[is_call]
        result[is_put] = put_price[is_put]
        digi_call_price = discount * norm_d2
        digi_put_price = discount * (1 - norm_d2)
        result[is_digicall] = digi_call_price[is_digicall]
        result[is_digiput] = digi_put_price[is_digiput]

    elif quantity == 'delta':
        call_delta = np.exp(-Dividend * TimeToMaturity) * norm_d1
        put_delta = np.exp(-Dividend * TimeToMaturity) * (norm_d1 - 1)
        result[is_call] = call_delta[is_call]
        result[is_put] = put_delta[is_put]
        digi_call_delta = discount * normpdf_d2 / vol_sqrt_t_safe / Spot
        digi_put_delta = -digi_call_delta
        result[is_digicall] = digi_call_delta[is_digicall]
        result[is_digiput] = digi_put_delta[is_digiput]

    elif quantity == 'gamma':
        vanilla_gamma = np.exp(-Dividend * TimeToMaturity) * normpdf_d1 / (Spot * vol_sqrt_t_safe)
        result[is_call] = vanilla_gamma[is_call]
        result[is_put] = vanilla_gamma[is_put]
        digi_call_gamma = -discount * d1 * normpdf_d2 / (Spot * vol_sqrt_t_safe)**2
        digi_put_gamma = -digi_call_gamma
        result[is_digicall] = digi_call_gamma[is_digicall]
        result[is_digiput] = digi_put_gamma[is_digiput]

    elif quantity == 'vega':
        vanilla_vega = Spot_adj * normpdf_d1 * Sqrt_T
        result[is_call] = vanilla_vega[is_call]
        result[is_put] = vanilla_vega[is_put]
        digi_call_vega = -discount * d1 * normpdf_d2 / Volatility
        digi_put_vega = -digi_call_vega
        result[is_digicall] = digi_call_vega[is_digicall]
        result[is_digiput] = digi_put_vega[is_digiput]

    elif quantity == 'theta':
        common_vanilla_theta = -Volatility * Spot_adj * normpdf_d1 / (2 * Sqrt_T)
        call_theta = common_vanilla_theta + Dividend * Spot_adj * norm_d1 - InterestRate * Strike_adj * norm_d2
        put_theta = common_vanilla_theta + Dividend * Spot_adj * (norm_d1 - 1) - InterestRate * Strike_adj * (norm_d2 - 1)
        result[is_call] = call_theta[is_call]
        result[is_put] = put_theta[is_put]
        
        ttm_safe = np.where(TimeToMaturity < 1e-100, 1e-100, TimeToMaturity)
        digi_call_theta = InterestRate * discount * norm_d2 + discount * normpdf_d2 * (d1 / (2 * ttm_safe) - (InterestRate - Dividend) / vol_sqrt_t_safe)
        digi_put_theta = InterestRate * discount - digi_call_theta
        result[is_digicall] = digi_call_theta[is_digicall]
        result[is_digiput] = digi_put_theta[is_digiput]

    elif quantity == 'rho':
        call_rho = Strike_adj * TimeToMaturity * norm_d2
        put_rho = Strike_adj * TimeToMaturity * (norm_d2 - 1)
        result[is_call] = call_rho[is_call]
        result[is_put] = put_rho[is_put]
        digi_call_rho = -TimeToMaturity * discount * norm_d2 + Sqrt_T * discount * normpdf_d2 / Volatility
        digi_put_rho = -TimeToMaturity * discount - digi_call_rho
        result[is_digicall] = digi_call_rho[is_digicall]
        result[is_digiput] = digi_put_rho[is_digiput]

    # --- Final output formatting ---
    return result.item() if result.size == 1 else result


def implied_volatility(MarketPrice, Spot, Strike, TimeToMaturity, InterestRate, OptionType, Dividend=0, Tolerance=1e-10):
    """
    implied_volatility - Computes implied volatility using Brent's method (from SciPy).

    This function finds the volatility that makes the Black-Scholes price
    equal to the given market price. It handles scalar inputs only.

    Args:
        MarketPrice (float): Current market price of the option.
        Spot (float): Current price of the underlying asset.
        Strike (float): Strike price of the option.
        TimeToMaturity (float): Time to maturity in years.
        InterestRate (float): Risk-free interest rate.
        OptionType (str): Type of option: 'call' or 'put'.
        Dividend (float, optional): Dividend yield of the underlying asset. Defaults to 0.
        Tolerance (float, optional): Tolerance for convergence. Defaults to 1e-10.

    Returns:
        float: The estimated implied volatility.

    Raises:
        ValueError: If inputs are invalid, arbitrage bounds are violated, or root finding fails.
        RuntimeWarning: If implied volatility is unusually low (<0.1%) or high (>200%).
    """

    # --- Input Validation ---
    if not all(x >= 0 for x in [MarketPrice, Spot, Strike, TimeToMaturity, InterestRate]):
        raise ValueError('MarketPrice, Spot, Strike, TimeToMaturity, and InterestRate must be non-negative.')
    if OptionType.lower() not in ['call', 'put']:
        raise ValueError('Invalid OptionType. Use "call" or "put".')
    
    if not all(np.isscalar(x) for x in [MarketPrice, Spot, Strike, TimeToMaturity, InterestRate, Dividend, Tolerance]):
        raise ValueError("implied_volatility function expects scalar inputs for all parameters except OptionType.")

    # Check arbitrage bounds
    discount_factor = np.exp(-InterestRate * TimeToMaturity)
    if OptionType.lower() == 'call':
        lower_bound = max(Spot - Strike * discount_factor, 0)
        upper_bound = Spot
    else: # put
        lower_bound = max(Strike * discount_factor - Spot, 0)
        upper_bound = Strike * discount_factor
    
    if MarketPrice < lower_bound - 1e-9 or MarketPrice > upper_bound + 1e-9: # Add small epsilon for float comparison
        raise ValueError(f'MarketPrice ({MarketPrice:.4f}) violates arbitrage bounds [{lower_bound:.4f}, {upper_bound:.4f}].')

    # Black-Scholes price as function of Volatility
    # It calls the black_scholes function defined within this same file.
    bs_price_func = lambda volatility: black_scholes(Spot, Strike, TimeToMaturity, InterestRate, volatility, OptionType, 'price', Dividend)

    # Objective function (difference between MarketPrice and BS price)
    objective_func = lambda volatility: bs_price_func(volatility) - MarketPrice

    # Initial interval for volatility (common bounds in finance)
    vol_l = 1e-5  # Lower bound (0.001%)
    vol_u = 5.0   # Upper bound (500%) - wide enough for most practical cases

    # Ensure the function changes sign over the interval
    obj_l = objective_func(vol_l)
    obj_u = objective_func(vol_u)

    if obj_l * obj_u >= 0:
        if abs(obj_l) < Tolerance:
            return vol_l
        if abs(obj_u) < Tolerance:
            return vol_u
        raise ValueError('Failed to find root. Objective function has same sign at initial interval endpoints. Adjust bounds or check inputs.')

    # Use Brent's method from scipy.optimize to find the root
    try:
        result_vol = brentq(objective_func, vol_l, vol_u, xtol=Tolerance)
    except ValueError as e:
        raise ValueError(f"SciPy brentq failed to find a root: {e}. Check initial interval or inputs.")

    # Check for unrealistic implied volatility (warnings, not errors)
    if result_vol < 1e-3: # 0.1%
        import warnings
        warnings.warn(f'Implied volatility is Unusually Low ({result_vol:.4f}). Check inputs or model assumptions.', RuntimeWarning)
    if result_vol > 2.0: # 200%
        import warnings
        warnings.warn(f'Implied volatility is Unusually High ({result_vol:.4f}). Check inputs or model assumptions.', RuntimeWarning)

    return result_vol

if __name__ == '__main__':
    # --- Basic Scalar Black-Scholes Price Example ---
    print("--- Scalar Black-Scholes Price ---")
    spot = 100; strike = 100; time_to_maturity = 1; interest_rate = 0.05; volatility = 0.2
    option_type = 'call'
    price = black_scholes(spot, strike, time_to_maturity, interest_rate, volatility, option_type, 'price')
    print(f"Call Price (scalar output): {price:.8f}")
    print("-" * 30)

    # --- Basic Vectorized Black-Scholes Price Example ---
    print("\n--- Vectorized Black-Scholes Price (Call & Put) ---")
    spot_vec = np.array([100, 100])
    strike_vec = np.array([100, 100])
    ttm_vec = np.array([1, 1])
    ir_vec = np.array([0.05, 0.05])
    vol_vec = np.array([0.2, 0.2])
    option_types_vec = ['call', 'put']
    
    prices_vec = black_scholes(spot_vec, strike_vec, ttm_vec, ir_vec, vol_vec, option_types_vec, 'price')
    print(f"Prices for {option_types_vec}: {prices_vec}")
    print("-" * 30)

    # --- Implied Volatility Example ---
    print("\n--- Implied Volatility Calculation ---")
    known_price_call = black_scholes(spot, strike, time_to_maturity, interest_rate, volatility, 'call', 'price')
    print(f"Known Call Price for Vol={volatility}: {known_price_call:.8f}")

    try:
        implied_vol_call = implied_volatility(known_price_call, spot, strike, time_to_maturity, interest_rate, 'call', Dividend=0, Tolerance=1e-10)
        print(f"Calculated Implied Volatility: {implied_vol_call:.8f}")
        print(f"Difference from original: {abs(implied_vol_call - volatility):.10f}")
    except (ValueError, RuntimeWarning) as e:
        print(f"Error/Warning for Implied Volatility: {e}")
    print("-" * 30)