## Quantitative Finance Toolkit in Python

This repository contains a collection of Python modules for derivatives pricing, risk management, and financial modeling.

### Key Features

* **Analytical Pricers:** Vectorized Black-Scholes formulas for European vanilla/digital options and all major Greeks.
* **Monte Carlo Engine:** Advanced MC pricer with variance reduction techniques (Antithetic Variates, Control Variates) and a multi-asset VaR/CVaR simulation model.
* **Advanced Models (FFT):** High-performance option pricer using the Carr-Madan (FFT) method for models including Heston, Bates, Merton, and Variance Gamma.
* **Model Calibration:** Includes a full calibration routine to fit advanced model parameters to market prices.

### Installation

Clone the repository and install the required packages:

```bash
git clone [https://github.com/sMohanta94/Quantitative-Finance-Toolkit.git](https://github.com/sMohanta94/Quantitative-Finance-Toolkit.git)
cd Quantitative-Finance-Toolkit
pip install -r requirements.txt
