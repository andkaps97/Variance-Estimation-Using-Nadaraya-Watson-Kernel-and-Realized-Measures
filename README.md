# Variance Estimation Using Nadaraya-Watson Kernel and Realized Measures

This project implements various variance estimation methods for financial time series data, including Nadaraya-Watson kernel estimation for daily data and realized variance measures for intraday data.

## Overview

The project estimates daily variances using the Nadaraya-Watson kernel estimator with optimal bandwidth selection, and compares these estimates with intraday variance measures including:
- Realized Variance (RV)
- Realized Kernel (RK)
- Bipower Variation (BV)

## Features

- **Nadaraya-Watson Kernel Estimation**: Non-parametric variance estimation using Epanechnikov kernel
- **Optimal Bandwidth Selection**: Cross-validation approach to find the best bandwidth parameter
- **Realized Variance**: Standard measure using squared intraday returns
- **Realized Kernel**: Bias-corrected variance estimator using Parzen kernel
- **Bipower Variation**: Jump-robust variance estimator
- **Visualization**: Comparative plots of different variance estimates

## Requirements## Data Requirements

The project expects two data files:
- `data/XOM_1023.csv`: Daily stock price data with columns including 'Date' and 'Adj Close'
- `data/gc1lpheleesmghui.csv.gz`: High-frequency intraday data with columns 'DATE', 'TIME_M', 'PRICE', 'SYM_SUFFIX', 'EX'

## Key Functions

### Data Processing
- `InsertData()`: Load daily stock data
- `getSqr_LReturns()`: Compute squared log returns
- `clean_data()`: Process and resample intraday data to 5-minute frequency

### Kernel Estimation
- `epanechnikov()`: Epanechnikov kernel function
- `nw_variance_estimator()`: Nadaraya-Watson variance estimator
- `find_best_bandwidth()`: Optimal bandwidth selection via cross-validation

### Realized Measures
- `compute_realized_variance()`: Calculate daily realized variance from intraday returns
- `realized_kernel_estimator()`: Compute realized kernel estimates using Parzen kernel
- `compute_bipower_variation()`: Calculate bipower variation estimates

### Visualization
- `plot_realized_variance()`: Compare RV with kernel estimates
- `plot_RK()`: Plot realized kernel estimates
- `plot_bv()`: Compare bipower variation with kernel estimates
- `plot_intraday_price()`: Display intraday price movements

## Usage

Run the main script:

```python
python main.py


The script will:
Load and process daily stock data
Find optimal bandwidth for Nadaraya-Watson estimation
Compute daily variance estimates using the kernel method
Process intraday data and calculate realized measures
Generate comparative plots
Identify days with largest differences between estimates
Output
The program outputs:
Optimal bandwidth parameter
Comparative plots of different variance estimates
Dates with maximum differences between estimators
Intraday price plots for significant dates
Methodology
The Nadaraya-Watson estimator uses:
Epanechnikov kernel for weight calculation
Cross-validation for bandwidth selection
Time series structure preservation
Realized measures use:
5-minute return intervals
Parzen kernel for realized kernel estimation
Trading hours from 9:30 AM to 4:00 PM

