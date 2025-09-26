"""

Purpose:
    Estimating daily variances and the optimal bandwidth by using Nadaraya-Watson for daily data,
    and estimating variances by Realized Variance, Realized Kernel and Bipower Vaarience for intraday data.
"""

###############################################
###imports
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize
import matplotlib.pyplot as plt


###############################################
def InsertData():
    '''
    Purpose:
        Read the data set
    Inputs:

    Return  Value:
        df    dataframe
    '''
    df = pd.read_csv("data/XOM_1023.csv")
    return df


##################################################
def getSqr_LReturns(df):
    """
    Purpose:
        Obtain the squared log returns of a stock

    Inputs:
        stock        string, name.csv of a stock


    Return value:
        dfR2           n x 1, dataframe of squared log returns of a stock
        Date            n x 1, dataframe of timepoints
    """
    df['logr'] = 100 * np.log(df['Adj Close']).diff()
    df['log_sq'] = df['logr'] ** 2
    df.dropna(inplace=True)
    df = df.set_index('Date')

    return df['log_sq']


############################################
def epanechnikov(u):
    '''
    Purpose:
        Compute the Epanechnikov Kernel

    Inputs:
        u       float

    Return value:
        Epanechnikov Kernel estimator
    '''
    return np.where(np.abs(u) <= 1, 0.75 * (1 - u ** 2), 0)


###########################################
def nw_variance_estimator(t, data, bandwidth):
    '''
    Purpose:
        Estimate Nadarata-Watson Kernel

    Inputs:
        t             yy/mm/dd dates
        data         string, name.csv of a stock
        bandwidth    centre of the bin

    Return value:
        local variance    n x 1, dataframe of the estimated variance
    '''
    # Use all available data up to day t
    n = len(data)
    weights = epanechnikov((t - np.arange(n)) / bandwidth)
    normalized_weights = weights / np.sum(weights)
    local_variance = np.sum(normalized_weights * (data - np.sum(normalized_weights * data)) ** 2)

    return local_variance


###############################################
def nadaraya_watson_except_i(X, Y, h, i):
    X = np.array(X)
    Y = np.array(Y)

    xi = X[i]
    xi_as_date = np.datetime64(xi)
    differences_in_days = (xi_as_date - np.delete(X, i)).astype('timedelta64[D]').astype(int)

    weights = epanechnikov(differences_in_days / h)

    denominator = np.sum(weights)
    if denominator == 0:
        return Y[i]  # return the same y value if all weights are zero to prevent division by zero

    return np.sum(weights * np.delete(Y, i)) / denominator


##################################################
def msfecv(X, Y, h):
    """
     Purpose:
          compare msfecv values

     Inputs:
         X         n x 1, array of the dates of stock data
         Y          n x 1, array of the values of stock data
         h          float, number of lags



     Return value:
         mean of errors  float

     """
    X = np.array(X)
    Y = np.array(Y)

    errors = np.array([(Y[i] - nadaraya_watson_except_i(X, Y, h, i)) ** 2 for i in range(len(X))])

    return np.mean(errors)


#####################################################
def find_best_bandwidth(X, Y, bandwidths):
    """
         Purpose:
                      Find the bandwidth that minimizes the MSFECV
         Inputs:
             X         n x 1, array of the dates of stock data
             Y          n x 1, array of the values of stock data
             bandwidths          float, number of lags



         Return value:
                best_h           float ,the best bandwidth
    """

    errors = [msfecv(X, Y, h) for h in bandwidths]
    best_h = bandwidths[np.argmin(errors)]
    return best_h


############################################
def clean_data(asTime=['9:30', '16:00'], iN=20000000000, sFreq='5Min'):
    """
           Purpose:
                cleaning and preparing data

           Inputs:
               asTime        string, selecting time
               iN            integer
               sFreq         string  selecting the frequence


           Return value:
               src           n x 1, dataframe of the 5min data

           """
    # Read the data
    sF = 'data/gc1lpheleesmghui.csv.gz'

    df = pd.read_csv(sF, nrows=iN, parse_dates=[['DATE', 'TIME_M']])
    df.set_index(['DATE_TIME_M'], inplace=True)

    # Drop items with sym_suffix filled in
    vI = [isinstance(sSuf, float) for sSuf in df['SYM_SUFFIX']]
    df = df[vI]

    # Select single exchange
    srC = df['EX'].value_counts()
    vI = df['EX'] == srC.index[0]
    df = df[vI]

    # Select timeslot 9.30 - 16.00
    vT = pd.to_datetime(asTime, format='%H:%M').time
    vI = (df.index.time >= vT[0]) & (df.index.time <= vT[1])
    df = df[vI]

    # Resample to get 'last' or 'closing' prices for the specified frequency
    srC = df['PRICE'].resample(sFreq).last().dropna()

    return srC


####################################################
def compute_realized_variance(srC):
    """
    Purpose:
        compute realised variance

    Inputs:
        srC           n x 1, dataframe of the 5min data


    Return value:
        rv_daily          float,daily realised variance

    """
    # Compute 5-minute returns
    srR = 100 * srC.diff() / srC
    srR.dropna(inplace=True)

    # Square the returns
    srSquaredR = srR ** 2

    # Sum up the squared returns for each day to get the RV for that day
    rv_daily = srSquaredR.resample('D').sum()

    return rv_daily


#################################################
def plot_realized_variance(rv, ke):
    """
    Purpose:
        plot realised variance with kernel  estimates

    Inputs:
        rv          n x 1, dataframe of the realised variance estimates
        ke          n x 1, dataframe of the realised variance estimates


    Return value:
        plot of Realized Variance estimator


    """
    rv.index = pd.to_datetime(rv.index)
    ke.index = pd.to_datetime(ke.index)
    start_date = pd.Timestamp("2023-09-01")
    end_date = pd.Timestamp("2023-09-20")
    rv = rv[rv != 0]

    # Filter dates for plots
    rv = rv[start_date:end_date]
    kernel = ke[start_date:end_date]

    plt.figure(figsize=(14, 7))
    plt.plot(rv, label='RV estimates', color='midnightblue')
    plt.plot(kernel, label='NW-Kernel estimates', linestyle='--', color='goldenrod')
    plt.legend()
    plt.title('Realized Variance vs NW-Kernel Estimates', color='crimson', fontsize=14)
    plt.show()


##################################################
def plot_RK(rk, ke):
    """
    Purpose:
        plot realised kernel  estimates

    Inputs:
        rk          n x 1, dataframe of the realised kernel estimates


    Return value:
        plot of Realized Kernel estimator


    """
    rk.index = pd.to_datetime(rk.index)
    ke.index = pd.to_datetime(ke.index)
    # start_date = pd.Timestamp("2023-09-01")
    # end_date = pd.Timestamp("2023-09-20")
    # rk = rk[rk != 0]
    rk.dropna(inplace=True)

    # Filter dates for plots
    # rk = rk[start_date:end_date]
    # kernel = ke[start_date:end_date]

    plt.figure(figsize=(14, 7))
    plt.plot(rk, label='RK estimates', color='midnightblue')

    plt.legend()
    plt.title('Realized Kernel', color='crimson', fontsize=14)
    plt.show()


################################################
def compute_bipower_variation(srR):
    """
    Purpose:
        compute bipower variation

    Inputs:
        srR         n x 1, dataframe of the 5 min stock data



    Return value:
         bv_daily      n x 1, dataframe of the daily bipower varation estimates

        """
    sr = 100 * srR.diff() / srR
    sr.dropna(inplace=True)
    bv_daily = (sr.abs() * sr.abs().shift(-1)).resample('D').sum()

    return bv_daily.dropna()


################################################
def plot_bv(bv, ke):
    """
    Purpose:
         plot bipower variation with kernel  estimates

    Inputs:
        bv         n x 1, dataframe of the bipower variation estimates
        ke          n x 1, dataframe of the realised variance estimates


    Return value:
        plot of Bipower Variance estimator


    """
    bv.index = pd.to_datetime(bv.index)
    ke.index = pd.to_datetime(ke.index)
    start_date = pd.Timestamp("2023-09-01")
    end_date = pd.Timestamp("2023-09-20")
    bv = bv[bv != 0]

    # Filter dates for plots
    bv = bv[start_date:end_date]
    kernel = ke[start_date:end_date]

    plt.figure(figsize=(14, 5))
    plt.plot(bv, label='BV estimates', color='midnightblue')
    plt.plot(kernel, label='NW-Kernel estimates', linestyle='--', color='goldenrod')
    plt.legend()
    plt.title('BV vs NW-Kernel Estimates', color='crimson', fontsize=14)
    plt.show()


###############################################
def plot_intraday_price(cdf, date):
    """
    Purpose:
        plot intraday price  12 of septemper

    Inputs:
        cdf         n x 1, dataframe of the 5 min stock data
        date         n x 1, dataframe of the dates



     Return value:
         plot of intraday price


    """
    day_data = cdf[str(date.date())]  # Ensure date is in string format and only the date part
    plt.figure(figsize=(12, 6))
    plt.plot(day_data, label='Intraday Price on ' + str(date.date()), color='midnightblue')
    plt.title('Intraday Price on ' + str(date.date()), color='crimson', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#############################################
def gamma(returns, h):
    """Calculate the autocovariance at lag h"""
    if h > 0:
        return np.mean(returns[h:] * returns[:-h])
    elif h == 0:
        return np.var(returns)
    else:  # Negative lags
        return np.mean(returns[:h] * returns[-h:])


###############################################
def parzen_kernel(x):
    """Parzen Kernel function - you can adjust based on your needs."""
    if abs(x) <= 0.5:
        return 1 - 6 * x ** 2 + 6 * abs(x) ** 3
    elif 0.5 < abs(x) <= 1:
        return 2 * (1 - abs(x)) ** 3
    else:
        return 0


##############################################
def realized_kernel_estimator(r, H, kernel_func):
    """
    Purpose:
        Calculate the Realized Kernel (RK) estimate for integrated variance.

    Inputs:
        r         n x 1, dataframe of the 5 min stock data
        H               integer, number of lags



    Return value:
       rk         float reliased kernel value of each day

     """
    returns = 100 * r.diff() / r
    returns.dropna(inplace=True)
    # gamma_0(r)
    gamma_0 = np.var(returns)

    # Sum over h=1 to H of kernel weighted autocovariances
    sum_k_gamma = 0.0
    for h in range(1, H + 1):
        weight = kernel_func(float(h) / H)  # Adjusting kernel function call
        sum_k_gamma += weight * (gamma(returns, h) + gamma(returns, -h))

    RK = gamma_0 + sum_k_gamma

    return RK


########################################################
###main()
def main():
    data = InsertData()
    df1 = getSqr_LReturns(data)
    df1.index = pd.to_datetime(df1.index)

    bandwidths = np.linspace(2, 100, 40)

    bandwidth = find_best_bandwidth(df1.index.values, df1.values, bandwidths)

    print(f"Best bandwidth is: {bandwidth}")

    daily_variances = [nw_variance_estimator(t, df1.values, bandwidth) for t in range(len(df1))]
    dv = pd.Series(daily_variances, index=df1.index)

    #####intraday part
    ###realized variance
    cdf = clean_data()
    rv = compute_realized_variance(cdf)

    plot_realized_variance(rv, dv)

    ###bipower variation
    bv = compute_bipower_variation(cdf)
    plot_bv(bv, dv)

    # Find the day with the largest difference between BV and Kernel Estimates
    diff = (bv - rv).abs()
    max_diff_date = diff.idxmax()
    print(f"Largest difference is on: {max_diff_date}")

    plot_intraday_price(cdf, max_diff_date)

    ###realized kernel
    h = round(bandwidth)
    rk = cdf.resample('D').apply(realized_kernel_estimator, H=h, kernel_func=parzen_kernel)
    plot_RK(rk, dv)
    # Compute the differences
    diff_rv = (rk - rv).abs()
    diff_bv = (rk - bv).abs()
    # Identify the day with the max difference
    max_diff_date_rv = diff_rv.idxmax()
    max_diff_date_bv = diff_bv.idxmax()

    print(f"Largest difference with RV is on: {max_diff_date_rv}")
    print(f"Largest difference with BV is on: {max_diff_date_bv}")

    ### using different bandwidth for comparison
    h = 2  # using bandwitdh as 2 days
    start_date = pd.Timestamp("2023-09-01")
    end_date = pd.Timestamp("2023-09-20")
    dfsten = df1[start_date:end_date]
    daily_va2 = [nw_variance_estimator(t, dfsten.values, h) for t in range(len(dfsten))]
    dv2 = pd.Series(daily_va2, index=dfsten.index)

    plot_realized_variance(rv, dv2)

    plot_bv(bv, dv2)


if __name__ == '__main__':
    main()