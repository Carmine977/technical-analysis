import numpy as np
import pandas as pd

def compute_daily_returns(df, var='Close'):
    """ Compute daily returns. """   
    daily_return = (df[var] / df[var].shift(1).values) - 1
    return daily_return
    

def compute_sharpe_ratio(df, daily_rf=0, samples_per_year=252):
    """ Compute sharpe ratio. """   
    daily_returns = compute_daily_returns(df)
    sharpe_ratio = ((daily_returns - daily_rf).mean()/daily_returns.std()) * np.sqrt(samples_per_year)
    return sharpe_ratio
    
    
def MAV(df, var='Close', window=20):
    """ Compute moving average. """
    mav = df[var].rolling(window=window).mean()
    mav.name = 'MAV{}'.format(window)
    return mav
    
    
def momentum(df, var='Close', window = 14):
    """ Compute momentum indicator as the rate of change in the price and is expressed as a percentage. """
    mom = df[var]/df[var].shift(periods=window)-1
    return mom
    
    
def ATR(df, window=14):
    """ Compute the Average True Range Indicator. """
    rng1 = df['High']-df['Low']
    rng2 = np.abs(df['High']-df['Close'].shift(1))
    rng3 = np.abs(df['Low']-df['Close'].shift(1))
    rng = pd.concat([rng1, rng2, rng3], axis=1)
    tr = rng.max(axis=1)
    
    ind = tr.iloc[:window].mean()  # atr initial value
    atr = [np.nan]*(window-1) # fill initial part of the vector
    atr.append(ind) 
    # compute exponential moving average
    for i in range(window, len(tr)):
        ind = (ind*(window-1)+tr[i])/window
        atr.append(ind)

    atr = pd.Series(atr, index=tr.index)
    return atr
    
 
def stochastic(df, mode='full', window=[14,3,3]):
    """ Compute the Stochastic Oscillator. """
    roll = df.rolling(window=window[0])
    K = 100.0*(df['Close']-roll['Low'].min())/(roll['High'].max()-roll['Low'].min())
    if mode == 'fast':
        D = K.rolling(window=window[1]).mean()
    elif mode == 'slow':
        K = K.rolling(window=window[1]).mean()
        D = K.rolling(window=window[1]).mean()
    elif mode == 'full':
        K = K.rolling(window=window[1]).mean()
        D = K.rolling(window=window[2]).mean()
    return K, D
    
    
def RSI(df, var='Close', window=14):
    """ Compute the Relative Strength Index. """
    diff = df[var].diff()
    bull = diff.apply(lambda x: x if x>=0 else 0) # Compute price rise
    bear = diff.apply(lambda x: abs(x) if x<0 else 0) # Compute price drop
    SMA_bull = bull.rolling(window=window).mean()
    SMA_bear = bear.rolling(window=window).mean()
    RS = SMA_bull/SMA_bear
    rsi = 100.0-(100.0/(1+RS))
    return rsi
    
    
