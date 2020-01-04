import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import date2num
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def load_stock_price(stock, archive, date_start=None, date_end=None, folder='data'):
    """ Load stock price from pickle file and return it as pandas DataFrame """

    with open(os.path.join(folder, '{}.pkl'.format(archive)),'rb') as f:
        data = pickle.load(f)

    if date_start is not None:
        T_start = datetime.strptime(date_start, '%Y-%m-%d')
    else:
        T_start = data.index[0]

    if date_end is not None:
        T_end = datetime.strptime(date_end, '%Y-%m-%d')
    else:
        T_end = data.index[-1]

    T_ix = (data.index >= T_start) & (data.index <= T_end)

    openp = data.loc[T_ix, 'Open'][stock].values
    highp = data.loc[T_ix, 'High'][stock].values
    lowp = data.loc[T_ix, 'Low'][stock].values
    closep = data.loc[T_ix, 'Close'][stock].values
    adj_closep = data.loc[T_ix, 'Adj Close'][stock].values
    volumep = data.loc[T_ix, 'Volume'][stock].values

    prices = pd.DataFrame(np.array([openp, highp, lowp, closep, adj_closep, volumep]).T, index=data.index[T_ix],
                          columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    return prices


def get_ohlc(df):
    """ Prepare data in the format required by old matplotlib.finance.candlestick_ohlc package """
    # Data Preparation
    date = [date2num(t) for t in df.index]
    closep = df['Close'].values
    highp = df['High'].values
    lowp = df['Low'].values
    openp = df['Open'].values
    volume = df['Volume'].values

    x = 0
    y = len(date)
    ohlc = []
    while x < y:
        append_me = date[x], openp[x], highp[x], lowp[x], closep[x], volume[x]
        ohlc.append(append_me)
        x += 1

    return ohlc


def candlestick_ohlc(ax, quotes, width=0.2, colorup='k', colordown='r',
                     alpha=1.0):
    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    quotes : sequence of (time, open, high, low, close, ...) sequences
        As long as the first 5 elements are these values,
        the record can be as long as you want (e.g., it may store volume).

        time must be in float days format - see date2num

    width : float
        fraction of a day for the rectangle width
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level

    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added

    """
    return _candlestick(ax, quotes, width=width, colorup=colorup,
                        colordown=colordown,
                        alpha=alpha, ochl=False)


def _candlestick(ax, quotes, width=0.2, colorup='k', colordown='r',
                 alpha=1.0, ochl=True):
    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    quotes : sequence of quote sequences
        data to plot.  time must be in float date format - see date2num
        (time, open, high, low, close, ...) vs
        (time, open, close, high, low, ...)
        set by `ochl`
    width : float
        fraction of a day for the rectangle width
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level
    ochl: bool
        argument to select between ochl and ohlc ordering of quotes

    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added

    """

    OFFSET = width / 2.0

    lines = []
    patches = []
    for q in quotes:
        if ochl:
            t, open, close, high, low = q[:5]
        else:
            t, open, high, low, close = q[:5]

        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close

        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )

        rect = Rectangle(
            xy=(t - OFFSET, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        rect.set_alpha(alpha)

        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
    ax.autoscale_view()

    return lines, patches


def weekday_candlestick(stock_data, ax, stock_name, fmt='%b %d', freq=7, **kwargs):
    """ Wrapper function for matplotlib.finance.candlestick_ohlc that artificially spaces data to avoid gaps from weekends.
        Takes data in the format required by matplotlib.finance.candlestick_ohlc """
    # Convert data to numpy array
    ohlc_data = get_ohlc(stock_data)
    ohlc_data_arr = np.array(ohlc_data)
    ohlc_data_arr2 = np.hstack(
        [np.arange(len(ohlc_data_arr))[:,np.newaxis], ohlc_data_arr[:,1:]])
    ndays = ohlc_data_arr2[:,0]  # array([0, 1, 2, ... n-2, n-1, n])

    # Convert matplotlib date numbers to strings based on `fmt`
    dates = mdates.num2date(ohlc_data_arr[:, 0])
    date_strings = []
    for date in dates:
        date_strings.append(date.strftime(fmt))

    # Plot candlestick chart
    candlestick_ohlc(ax, ohlc_data_arr2, colorup='#77d879', colordown='#db3f3f', **kwargs)

    # Format x axis
    ax.set_xticks(ndays[::freq])
    ax.set_xticklabels(date_strings[::freq], rotation=90, ha='center')
    #ax.set_xticklabels([dt.strftime("%Y-%m-%d") for dt in stock_data.index], rotation=0)
    #fig.autofmt_xdate()
    ax.set_xlim(ndays.min(), ndays.max()+1)
    ax.grid(alpha=0.2)
    plt.title(stock_name)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()
    
    
class trading_chart():
    
    def __init__(self, figsize=(16, 8), freq=5, stock_name=""):
        # Start with one
        self.figsize = figsize
        self.trace = list()
        self.n_axis = 0
        self.freq = freq
        self.stock = stock_name
        self.colorup = '#77d879'
        self.colordown = '#db3f3f'
        self._cndlstck_called = False
        
    def add_subplot(self):
        self.n_axis += 1
        
    def candlestick(self, df, width=0.5):
        self.df = df
        self.width = width
        self._cndlstck_called = True
        self.index = df.index
        
    def mav(self, df, var='Close', window=20):
        # compute moving average
        mav = df[var].rolling(window=window).mean()
        mav.name = 'MAV{}'.format(window)
        self.trace.append((self.n_axis, mav.values, 'MAV{}'.format(window)))       
        self.values = mav
        self.index = mav.index
        
    def hline(self, val):
        y = pd.Series([val] * len(self.index))
        self.trace.append((self.n_axis, y, '{}'.format(val)))
        self.values = y     
    
    def _get_color(self, pclose, popen):
        if pclose >= popen:
            return self.colorup
        else:
            return self.colordown
    
    def volumes(self, df):
        self.trace.append((self.n_axis, df['Volume'].values, 'Volume'))
        self.colors = df.apply(lambda x: self._get_color(x['Close'], x['Open']), axis=1)
        self.values = df['Volume']
        self.index = df.index
        
    def momentum(self, df, var='Close', window = 14):
        momentum = df[var]/df[var].shift(periods=window)-1
        self.trace.append((self.n_axis, momentum.values, 'Momentum'))
        self.values = momentum
        self.index = df.index
        
    def ATR(self, df, window=14):
        """ Average True Range """
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
        self.trace.append((self.n_axis, atr.values, 'ATR'))
        self.values = atr
        self.index = df.index
        
    def stoch(self, df, mode='full', window=[14,3,3]):
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
        self.trace.append((self.n_axis, K.values, 'stoch %K'))
        self.trace.append((self.n_axis, D.values, 'stoch %D'))
        self.values = np.array([K, D]).T
        self.index = df.index
        
    def RSI(self, df, var='Close', window=14):
        """ Compute the Relative Strength Index"""
        diff = df[var].diff()
        bull = diff.apply(lambda x: x if x>=0 else 0) # Compute price rise
        bear = diff.apply(lambda x: abs(x) if x<0 else 0) # Compute price drop
        SMA_bull = bull.rolling(window=window).mean()
        SMA_bear = bear.rolling(window=window).mean()
        RS = SMA_bull/SMA_bear
        RSI = 100.0-(100.0/(1+RS))
        self.values = RSI
        self.trace.append((self.n_axis, RSI.values, 'RSI'))
        self.index = df.index
    
    def build_chart(self):
        ndays = np.arange(len(self.index))
        if self.trace != []:
            n_plots = max([tr[0] for tr in self.trace]) + 1
        else:
            n_plots = 1
        
        # 1 subplot case
        if n_plots == 1:
            self.fig, ax = plt.subplots(figsize=self.figsize)
            ax = np.array([ax])
                   
        # multi-subplots case
        else:
            self.fig, ax = plt.subplots(n_plots, 1, gridspec_kw = {'height_ratios': [2]+[1]*(n_plots-1)}, 
                                   sharex=True, figsize=self.figsize) 
            
        # plot canflestick only if instantiated
        if self._cndlstck_called:
            ohlc_data = np.array(get_ohlc(self.df))
            ohlc_data_arr = np.hstack([np.arange(len(ohlc_data))[:,np.newaxis], ohlc_data[:, 1:]])
            candlestick_ohlc(ax[0], ohlc_data_arr, colorup=self.colorup, colordown=self.colordown, width=self.width)

        # plot other traces
        for tr in self.trace:
            if tr[2] != 'Volume':
                ax[tr[0]].plot(tr[1], label=tr[2])
                ax[tr[0]].legend(loc='upper left')
            else:
                # volume bars
                ax[tr[0]].bar(ndays, tr[1], color=self.colors)
            
            # plot decorators
            if tr[2] == 'Momentum':
                ax[tr[0]].axhline(y=0.0, color='k', alpha=.2, linewidth=1.)
            if tr[2] in ['stoch %K', 'RSI']:
                ax[tr[0]].axhline(y=20.0, color='k', alpha=.2, linewidth=1.)
                ax[tr[0]].axhline(y=80.0, color='k', alpha=.2, linewidth=1.)
                
        ax[0].set_title(self.stock)
        
        # Format x axis
        for axis in ax:
            axis.set_xticks(ndays[::self.freq])
            axis.set_xticklabels(self.index.strftime('%b %d')[::self.freq], rotation=90, ha='left')
            axis.set_xlim(ndays.min()-1, ndays.max()+1)
            axis.grid(alpha=0.2, linestyle='--') 
        plt.subplots_adjust(hspace=0.1)
        
    def show(self):
        self.build_chart()
        plt.show()  
        
    def save(self, figname='chart'):
        self.build_chart()
        plt.close()
        self.fig.savefig(figname+'.png') 
        

