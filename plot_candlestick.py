import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from mplfinance.original_flavor import *
from matplotlib.dates import date2num
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def load_data(date_start, date_end, stock, folder_data):
    """ Reads stock data for a given ticker from a CSV file """
    dates = pd.date_range(date_start, date_end)
    df = pd.DataFrame(index=dates)
    file_name = os.path.join(folder_data,'{}.csv'.format(stock))
    df_temp = pd.read_csv(file_name, index_col='Date', parse_dates=True, na_values=['NaN'])
    df = df.join(df_temp)
    df = df.dropna()
    return df


def get_ohlc(df):
    """ Prepare data in the format required by matplotlib.finance.candlestick_ohlc """
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


def weekday_candlestick(ohlc_data, ax, stock_name, fmt='%b %d', freq=7, **kwargs):
    """ Wrapper function for matplotlib.finance.candlestick_ohlc that artificially spaces data to avoid gaps from weekends.
        Takes data in the format required by matplotlib.finance.candlestick_ohlc """
    # Convert data to numpy array
    ohlc_data_arr = np.array(ohlc_data)
    ohlc_data_arr2 = np.hstack(
        [np.arange(ohlc_data_arr[:,0].size)[:,np.newaxis], ohlc_data_arr[:,1:]])
    ndays = ohlc_data_arr2[:,0]  # array([0, 1, 2, ... n-2, n-1, n])

    # Convert matplotlib date numbers to strings based on `fmt`
    dates = mdates.num2date(ohlc_data_arr[:,0])
    date_strings = []
    for date in dates:
        date_strings.append(date.strftime(fmt))

    # Plot candlestick chart
    candlestick_ohlc(ax, ohlc_data_arr2, colorup='#77d879', colordown='#db3f3f', **kwargs)

    # Format x axis
    ax.set_xticks(ndays[::freq])
    ax.set_xticklabels(date_strings[::freq], rotation=45, ha='right')
    ax.set_xlim(ndays.min(), ndays.max())
    axes.grid(True)
    plt.title(stock_name)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def plot_candlestick(df, axes, **kwargs):
    """ Plot candlestick chart for given stock data """
    ohlc = get_ohlc(df)
    weekday_candlestick(ohlc, ax=axes, stock_name=stock, **kwargs)

date_start = '2016-06-01'
date_end = '2016-11-29'
folder_data = 'data'
stock = 'FB'

df = load_data(date_start, date_end, stock, folder_data)
fig, axes = plt.subplots(figsize=(14, 6))
plot_candlestick(df, axes, fmt='%b %d', freq=5, width=0.5)

print("Done !")