from pandas_datareader import DataReader
from datetime import timedelta
from datetime import datetime
import numpy as np
import pandas as pd
import os
import pickle

class data_manager():
    
    def __init__(self, folder=None, archive=None, date_start=None, date_end=None):
        self.folder = 'data' if folder is None else folder 
        self.archive = 'FMIB_prices' if archive is None else archive
        self.file_name = os.path.join(self.folder, '{}.pkl'.format(self.archive))
        # load data if archive already exists 
        if os.path.isfile(self.file_name):
            with open(self.file_name, 'rb') as f:
                self.data = pickle.load(f)
            self.ticker_list = self.data['Close'].columns.tolist()
            self._dataset_loaded = True
            
            if date_start is not None:
                T_start = datetime.strptime(date_start, '%Y-%m-%d')
            else:
                T_start = self.data.index[0]

            if date_end is not None:
                T_end = datetime.strptime(date_end, '%Y-%m-%d')
            else:
                T_end = self.data.index[-1]

            T_ix = (self.data.index >= T_start) & (self.data.index <= T_end)
            self.data = self.data.loc[T_ix, :]
            
        else:
            print('The archive "{}" does not exist yet.'.format(self.file_name))
            self._dataset_loaded = False
            
    def stock_price(self, stock=None, date_start=None, date_end=None):
        """ Return requested stock price as pandas DataFrame """
        self.stock_name = self.ticker_list[0] if stock is None else stock 
        if date_start is not None:
            T_start = datetime.strptime(date_start, '%Y-%m-%d')
        else:
            T_start = self.data.index[0]

        if date_end is not None:
            T_end = datetime.strptime(date_end, '%Y-%m-%d')
        else:
            T_end = self.data.index[-1]

        T_ix = (self.data.index >= T_start) & (self.data.index <= T_end)

        openp = self.data.loc[T_ix, 'Open'][self.stock_name].values
        highp = self.data.loc[T_ix, 'High'][self.stock_name].values
        lowp = self.data.loc[T_ix, 'Low'][self.stock_name].values
        closep = self.data.loc[T_ix, 'Close'][self.stock_name].values
        adj_closep = self.data.loc[T_ix, 'Adj Close'][self.stock_name].values
        volumep = self.data.loc[T_ix, 'Volume'][self.stock_name].values

        prices = pd.DataFrame(np.array([openp, highp, lowp, closep, adj_closep, volumep]).T, index=self.data.index[T_ix],
                              columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        return prices        
            
    def tickers(self):
        if self._dataset_loaded:
            print("Tickers: ")
            print('\n'.join(self.ticker_list))
            print('#{} tickers found in "{}".\n'.format(len(self.ticker_list), self.file_name))
        else:
            print('ERROR: The archive "{}" has not yet been initialized!'.format(self.file_name))
        
    def status(self):
        if self._dataset_loaded:
            print("First available record: {}".format(self.data.index[0].strftime("%Y-%m-%d")))
            print("Last available record:  {}".format(self.data.index[-1].strftime("%Y-%m-%d")))
            print("Total number of records: {}\n".format((self.data.index[-1] - self.data.index[0]).days))
        else:
            print('ERROR: The archive "{}" has not yet been initialized!'.format(self.file_name))
        
    def update(self, data_source='yahoo', save=True):
        ''' data_source: use 'yahoo' for stocks and 'stooq' for index '''
        if self._dataset_loaded:
            T_start = (self.data.index[-1] + timedelta(days=1)).strftime("%Y-%m-%d")
            T_end = datetime.now().strftime("%Y-%m-%d")
            if T_start <= T_end:
                print("Downloading new records from {} to {}.".format(T_start, T_end))
                self.new_data = DataReader(self.ticker_list, data_source, T_start, T_end)        
                if len(self.new_data) > 0:
                    if T_start != T_end:
                        # concatenate new data to the existing dataset
                        self.data = pd.concat([self.data, self.new_data])
                    else:
                        # update most updated value on the same row
                        self.data[self.data.columns.levels[0]].loc[self.new_data.index] = self.new_data
                print('#{} records added to "{}"'.format(len(self.new_data), self.file_name))
                # save to pickle file    
                if save: 
                    with open(self.file_name,'wb') as f:
                        pickle.dump(self.data, f)
            else:
                print("ERROR: Start and end dates seem to be inconsistent for the Downloader!")
        else:
            print('ERROR: The archive "{}" has not yet been initialized!'.format(self.file_name))
                
    def downlaod(self, ticks=None, T_start=None, T_end=None, data_source='yahoo', save=True):
        ''' data_source: use 'yahoo' for stocks and 'stooq' for index '''
        if ticks is not None:
            symbols = ticks
        elif self._dataset_loaded:
            symbols = self.ticker_list
        else:
            print("ERROR: Tickers must be specified!")
            return None
            
        print("Downloading new records from {} to {}.".format(T_start, T_end))
        self.data = DataReader(symbols, data_source, T_start, T_end)
        print("#{} records downloaded".format(len(self.data)))
        if save: 
            with open(self.file_name,'wb') as f:
                pickle.dump(self.data, f)
        
        self._dataset_loaded = True
        self.ticker_list = self.data['Close'].columns.tolist()
        
    def remove(self):
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)
            print('The archive "{}" has been removed.'.format(self.file_name))
        else:
            print('ERROR: The archive "{}" does not exist'.format(self.file_name))
