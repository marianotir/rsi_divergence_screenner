
 
# -----------------------
# Libraries
# -----------------------

import streamlit as st 

import pandas as pd
import numpy as np
import datetime as dt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

import requests
from binance import Client



# ----------------------
# Define functions
# ----------------------

def get_symbols_api():

    base_url = 'https://api.binance.com'

    # Define the path for the information we can to get using the api
    path = '/api/v3/exchangeInfo'

    # request the data using get
    r = requests.get(base_url + path)

    info = r.json()

    symbols = [x['symbol'] for x in info['symbols']]

    # exclude low bear and all those symbols
    exclude_list = ['UP','DOWN','BULL','BEAR']
    symbols = [symbol for symbol in symbols if all(excludes not in symbol for excludes in exclude_list)]

    # keep only usdt pairs
    symbols = [symbol for symbol in symbols if symbol.endswith('USDT')]

    return symbols



# -------------- Get history for each symbol -------------- # 

def get_symbol_hist(symbol,interval):

    base_url = 'https://api.binance.com'

    path = '/api/v3/klines'

    parameters = {'symbol':symbol,'interval':interval}

    r = requests.get(base_url + path, parameters)

    data = r.json()

    return data


def data_to_dataframe(data_symbol):

    df = pd.DataFrame(data_symbol)
    df.columns = ['open_time',
                  'o', 'h', 'l', 'c', 'v',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']

    df = df.iloc[:,:6]

    df.columns = ['Time','Open','High','Low','Close','Volume']

    # Set index as Time
    df = df.set_index('Time')

    # Define time in ms units
    df.index  = pd.to_datetime(df.index,unit='ms')

    # Define the data as float
    df = df.astype('float')

    return df


#  ---------- define rsi ---------- #

def rsi(price,n=14):
    delta = price['Close'].diff()
    dUp,dDown = delta.copy(), delta.copy()
    dUp[dUp<0] = 0
    dDown[dDown>0] = 0

    RolUp = dUp.rolling(window = n).mean()
    RolDown = dDown.rolling(window = n).mean().abs()

    RS = RolUp/RolDown
    rsi = 100.0 - (100.0/(1.0+RS))

    return rsi 


#  ---------- find local buttom and local tops ---------- #

# find local buttoms and local tops 
# look at the candel with location l as index in the dataframe and compare with n_back candels
# and n_forward candels names as n_b and n_f and check if it is a minima or a maxima 
def find_max_min_local(df,candle_pos,n_b,n_f):
    if candle_pos-n_b<0 or candle_pos+n_f>=len(df): # This means the candel position is either at the begining or at the end of the price series
        return 0 

    local_low = 1
    local_high = 1
    
    # seach for local high and low around the candle_pos
    for i in range(candle_pos-n_b,candle_pos+n_f+1):
        if(df.Low[candle_pos]>df.Low[i]): # => There are candles lower than the candle position
            local_low = 0
        if(df.High[candle_pos]<df.High[i]): # => There are candles higher than thecandle position
            local_high = 0

    if local_low and local_high:
        return 3
    if local_low: 
        return 1
    if local_high:
        return 2
    else: 
        return 0 


#  ---------- find local buttom and local tops in the rsi ---------- #

# find local buttoms and tops in the rsi series 
def rsi_find(df,candle_pos,n_b,n_f): 
    if candle_pos-n_b<0 or candle_pos+n_f>=len(df): # This means the candel position is either at the begining or at the end of the price series
       return 0 

    local_low = 1
    local_high = 1
    
    # seach for local high and low around the candle_pos
    for i in range(candle_pos-n_b,candle_pos+n_f+1):
        if(df.RSI[candle_pos]>df.RSI[i]): # => There are candles lower than the candle position
            local_low = 0
        if(df.RSI[candle_pos]<df.RSI[i]): # => There are candles higher than thecandle position
            local_high = 0

    if local_low and local_high:
        return 3
    if local_low: 
        return 1
    if local_high:
        return 2
    else: 
        return 0 


#  ---------- Define position of each local buttom and top ---------- #

def candle_pos(df):
    if df['Candle_category'] == 1: # The candle is local low
        return df.Low - 1e-3
    elif df['Candle_category'] == 2: # The candles is local high 
        return df.High + 1e-3
    else: 
        return np.nan

def rsi_candle_pos(df):
    if df['RSI_Candle_category'] == 1:
        return df.RSI - 1
    elif df['RSI_Candle_category'] == 2:
        return df.RSI + 1
    else: 
        return np.nan


# ------------------------------------
# Get Slopes from current candle
# ------------------------------------
# Get Slopes from current candle using the local maxima and minima located in the previous 60 candles form the current candle

# All local max and min candles will be storaged for later on make the fits 
def find_slopes(x,df,nbackcandles):

    try:

        candle_id = int(x.name)

        backcandles = nbackcandles

        maxim = np.array([])
        minim = np.array([])
        xxmax = np.array([])
        xxmin = np.array([])

        maximRSI = np.array([])
        minimRSI = np.array([])
        xxmaxRSI = np.array([])
        xxminRSI = np.array([])

        for i in range(candle_id-backcandles,candle_id+1):
            if df.iloc[i].Candle_category == 1: # => The candle i is a local low
                minim = np.append(minim, df.iloc[i].Low) 
                xxmin = np.append(xxmin,i)
            if df.iloc[i].Candle_category == 2: # => The candle i is a local low
                maxim = np.append(maxim, df.iloc[i].High) 
                xxmax = np.append(xxmax,i)

            if df.iloc[i].RSI_Candle_category == 1: # => The candle i is a local low
                minimRSI = np.append(minimRSI, df.iloc[i].RSI) 
                xxminRSI = np.append(xxminRSI,i)
            if df.iloc[i].RSI_Candle_category == 2: # => The candle i is a local low
                maximRSI = np.append(maximRSI, df.iloc[i].RSI) 
                xxmaxRSI = np.append(xxmaxRSI,i)

        if maxim.size<2 or minim.size<2 or maximRSI.size<2 or minimRSI.size<2:
            return "No Signal"

        # Fit the arrays found above 
        slmax, intercmax = np.polyfit(xxmax,maxim,1)
        slmin, intercmin = np.polyfit(xxmin,minim,1)
        slmaxRSI, intercmaxRSI = np.polyfit(xxmaxRSI,maximRSI,1)
        slminRSI, intercminRSI = np.polyfit(xxminRSI,minimRSI,1)             

        if slmin>1 and slmax>1 and slmaxRSI<-0.1 and slminRSI<-0.1: # => Up trend but rsi down 
            return "Sell"    
        elif slmin<-1 and slmax<-1 and slminRSI>0.1 and slmaxRSI>0.1: # => DownTrend but rsi up
            return "Buy"
        else: 
            return "No Signal"   

    except:
        return "No Signal"

 
# --------------------------------
# Define all into a function 
# --------------------------------

def get_div(df,backcandles): 

    # Get rsi 
    df['RSI'] = rsi(df,n=14)

    df['Candle_category'] = df.apply(lambda x: find_max_min_local(df,x.name,5,5),axis=1) # x.name is the index or position of the candle 
    df['RSI_Candle_category'] = df.apply(lambda x: rsi_find(df,x.name,5,5),axis=1)


    df['postCandle'] = df.apply(lambda row: candle_pos(row),axis=1)
    df['postRSI'] = df.apply(lambda row: rsi_candle_pos(row),axis=1)

    df['Signal_Div'] =df.apply(lambda row: find_slopes(row,df,backcandles),axis=1)

    return df


# --------------------------------------
# Apply over all symbols 
# --------------------------------------

def main(): 

    print('Analyze divergence begins')

    symbols = get_symbols_api()

    interval = '4h'

    backcandles = 60

    tokens_detected = []

    for symbol in symbols:

        print('Analyze divergente for: ' + str(symbol))

        data_symbol = get_symbol_hist(symbol,interval) 

        df = data_to_dataframe(data_symbol)
                        
        df = df[len(df)-backcandles-30:]

        df.reset_index(inplace=True)

        df = get_div(df,backcandles)

        # Get symbol if there is divergence 
        if len(df[df.Signal_Div != 'No Signal']) > 0:
            print('Divergence detected on: ' + str(symbol) + '\n')
            tokens_detected.append(symbol)

    if len(tokens_detected) > 0:
        print('Tokens with divergence: ' + str(tokens_detected))      
    else: 
        print('No divergences found')

    print('Script completed successfully')


if __name__ == "__main__":
      main()

