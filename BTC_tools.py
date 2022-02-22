
import ta as tb
import pandas as pd
from binance.client import Client
import numpy as np
import math 
client = Client()
import matplotlib.pyplot as plt
from scipy import stats, signal

def emaa(s, n):
    """
    returns an n period exponential moving average for
    the time series s

    s is a list ordered from oldest (index 0) to most
    recent (index -1)
    n is an integer

    returns a numeric array of the exponential
    moving average
    """
    s = np.array(s)
    ema = []
    j = 1

    #get n sma first and calculate the next n period ema
    sma = sum(s[:n]) / n
    multiplier = 2 / float(1 + n)
    ema.append(sma)

    #EMA(current) = ( (Price(current) - EMA(prev) ) x Multiplier) + EMA(prev)
    ema.append(( (s[n] - sma) * multiplier) + sma)

    #now calculate the rest of the values
    for i in s[n+1:]:
        tmp = ( (i - ema[j]) * multiplier) + ema[j]
        j = j + 1
        ema.append(tmp)
    print(ema)
    return ema
def MomentumLazyBear(df):
    """
        Inputs: dataframe of the coin
        
        Outputs: the column of the momentum index 
    """

    df=df.copy()
    # parameter setup (default values in the original indicator)
    length = 20
    length_KC = 20

    # calculate Momentum
    m_avg = df['Close'].rolling(window=length).mean() # moving average
    highest = df['High'].rolling(window = length_KC).max()
    lowest = df['Low'].rolling(window = length_KC).min()

    df['Momentum'] = (df['Close'] - ((highest + lowest)/2 + m_avg)/2)

    fit_y = np.array(range(0,length_KC))
    df['Momentum'] = df['Momentum'].rolling(window = length_KC).apply(lambda x : 
                np.polyfit(fit_y, x, 1)[0] * (length_KC-1) + np.polyfit(fit_y, x, 1)[1], raw=True)
    
    return df['Momentum']

def getdata(symbol,timeframe,lookback):
    """
    Inputs: 
            symbol: coin to be use
            timeframe: 1m 5m .. 1h 4h ... 1d 
            lookbak: number of days 
    Return:
            dataframe: [Time Open High Low Close]
    """
    frame = pd.DataFrame(client.get_historical_klines(symbol,timeframe, lookback + 'days ago UTC'))
    frame =frame.iloc[:,:6]
    frame.columns = ['Time','Open','High','Low','Close','Volume']
    frame[['Open','High','Low','Close','Volume']] = frame[['Open','High','Low','Close','Volume']].astype(float)
    frame.Time =pd.to_datetime(frame.Time,unit='ms')
    #print(frame)
    return frame
def volume_profile(data, show=False):
    """
    Inputs:
            data: dataframe of the coin, it need to include Volume and Close values
            show: show the graphics
    Output:
            POC: point of control

    """
    close=data['Close']
    volume = data['Volume']

    kde_factor = 0.13   # a menor valor, más sensible al precio se comporta la curva
    num_samples = 70    # similar than used in traing view
    kde = stats.gaussian_kde(close,weights=volume,bw_method=kde_factor)
    
    # values of the volumen profile and ticks
    xr = np.linspace(close.min(),close.max(),num_samples) #coin value
    kdy = kde(xr)

    if show==True:
        plt.plot(xr,kdy)
        plt.show()
    peaks,_ = signal.find_peaks(kdy)
    prominences = signal.peak_prominences(kdy,peaks)[0]
    
    # calculamos el maxi prominence, lo ubicamos dentro de su arreglo, 
    # su posición será la misma para el peak. Este peak lo usamos para 
    # calcular el precio maximo
    
    POC = xr[peaks[np.where(prominences==max(prominences))]] 
    
    return POC

def technical_indicators_df(self, daily_data):
        """
        Assemble a dataframe of technical indicator series for a single stock
        """
        o = daily_data['Open'].values
        c = daily_data['Close'].values
        h = daily_data['High'].values
        l = daily_data['Low'].values
        v = daily_data['Volume'].astype(float).values
        # define the technical analysis matrix

        # Most data series are normalized by their series' mean
        ta = pd.DataFrame()
        ta['MA5'] = tb.MA(c, timeperiod=5) / tb.MA(c, timeperiod=5).mean()
        ta['MA10'] = tb.MA(c, timeperiod=10) / tb.MA(c, timeperiod=10).mean()
        ta['MA20'] = tb.MA(c, timeperiod=20) / tb.MA(c, timeperiod=20).mean()
        ta['MA60'] = tb.MA(c, timeperiod=60) / tb.MA(c, timeperiod=60).mean()
        ta['MA120'] = tb.MA(c, timeperiod=120) / tb.MA(c, timeperiod=120).mean()
        ta['MA5'] = tb.MA(v, timeperiod=5) / tb.MA(v, timeperiod=5).mean()
        ta['MA10'] = tb.MA(v, timeperiod=10) / tb.MA(v, timeperiod=10).mean()
        ta['MA20'] = tb.MA(v, timeperiod=20) / tb.MA(v, timeperiod=20).mean()
        ta['ADX'] = tb.ADX(h, l, c, timeperiod=14) / tb.ADX(h, l, c, timeperiod=14).mean()
        ta['ADXR'] = tb.ADXR(h, l, c, timeperiod=14) / tb.ADXR(h, l, c, timeperiod=14).mean()
        ta['MACD'] = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0] / \
                     tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0].mean()
        ta['RSI'] = tb.RSI(c, timeperiod=14) / tb.RSI(c, timeperiod=14).mean()
        ta['BBANDS_U'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0] / \
                         tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0].mean()
        ta['BBANDS_M'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1] / \
                         tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1].mean()
        ta['BBANDS_L'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2] / \
                         tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2].mean()
        ta['AD'] = tb.AD(h, l, c, v) / tb.AD(h, l, c, v).mean()
        ta['ATR'] = tb.ATR(h, l, c, timeperiod=14) / tb.ATR(h, l, c, timeperiod=14).mean()
        ta['HT_DC'] = tb.HT_DCPERIOD(c) / tb.HT_DCPERIOD(c).mean()
        ta["High/Open"] = h / o
        ta["Low/Open"] = l / o
        ta["Close/Open"] = c / o

        self.ta = ta 
