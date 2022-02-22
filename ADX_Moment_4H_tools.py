from cmath import nan
from binance import Client
import pandas as pd 
from datetime import datetime
from scipy import stats, signal
import matplotlib.pyplot as plt
import numpy as np
import ta 

api_key = 'OP46t00GHZ3c505N9Ne9l1u92CQ4Vquk43y4844bC2toKblnuI0cM8dkYeTha9ZK'
secret_key = 'OJj37klrLOfD6T7E08lR0nWseH44CNP3meyoIwVWZRxMeopeE9R0BRfGsQ01Tiyd'

client = Client(api_key,secret_key)

def get_data(symbol,interval, lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol,interval, lookback+' ago UTC'))
    frame = frame.iloc[:,:6]
    frame.columns = ['Time','Open','High','Low','Close','Volume']
    frame =frame.set_index('Time')
    frame.index= pd.to_datetime(frame.index,unit='ms')
    frame =frame.astype(float)
    return frame

def technicals(df):
    df=df.copy()
    df['ADX'] = ta.trend.adx(df.High, df.Low, df.Close,window=14)
    df['Momentum'] = MomentumLazyBear(df)
    df['EMA10'] = ta.trend.ema_indicator(df.Close, window=10)
    df['EMA55'] = ta.trend.ema_indicator(df.Close, window=55)
    df.dropna(inplace=False) # eliminamos las filas donde no se puedan calcular las funciones

    df['Tendency'] = get_tendency(df['EMA10'],df['EMA55'],"Alcista")
    df['momentumCheck'] = momentumCheck(df['Momentum'])
    df['adxCheck'] = adxCheck(df['ADX'])
    
    return df

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

def momentumCheck(Momentum):
    """
        comparamos el Momentum[-1]>Momentum[-2], señal  de compra activada, activar momentum check
        Input:
                dataframe original dataframe
        Return: 
                dataframe with the Momenchek 
    """
    for momentum in Momentum:
        momentumCheck.append(np.where(momentum[-1]>momentum[-2],True,False))
        
    return momentumCheck
def adxCheck(Adx):
    """
        Comparamos el ADX[-1]<ADX[-2], señal de compra activada, activar adxCheck
        Input:
                dataframe original dataframe
        Return: 
                dataframe with the adxCheck
    """
    for adx in Adx:
        adxCheck.append(np.where(adx[-1]<adx[-2],True,False))
    
    return adxCheck
        
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

def get_tendency(ema10,ema55,tendencia_Ini):
    """
        entradas: ema10, ema55, Tendencia_Ini
        salida: tendencia en la que se encuentra la vela
        NOTA:   Es necesario definir la tendencia inicial del TimeWindow que se elige
                Si se coloca mal, las primeras 55 velas saldrán erroneas
    """
    tendencia_actual="Alcista"
    tendencia=[]
    for ema10, ema55 in zip(ema10, ema55): #obtenemos los valores en cada iteración   
        if ema10 < ema55*0.94 and tendencia[-1] =="Alcista":
            tendencia_actual="Bajista"
        if ema10 > ema55*1.06 and tendencia[-1] =="Bajista":
            tendencia_actual="Alcista"
        tendencia.append(tendencia_actual)
    return tendencia