from tqdm import tqdm
import pandas as pd
from binance import Client
import numpy as np
from BTC_tools import *
client = Client()
from indicator import *
from sqlalchemy import create_engine
import ta 
from scipy import stats, signal
import matplotlib.pyplot as plt


def technicals(df):
    df=df.copy()
    df['ADX'] = ta.trend.adx(df.High, df.Low, df.Close,window=14)
    df['Momentum'] = MomentumLazyBear(df)
    df['EMA10'] = ta.trend.ema_indicator(df.Close, window=10)
    df['EMA55'] = ta.trend.ema_indicator(df.Close, window=55)

    df.dropna(inplace=False) # eliminamos las filas donde no se puedan calcular las funciones
    return df
    
engine = create_engine('sqlite:///Cryptoprices.db')
data_test=getdata('BTCUSDT','1h','8').set_index('Time') # get the data 
#test.to_sql('BTCUSDT',engine, index = False) #almacenamos en sql
#test = pd.read_sql('BTCUSDT',engine)#.set_index('Time')

indicators= technicals(data_test)

print(indicators[['ADX', 'Momentum','Volume']])

print(volume_profile(data_test,True)) # Se obtinee el POC de una sesión, pero se podría obtener 2 y asignarle un peso a cada uno


print('listo')

