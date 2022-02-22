import imp
from tqdm import tqdm
import pandas as pd
from binance import Client
import numpy as np
client = Client()
from indicator import *
from sqlalchemy import create_engine

coins = ('BTCUSDT','ETHUSDT')

def getminutedata(symbol,lookback):
    frame = pd.DataFrame(client.get_historical_klines(symbol,'1h', lookback + 'days ago UTC'))
    frame =frame.iloc[:,:5]
    frame.columns = ['Time','Open','High','Low','Close']
    frame[['Open','High','Low','Close']] = frame[['Open','High','Low','Close']].astype(float)
    frame.Time =pd.to_datetime(frame.Time,unit='ms')
    print(frame)
    return frame

def technicals(df):
    df=df.copy()
    df['return'] = np.log(df.Close.pct_change()+1) #log ofpercentage change of the close price +1
    df['SMA_fast'] = df.Close.rolling(3).mean()
    df['SMA_slow'] = df.Close.rolling(10).mean()
    df['position'] = np.where(df['SMA_fast']>df['SMA_slow'],1,0)
    df['strategyreturn'] = df['position'].shift(1)*df['return']
    df.dropna(inplace=True) # eliminamos las filas donde no se puedan calcular las funciones
    
    return df
    

engine = create_engine('sqlite:///Cryptoprices.db')

#for coin in tqdm(coins):
#    getminutedata(coin,'3').to_sql(coin,engine, index = False)
getminutedata('BTCUSDT','1').to_sql('BTCUSDT',engine, index = False)


test = pd.read_sql('BTCUSDT',engine)#.set_index('Time')
print(technicals(test))
print('listo')