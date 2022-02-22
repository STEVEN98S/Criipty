"""
    Se realiza el backtesting de la estrategia del ADX + Momentum indicator en temporalidad de 4H.
    El backtesting utilizará un timeWindow de 2 año
    Estrategia:
           Entrada en long:
                            -Tendencia alcista
                            -Direccionalidad alcista
                            -Pendiente negativa
                            -ema de 10 tocando la ema de 55 por encima     
            Entrada en short:
                            -Tendencia bajista
                            -Direccionalidad bajista
                            -Pendiente negativa
                            -ema de 10 tocando la ema de 55 por debajo
""" 
from binance import Client
from ADX_Moment_4H_tools import *
from sqlalchemy import create_engine
#engine = create_engine('sqlite:///DayTendency.db')

api_key = 'OP46t00GHZ3c505N9Ne9l1u92CQ4Vquk43y4844bC2toKblnuI0cM8dkYeTha9ZK'
secret_key = 'OJj37klrLOfD6T7E08lR0nWseH44CNP3meyoIwVWZRxMeopeE9R0BRfGsQ01Tiyd'

client = Client(api_key,secret_key)

data_year = get_data('BTCUSDT','1d','3 year')
datayear_indicators= technicals(data_year)

data_fourhour = get_data('BTCUSDT','4h','3 year')
datafour_indicators= technicals(data_fourhour)

def fourH_strategytest(symbol,open_positoin=False):
    
    for candle in datafour_indicators:            
        if not open_position:    
            if candle.momentumCheck and candle.adxCheck:
                order = client.create_order(symbol='BTCUSDT', side = 'BUY', type='MARKET',quantity='0.0001')
                open_position=True
                buyprice = float(candle.Open) #? Sería Close o Open? compro en el inicio de la vela o en el cierre
        if open_position:
            if candle.Close/buyprice >1.15 or  candle.Close/buyprice<0.90:
                order = client.create_order(symbol='BTCUSDT', side = 'SELL', type='MARKET',quantity='0.0001')
                sellprice = float(candle.Close)
                #print(f"You made {(sellprice-buyprice)/buyprice} profit")
                open_position= False

data_year = get_data('BTCUSDT','1d','3 year')

#data_indicators.to_sql('BTCUSDT',engine, index = False) #almacenamos en sql
#print(indicators.Tendency)
#print(indicators[['ADX', 'Momentum','Volume','Tendency']])

#print(volume_profile(data,True)) # Se obtinee el POC de una sesión, pero se podría obtener 2 y asignarle un peso a cada uno

print('listo')


