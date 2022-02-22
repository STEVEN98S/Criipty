from binance import Client
import pandas as pd 
api_key = 'OP46t00GHZ3c505N9Ne9l1u92CQ4Vquk43y4844bC2toKblnuI0cM8dkYeTha9ZK'
secret_key = 'OJj37klrLOfD6T7E08lR0nWseH44CNP3meyoIwVWZRxMeopeE9R0BRfGsQ01Tiyd'

client = Client(api_key,secret_key)

print(client.get_account())
print('listo')