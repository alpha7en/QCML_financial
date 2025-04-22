import pandas as pd
from moexalgo import Market

stocks = Market("shares/TQBR")
all_stocks = pd.DataFrame(stocks.tickers())
print(all_stocks)