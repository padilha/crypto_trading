from cryptocmd import CmcScraper
import pandas as pd
scraper = CmcScraper("BTC", fiat="USD", start_date='01-01-2009', end_date='08-10-2023')
headers, data = scraper.get_data()
df = scraper.get_dataframe()
df = df.set_index('Date')
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df.to_csv('data/CMC_BTCUSD-teste.csv')
