import yfinance as yf
import pandas as pd
import os

start_date = '2019-01-01'
end_date = '2023-12-31'

folders = ['BSE','Non_BSE','NSE', 'Non_NSE']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

bse_stocks = ['RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'HINDUNILVR.BO', 'INFY.BO','KOTAKBANK.BO', 'ICICIBANK.BO', 'LT.BO', 'HDFC.BO', 'SBIN.BO']
other_bse_stocks = ['GOOGL', 'AAPL', 'AMZN', 'MSFT', 'NVDA', 'GOOG', 'NFLX', 'TSLA', 'INTC', 'CSCO']


nse_stocks = ['RELIANCE.NS', 'TCS.NS', 'HINDUNILVR.NS', 'INFY.NS','KOTAKBANK.NS', 'ICICIBANK.NS', 'LT.NS', 'SBIN.NS', 'ITC.NS', 'ONGC.NS']
other_nse_stocks = ['BABA', 'BIDU', 'NVDA', 'JD', 'PYPL', 'SNAP', 'MCD', 'UBER', 'LYFT', 'SQ']

def download_stock_data():
    def fetch_and_save_stock_data(ticker, folder):
        stock_data = yf.download(ticker, start_date, end_date)
        stock_data.reset_index(inplace=True)
        stock_data.to_csv(f'{folder}/{ticker.replace(".", "_")}.csv', index=False)

    for stock in bse_stocks:
        fetch_and_save_stock_data(stock,'BSE')

    for stock in other_bse_stocks:
        fetch_and_save_stock_data(stock,'Non_BSE')

    for stock in nse_stocks:
        fetch_and_save_stock_data(stock, 'NSE')

    for stock in other_nse_stocks:
        fetch_and_save_stock_data(stock, 'Non_NSE')

bse_data = yf.download('^BSESN', start_date, end_date)
bse_data.reset_index('Date',inplace=True)
bse_data.to_csv('BSE/Sensex.csv', index=False)

nse_data = yf.download('^NSEI', start_date, end_date)
nse_data.reset_index('Date',inplace=True)
nse_data.to_csv('NSE/Nifty.csv', index=False)

def generate_data(bse_or_nse,bse_data,nse_data,bse_stocks,other_bse_stocks,nse_stocks,other_nse_stocks):
    if bse_or_nse == 'BSE':
        df = pd.DataFrame({'Date': bse_data['Date']})
        for stock in bse_stocks:
            stock_data = pd.read_csv(f'BSE/{stock.replace(".", "_")}.csv')
            df[stock.replace(".", "_")] = stock_data['Adj Close'].pct_change()

        for stock in other_bse_stocks:
            stock_data = pd.read_csv(f'Non_BSE/{stock.replace(".","_")}.csv')
            df[stock.replace(".","_")] = stock_data['Adj Close'].pct_change()

        bse_data = pd.read_csv('BSE/Sensex.csv')
        df['Sensex'] = bse_data['Adj Close'].pct_change()
        # print(df)
        df.to_csv('bsedata1.csv', index=False)
    else:
        df = pd.DataFrame({'Date': nse_data['Date']})
        for stock in nse_stocks:
            stock_data = pd.read_csv(f'NSE/{stock.replace(".", "_")}.csv')
            df[stock.replace(".", "_")] = stock_data['Adj Close'].pct_change()

        for stock in other_nse_stocks:
            stock_data = pd.read_csv(f'Non_NSE/{stock.replace(".", "_")}.csv')
            df[stock.replace(".", "_")] = stock_data['Adj Close'].pct_change()

        nse_data = pd.read_csv('NSE/Nifty.csv')
        df['Nifty'] = nse_data['Adj Close'].pct_change()
        # print(df)
        df.to_csv('nsedata1.csv', index=False)


download_stock_data()
generate_data("BSE",bse_data,nse_data,bse_stocks,other_bse_stocks,nse_stocks,other_nse_stocks)
generate_data("NSE",bse_data,nse_data,bse_stocks,other_bse_stocks,nse_stocks,other_nse_stocks)