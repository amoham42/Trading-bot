from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_market_calendars import get_calendar

# Define the date range for the weeks you're interested in
start_date = '2023-03-30'
end_date = '2023-04-27'
INTERVAL = '1m'

def load_data(start_date, end_date):
    nyse = get_calendar('NYSE')
    trading_days = nyse.schedule(start_date=start_date, end_date=end_date).index
    dfs = []
    for date in trading_days:
        data = yf.download('NQM23.CME', start=date, end=date + pd.Timedelta(days=1), interval=INTERVAL)
        data = process_indicators(data)
        data = data.dropna(axis=0)
        dfs.append(data)

    data = pd.concat(dfs)

    return data

def normalize_data(df):
    min_max_scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=['float', 'int']).columns

    for col in numeric_cols:
        df[col] = min_max_scaler.fit_transform(df[col].values.reshape(-1, 1))

    return df

def process_indicators(data):

    #EMAs
    ema_first = data['Close'].ewm(span=14, adjust=False).mean()
    ema_second = data['Close'].ewm(span=8, adjust=False).mean()
    data['EMA'] = ema_first
    data['EMAA'] = ema_second

    #Volume Weighted Average Price
    val = data["Close"] * data["Volume"]
    cum_val = val.cumsum()
    cum_vol = data["Volume"].cumsum()
    data["vwap"] = cum_val / cum_vol

    std = data["vwap"].std()
    data["UpperBand1"] = data["vwap"] + (1 * std)
    data["LowerBand1"] = data["vwap"] - (1 * std)
    data["UpperBand2"] = data["vwap"] + (2 * std)
    data["LowerBand2"] = data["vwap"] - (2 * std)
    data["UpperBand3"] = data["vwap"] + (3 * std)
    data["LowerBand3"] = data["vwap"] - (3 * std)

    return data


def main():
    
   
    data = load_data(start_date, end_date)
    data = data.drop(['Adj Close'], axis=1)

    # Drop all data points that are not during regular trading hours
    mask = (data.index.hour <= 16)
    data = data.loc[mask]
    data = data.reset_index(drop=True)

    # Create a list of buy/sell labels and risk-reward ratios for each data point
    norm_data = data.copy()
    norm_data = normalize_data(norm_data)

    norm_data.to_csv('processed_data_minute.csv', index=False)
    data.to_csv('raw_data_minute.csv', index=False)


if __name__ == '__main__':
    main()
    


