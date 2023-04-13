import tensorflow as tf
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

my_model = tf.keras.models.load_model('my_rnn_model.h5')



def normalize_data(df):
    min_max_scaler = MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df['Open'].values.reshape(-1, 1))
    df['High'] = min_max_scaler.fit_transform(df['High'].values.reshape(-1, 1))
    df['Low'] = min_max_scaler.fit_transform(df['Low'].values.reshape(-1, 1))
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    df['Volume'] = min_max_scaler.fit_transform(df['Volume'].values.reshape(-1, 1))
    df['EMA'] = min_max_scaler.fit_transform(df['EMA'].values.reshape(-1, 1))
    df['EMAA'] = min_max_scaler.fit_transform(df['EMAA'].values.reshape(-1, 1))


    return df



PROFIT = 10
LOSS = 30
percent = 100.0

# Load your candle data
data = yf.download('NQM23.CME', start='2023-04-05', end='2023-04-06', interval='1m')



ema = data['Close'].ewm(span=21, adjust=False).mean()
emaa = data['Close'].ewm(span=14, adjust=False).mean()
data['EMA'] = ema
data['EMAA'] = emaa
val = data["Close"] * data["Volume"]
cum_val = val.cumsum()
cum_vol = data["Volume"].cumsum()
data["vwap"] = cum_val / cum_vol
sma = data["vwap"].rolling(window=20).mean()
std = data["vwap"].rolling(window=20).std()
data["UpperBand"] = sma + std
data["LowerBand"] = sma - std

mask = (data.index.hour >= 9) & (data.index.minute >= 30) & (data.index.hour < 16)
data = data.loc[mask]
sec_data = data.copy()
data = data.drop(['Adj Close'], axis=1)
data = data.reset_index(drop=True)
norm_data = data.copy()


norm_data = normalize_data(norm_data)


labels = []
trades = []
sequences = []
for i in range(len(data)-6):
    if i< 9:
        continue
    if data.iloc[i]["Open"] < data.iloc[i]["Close"] and \
                data.iloc[i + 1]["Open"] > data.iloc[i + 1]["Close"] and \
                data.iloc[i + 2]["Open"] > data.iloc[i + 2]["Close"]:
        if ((data.iloc[i + 3]["High"] - data.iloc[i + 3]["Open"]) / 0.25) >= PROFIT or \
                    ((data.iloc[i + 4]["High"] - data.iloc[i + 3]["Open"]) / 0.25) >= PROFIT or \
                    ((data.iloc[i + 5]["High"] - data.iloc[i + 3]["Open"]) / 0.25) >= PROFIT:
            labels.append('Buy')
            tred = sec_data.iloc[i + 4 : i + 7]
            trades.append(tred)
            seq = norm_data.iloc[i - 10:i + 3].values
            
            sequences.append(seq)
        else:
            labels.append('UNK')
            tred = sec_data.iloc[i + 4 : i + 7]
            trades.append(tred)
            seq = norm_data.iloc[i - 10:i + 3].values
            sequences.append(seq)
    elif data.iloc[i]["Open"] > data.iloc[i]["Close"] and \
                data.iloc[i + 1]["Open"] < data.iloc[i + 1]["Close"] and \
                data.iloc[i + 2]["Open"] < data.iloc[i + 2]["Close"]:
        if ((data.iloc[i + 3]["Open"] - data.iloc[i + 3]["Low"]) / 0.25) >= PROFIT or \
                    ((data.iloc[i + 3]["Open"] - data.iloc[i + 4]["Low"]) / 0.25) >= PROFIT or \
                        ((data.iloc[i + 3]["Open"] - data.iloc[i + 5]["Low"]) / 0.25) >= PROFIT:
            labels.append('Sell')
            tred = sec_data.iloc[i + 4 : i + 7]
            trades.append(tred)
            seq = norm_data.iloc[i - 10:i + 3].values
            
            sequences.append(seq)
        else: 
            labels.append('UNK')
            tred = sec_data.iloc[i + 4 : i + 7]
            trades.append(tred)
            seq = norm_data.iloc[i - 10:i + 3].values

            sequences.append(seq)
X = np.array(sequences)


X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
profit = 0
loss = 0
for seq in range(X.shape[0]):
    array = np.array(X[seq])

    array = np.reshape(array, (1, array.shape[0], array.shape[1], 1))
    predicted_probability = my_model.predict(array)
    predicted_percentages = predicted_probability * 100

    if labels[seq] == "Buy" and predicted_percentages[0][0] >= percent:

        if ((trades[seq].iloc[0]["High"] - trades[seq].iloc[0]["Open"]) / 0.25) >= PROFIT or \
            ((trades[seq].iloc[1]["High"] - trades[seq].iloc[1]["Open"]) / 0.25) >= PROFIT or \
            ((trades[seq].iloc[2]["High"] - trades[seq].iloc[2]["Open"]) / 0.25) >= PROFIT:
            profit += (PROFIT * 5) - 5
        else:
            loss += (LOSS * 5) + 5
            print(trades[seq])
            print(labels[seq])
    if labels[seq] == "Sell" and predicted_percentages[0][1] >= percent:
        if ((trades[seq].iloc[0]["Open"] - trades[seq].iloc[0]["Low"]) / 0.25) >= PROFIT or \
            ((trades[seq].iloc[1]["Open"] - trades[seq].iloc[1]["Low"]) / 0.25) >= PROFIT or \
            ((trades[seq].iloc[2]["Open"] - trades[seq].iloc[2]["Low"]) / 0.25) >= PROFIT:
            profit += (PROFIT * 5) - 5
        else:
            loss += (LOSS * 5) + 5
            print(trades[seq])
            print(labels[seq])
            

print("---------------------")
print("Profits: " + str(profit))
print("Losses: " + str(loss))
precentage = abs(profit) - (abs(loss))
print("Overall: " + str(precentage))
print("---------------------")
    