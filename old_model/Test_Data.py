import tensorflow as tf
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import preprocess as pp
import Strategy as st


PAST_CANDLES = 10
TICK = 0.25
PROFIT = 10
LOSS = 20
CONTRACT = 5
PERCENT = 100.0


my_model = tf.keras.models.load_model('my_rnn_model.h5')
pp.main()
norm_data = pd.read_csv('processed_data.csv')
data = pd.read_csv('raw_data.csv')
X, _, labels, trades = st.strategy_logic(norm_data, data)


# testing to see if the model is profitable
profit = 0
loss = 0
for seq in range(X.shape[0]):
    array = np.array(X[seq])

    array = np.reshape(array, (1, array.shape[0], array.shape[1], 1))
    predicted_probability = my_model.predict(array)
    predicted_percentages = predicted_probability * 100

    if labels[seq] == "Buy" and predicted_percentages[0][0] >= PERCENT:

        if ((trades[seq].iloc[0]["Open"] - trades[seq].iloc[1]["Close"]) / 0.25) >= LOSS or \
            ((trades[seq].iloc[0]["Open"] - trades[seq].iloc[2]["Close"]) / 0.25) >= LOSS:

            loss += (LOSS * 5 * CONTRACT) + 5
            print(trades[seq])
            print(labels[seq])
            print(predicted_percentages)

        elif any([(trade['High'] - trade['Open']) / 0.25 >= PROFIT for trade in trades[seq][:3]]):
            profit += (PROFIT * 5 * CONTRACT) - 5
            
        else:
            loss += (LOSS * 5 * CONTRACT) + 5
            print(trades[seq])
            print(labels[seq])
            print(predicted_percentages)
        
    if labels[seq] == "Sell" and predicted_percentages[0][1] >= PERCENT:

        if ((trades[seq].iloc[1]["Close"] - trades[seq].iloc[0]["Open"]) / 0.25) >= LOSS or \
            ((trades[seq].iloc[2]["Close"] - trades[seq].iloc[0]["Open"]) / 0.25) >= LOSS:

            loss += (LOSS * 5 * CONTRACT) + 5
            print(trades[seq])
            print(labels[seq])
            print(predicted_percentages)
            
        elif any([(trade['Open'] - trade['Low']) / 0.25 >= PROFIT for trade in trades[seq][:3]]):
            profit += (PROFIT * 5 * CONTRACT) - 5
           
        else:
            loss += (LOSS * 5 * CONTRACT) + 5
            print(trades[seq])
            print(labels[seq])
            print(predicted_percentages)
        
            

print("---------------------")
print("Profits: " + str(profit))
print("Losses: " + str(loss))
precentage = abs(profit) - (abs(loss))
print("Overall: " + str(precentage))
print("---------------------")
    