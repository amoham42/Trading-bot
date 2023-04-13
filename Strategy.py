import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout



# MIN_INDEX = 340
# SKIPPER = 5
# PROFIT = 4

# nq_data = yf.download('NQM23.CME', start='2023-04-03', end='2023-04-05', interval='1m')
# nq_data_new = pd.DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

# for i in range(0, len(nq_data)):

#     temp = nq_data.iloc[i].name
#     if temp.hour == 4 and temp.minute == 30:
#         nq_data_new = pd.concat([nq_data_new, nq_data.iloc[i:i + MIN_INDEX]])



# def ema_calculator(data, column_name, window, start_index=0):
#     ema = pd.Series(data[column_name].iloc[start_index:].ewm(span=window, adjust=False).mean(), name='EMA_' + str(window))
#     return pd.concat([data.iloc[start_index:], ema], axis=1)


# def strategy():
#     global SKIPPER
#     global PROFIT
#     suc_count = 0
#     fail_count = 0
#     tick = 0.25
#     skip = 0
#     prft = 0

#     for i in range(0, len(nq_data_new)):
#         if skip > 0:
#             skip -= 1
#             continue

#         if (i + 4) < len(nq_data_new):
#             if nq_data_new.iloc[i]["Open"] < nq_data_new.iloc[i]["Close"] and \
#                 nq_data_new.iloc[i + 1]["Open"] > nq_data_new.iloc[i + 1]["Close"] and \
#                 nq_data_new.iloc[i + 2]["Open"] > nq_data_new.iloc[i + 2]["Close"] and \
#                 nq_data_new.iloc[i + 2]["Close"] > nq_data_new.iloc[i + 1]["Low"]:


#                 if ((nq_data_new.iloc[i + 3]["High"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT or \
#                     ((nq_data_new.iloc[i + 4]["High"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT or \
#                     ((nq_data_new.iloc[i + 5]["High"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT:
#                     suc_count += 1
#                     skip += SKIPPER
#                     prft += PROFIT * 5
#                 else:
                    
#                     fail_count += 1
#                     prft -= PROFIT * 6
#                     print(nq_data_new.iloc[i])


#             if nq_data_new.iloc[i]["Open"] < nq_data_new.iloc[i]["Close"] and \
#                 nq_data_new.iloc[i + 1]["Open"] > nq_data_new.iloc[i + 1]["Close"] and \
#                 nq_data_new.iloc[i + 2]["Open"] > nq_data_new.iloc[i + 2]["Close"] and \
#                 nq_data_new.iloc[i + 2]["Close"] > nq_data_new.iloc[i + 1]["High"]:

#                 if ((nq_data_new.iloc[i + 3]["Low"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT or \
#                     ((nq_data_new.iloc[i + 4]["Low"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT or \
#                         ((nq_data_new.iloc[i + 5]["Low"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT:
#                     suc_count += 1
#                     skip += SKIPPER
#                     prft += PROFIT * 5
#                 else:
#                     fail_count += 1
#                     prft -= PROFIT * 6
#                     print("shorted")
#                     print(nq_data_new.iloc[i])

#     success_rate = (suc_count * 100) / (suc_count + fail_count)

#     print("Success rate: " + str(success_rate) + "%")
#     print("Possible Successful trades: " + str(suc_count))
#     print("Possible trades: " + str(suc_count + fail_count))
#     print("Possible profit: $" + str(prft))

    


# strategy()

def normalize_data(df):
    min_max_scaler = MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df['Open'].values.reshape(-1, 1))
    df['High'] = min_max_scaler.fit_transform(df['High'].values.reshape(-1, 1))
    df['Low'] = min_max_scaler.fit_transform(df['Low'].values.reshape(-1, 1))
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    df['Volume'] = min_max_scaler.fit_transform(df['Volume'].values.reshape(-1, 1))
    df['EMA'] = min_max_scaler.fit_transform(df['EMA'].values.reshape(-1, 1))
    df['EMAA'] = min_max_scaler.fit_transform(df['EMAA'].values.reshape(-1, 1))
    df['vwap'] = min_max_scaler.fit_transform(df['vwap'].values.reshape(-1, 1))
    df['UpperBand'] = min_max_scaler.fit_transform(df['UpperBand'].values.reshape(-1, 1))
    df['LowerBand'] = min_max_scaler.fit_transform(df['LowerBand'].values.reshape(-1, 1))

    return df



PROFIT = 5


# Load your candle data
data = yf.download('NQM23.CME', start='2023-03-28', end='2023-04-04', interval='1m')
new_data = yf.download('NQM23.CME', start='2023-03-23', end='2023-03-27', interval='1m')
ne_data = yf.download('NQM23.CME', start='2023-03-15', end='2023-03-22', interval='1m')
n_data = yf.download('NQM23.CME', start='2023-03-13', end='2023-03-14', interval='1m')

data = pd.concat([new_data, data])
data = pd.concat([ne_data, data])
data = pd.concat([n_data, data])

ema = data['Close'].ewm(span=14, adjust=False).mean()
emaa = data['Close'].ewm(span=8, adjust=False).mean()
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
data = data.drop(['Adj Close'], axis=1)
data = data.reset_index(drop=True)
norm_data = data.copy()

norm_data = normalize_data(norm_data)
# Drop unnecessary columns



# Create a list of buy/sell labels and risk-reward ratios for each data point
labels = []
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

            seq = norm_data.iloc[i - 10:i + 3].values
            
            sequences.append(seq)
        else:
            labels.append('UNK')
            seq = norm_data.iloc[i - 10:i + 3].values
            sequences.append(seq)
    elif data.iloc[i]["Open"] > data.iloc[i]["Close"] and \
                data.iloc[i + 1]["Open"] < data.iloc[i + 1]["Close"] and \
                data.iloc[i + 2]["Open"] < data.iloc[i + 2]["Close"]:
        if ((data.iloc[i + 3]["Open"] - data.iloc[i + 3]["Low"]) / 0.25) >= PROFIT or \
                    ((data.iloc[i + 3]["Open"] - data.iloc[i + 4]["Low"]) / 0.25) >= PROFIT or \
                        ((data.iloc[i + 3]["Open"] - data.iloc[i + 5]["Low"]) / 0.25) >= PROFIT:
            labels.append('Sell')
            seq = norm_data.iloc[i - 10:i + 3].values
            sequences.append(seq)
        else: 
            labels.append('UNK')
            seq = norm_data.iloc[i - 10:i + 3].values
            sequences.append(seq)
    

print(len(labels))
y = pd.get_dummies(labels).values


# Convert the sequences to numpy arrays
X = np.array(sequences)
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

# Split the data into training and testing sets
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define the architecture of your RNN model
# model = tf.keras.Sequential([
#     tf.keras.layers.RNN(tf.keras.layers.StackedRNNCells([
#         tf.keras.layers.GRUCell(units=1028),
#         tf.keras.layers.GRUCell(units=512)
#     ]), input_shape=(9, 6)),
    
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(units=128, activation='relu',
#                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.Dropout(0.2),
    
#     tf.keras.layers.Dense(units=64, activation=tf.nn.elu,
#                           kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(units=3, activation='softmax')
# ])

model = Sequential()

# Add a convolutional layer with 32 filters, a 3x3 kernel size, and ReLU activation
model.add(Conv2D(256, (2, 2), activation=tf.nn.elu, padding = 'same', input_shape=(13, 10, 1)))

# Add a max pooling layer with a pool size of 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add another convolutional layer with 64 filters and a 3x3 kernel size
model.add(Conv2D(512, (2, 2), activation=tf.nn.elu, padding = 'same'))

# Add another max pooling layer with a pool size of 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output of the previous layer
model.add(Flatten())
# Add a fully connected layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='leaky_relu'))

# Add an output layer with 10 neurons (for 10 classes) and softmax activation
model.add(Dense(3, activation='softmax'))
# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0007), metrics=[tf.keras.metrics.AUC()])

# Train the model
history = model.fit(X_train, y_train, batch_size=12, epochs=60, validation_split=0.2, shuffle=True)

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
model.save('my_rnn_model.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
