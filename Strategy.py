import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint


SAVE = False

PAST_CANDLES = 5
TICK = 0.25

PROFIT = 10
LOSS = 20


def strategy_logic(test):
    if test:
        norm_data = pd.read_csv('processed_data_minute2.csv')
        data = pd.read_csv('raw_data_minute2.csv')
    else:
        norm_data = pd.read_csv('processed_data_minute.csv')
        data = pd.read_csv('raw_data_minute.csv')

    sec_data = data.copy()
    # Create a list of buy/sell labels and risk-reward ratios for each data point
    labels = []
    sequences = []
    trades = []
    
    for i in range(len(data) - 6):
        if i < PAST_CANDLES - 1:
            continue
        if  data.iloc[i]["Open"] < data.iloc[i]["Close"] and \
            data.iloc[i + 1]["Open"] >= data.iloc[i + 1]["Close"] and \
            data.iloc[i + 2]["Open"] >= data.iloc[i + 2]["Close"]:
            if not any([(data.iloc[i + 3]["Open"] - data.iloc[i + j]["Low"]) / TICK >= LOSS for j in range(3, 5)]):
                if any([(data.iloc[i + j]["High"] - data.iloc[i + 3]["Open"]) / TICK >= PROFIT for j in range(3, 5)]):

                    labels.append('Buy')
                    tred = sec_data.iloc[i + 3 : i + 10]
                    trades.append(tred)
                    seq = norm_data.iloc[i - PAST_CANDLES:i + 3].values
                    sequences.append(seq)

            else:

                labels.append('UNK')
                tred = sec_data.iloc[i + 3 : i + 10]
                trades.append(tred)
                seq = norm_data.iloc[i - PAST_CANDLES:i + 3].values

                sequences.append(seq)


        elif data.iloc[i]["Open"] > data.iloc[i]["Close"] and \
             data.iloc[i + 1]["Open"] <= data.iloc[i + 1]["Close"] and \
             data.iloc[i + 2]["Open"] <= data.iloc[i + 2]["Close"]:
            if not any([(data.iloc[i + j]["High"] - data.iloc[i + 3]["Open"]) / TICK >= LOSS for j in range(3, 6)]):
                if any([(data.iloc[i + 3]["Open"] - data.iloc[i + j]["Low"]) / TICK >= PROFIT for j in range(3, 5)]):

                    labels.append('Sell')
                    tred = sec_data.iloc[i + 3 : i + 10]
                    trades.append(tred)
                    seq = norm_data.iloc[i - PAST_CANDLES:i + 3].values
                    sequences.append(seq)

            else: 

                labels.append('UNK')
                tred = sec_data.iloc[i + 3 : i + 10]
                trades.append(tred)
                seq = norm_data.iloc[i - PAST_CANDLES:i + 3].values
                sequences.append(seq)

    for i, elem in enumerate(sequences):
        if elem.shape != (18, 14):
            sequences.remove(elem)
            labels.remove(labels[i])
            trades.remove(trades[i])
            

    print(len(labels))
    Y = pd.get_dummies(labels).values

    # Convert the sequences to numpy arrays
    X = np.array(sequences)
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))
    print(X.shape)


    return X, Y, trades, labels

def model():
    # Split the data into training and testing sets
    X, Y, _, _ = strategy_logic(test = False)
    print(np.sum(Y, axis=0))
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    Y_train, Y_test = Y[:split_index], Y[split_index:]

    model = Sequential()

    model.add(Conv2D(256, (2, 2), activation=tf.nn.elu, padding = 'same', input_shape=(PAST_CANDLES + 3, 14, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (4, 4), activation=tf.nn.elu, padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='leaky_relu'))
    model.add(Dense(3, activation='softmax'))

    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_auc', save_best_only=True, mode='max', verbose=1)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0007), metrics=[tf.keras.metrics.AUC()])
    history = model.fit(X_train, Y_train, batch_size=12, epochs=70, validation_split=0.2, shuffle=True, callbacks=[checkpoint])
    score = model.evaluate(X_test, Y_test, verbose=0)

    # Save your model
    if SAVE == True:
        model.save('minute_trading_model.h5')
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['auc']
    val_acc = history.history['val_auc']

    # Plot the training and validation loss
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the training and validation accuracy
    plt.figure()
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show the plots
    plt.show()





if __name__ == '__main__':
    model()