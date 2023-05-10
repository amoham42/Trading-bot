import tensorflow as tf
import pandas as pd
import numpy as np
import preprocess as pp
import Strategy as st


PAST_CANDLES = 15
TICK = 0.25
PROFIT = 8
LOSS = 60
CONTRACT = 5
PERCENT = 95.0


my_model = tf.keras.models.load_model('best_model.h5')
norm_data = pd.read_csv('processed_data_minute2.csv')
data = pd.read_csv('raw_data_minute2.csv')
X, _, trades, labels = st.strategy_logic(test = True)


# testing to see if the model is profitable
profit = 0
loss = 0
for seq in range(X.shape[0]):
    array = np.array(X[seq])

    array = np.reshape(array, (1, array.shape[0], array.shape[1], 1))
    predicted_probability = my_model.predict(array)
    predicted_percentages = predicted_probability * 100

    if predicted_percentages[0][0] >= PERCENT:
        
        if any([(trades[seq].iloc[j]["High"] - trades[seq].iloc[j]["Open"]) / TICK >= PROFIT for j in range(0, 5)]):
            profit += (PROFIT * 5 * CONTRACT) - 5
            print("MONEY")
              
        else:
            loss += (LOSS * 5 * CONTRACT) + 5
            print(trades[seq])
            print(labels[seq])
            print(predicted_percentages)
        
    if predicted_percentages[0][1] >= PERCENT:
   
        if any([(trades[seq].iloc[j]["Open"] - trades[seq].iloc[j]["Low"]) / TICK >= PROFIT for j in range(0, 5)]):
            profit += (PROFIT * 5 * CONTRACT) - 5
            print("MONEY")
            
           
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
    