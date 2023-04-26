import pandas as pd
import numpy as np
import yfinance as yf




MIN_INDEX = 340
SKIPPER = 5
PROFIT = 5

nq_data = yf.download('NQM23.CME', start='2023-04-03', end='2023-04-07', interval='1m')
nq_data_new = pd.DataFrame(columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

for i in range(0, len(nq_data)):

    temp = nq_data.iloc[i].name
    if temp.hour == 4 and temp.minute == 30:
        nq_data_new = pd.concat([nq_data_new, nq_data.iloc[i:i + MIN_INDEX]])



def ema_calculator(data, column_name, window, start_index=0):
    ema = pd.Series(data[column_name].iloc[start_index:].ewm(span=window, adjust=False).mean(), name='EMA_' + str(window))
    return pd.concat([data.iloc[start_index:], ema], axis=1)


def strategy():
    global SKIPPER
    global PROFIT
    suc_count = 0
    fail_count = 0
    tick = 0.25
    skip = 0
    prft = 0

    for i in range(0, len(nq_data_new)):
        if skip > 0:
            skip -= 1
            continue

        if (i + 4) < len(nq_data_new):
            if nq_data_new.iloc[i]["Open"] < nq_data_new.iloc[i]["Close"] and \
                nq_data_new.iloc[i + 1]["Open"] > nq_data_new.iloc[i + 1]["Close"] and \
                nq_data_new.iloc[i + 2]["Open"] > nq_data_new.iloc[i + 2]["Close"] and \
                nq_data_new.iloc[i + 2]["Close"] > nq_data_new.iloc[i + 1]["Low"]:


                if ((nq_data_new.iloc[i + 3]["High"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT or \
                    ((nq_data_new.iloc[i + 4]["High"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT or \
                    ((nq_data_new.iloc[i + 5]["High"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT:
                    suc_count += 1
                    skip += SKIPPER
                    prft += PROFIT * 5
                else:
                    
                    fail_count += 1
                    prft -= PROFIT * 6
                    print(nq_data_new.iloc[i])


            if nq_data_new.iloc[i]["Open"] < nq_data_new.iloc[i]["Close"] and \
                nq_data_new.iloc[i + 1]["Open"] > nq_data_new.iloc[i + 1]["Close"] and \
                nq_data_new.iloc[i + 2]["Open"] > nq_data_new.iloc[i + 2]["Close"] and \
                nq_data_new.iloc[i + 2]["Close"] > nq_data_new.iloc[i + 1]["High"]:

                if ((nq_data_new.iloc[i + 3]["Low"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT or \
                    ((nq_data_new.iloc[i + 4]["Low"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT or \
                        ((nq_data_new.iloc[i + 5]["Low"] - nq_data_new.iloc[i + 3]["Open"]) / tick) >= PROFIT:
                    suc_count += 1
                    skip += SKIPPER
                    prft += PROFIT * 5
                else:
                    fail_count += 1
                    prft -= PROFIT * 6
                    print("shorted")
                    print(nq_data_new.iloc[i])

    success_rate = (suc_count * 100) / (suc_count + fail_count)

    print("Success rate: " + str(success_rate) + "%")
    print("Possible Successful trades: " + str(suc_count))
    print("Possible trades: " + str(suc_count + fail_count))
    print("Possible profit: $" + str(prft))

    


strategy()