import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras as keras



def test_data(name: str):
    assert name in ['generation', 'consumption'], 'input name not in list'

    testing_data_path = f'./sample_data/{name}.csv'

    df = pd.read_csv(testing_data_path)
    data = df[name].values
    length = len(df.index)

    test_x = np.zeros([1, 24, 7])
    test_y = np.zeros([1, 24, 1])

    test_x = data.reshape(1, 24, 7)
    # test_y[i, :, :] = data[i+7*24:i+8*24].reshape(1, 24, 1)

    return test_x, test_y
def train_data(name: str):

    train_x = np.zeros([0, 24, 7])
    train_y = np.zeros([0, 24, 1])

    training_data_path = './training_data'
    for dirPath, dirNames, fileNames in os.walk(training_data_path):
        for f in fileNames:
            fullPath = os.path.join(dirPath, f)

            df = pd.read_csv(fullPath)
            length = len(df.index)
            data = df[name].values

            train_x_target = np.zeros([length - 8*24 + 1, 24, 7])
            train_y_target = np.zeros([length - 8*24 + 1, 24, 1])

            for i in range(length - 8*24 + 1):
                train_x_target[i, :, :] = data[i:i+7*24].reshape(1, 24, 7)
                train_y_target[i, :, :] = data[i+7*24:i+8*24].reshape(1, 24, 1)


            train_x = np.concatenate((train_x, train_x_target), axis=0)
            train_y = np.concatenate((train_y, train_y_target), axis=0)

    return train_x, train_y

yhat = {}

for name in ['consumption', 'generation']:
    # LSTM
    path = f'./model_save/{name}_model.h5'
    model = keras.models.load_model(path)

    test_x, test_y = test_data(name)
    # test_x, test_y = train_data(name)
    # test_x,test_y = test_x[-1:],test_y[-1:]

    yhat[name] = model.predict(test_x).flatten()

    
    
    

# PLOT
plt.plot(yhat['generation'], label='predict_gen')
# plt.plot(y['generation'], '--', label='actual_gen')

plt.plot(yhat['consumption'], label='predict_con')
# plt.plot(y['consumption'], '--', label='actual_con')

plt.plot(yhat['generation'] - yhat['consumption'], label='predict_diff')
# plt.plot(y['generation'] - y['consumption'],'--' , label='actual_diff')

plt.legend(loc=1)
plt.show()
