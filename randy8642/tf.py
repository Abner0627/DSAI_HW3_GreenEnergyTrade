import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#################################################################################
# DATA


def train_data(name: str):

    train_x = np.zeros([0, 7*24, 1])
    train_y = np.zeros([0, 1])

    training_data_path = './training_data'
    for dirPath, dirNames, fileNames in os.walk(training_data_path):
        for f in fileNames:
            fullPath = os.path.join(dirPath, f)

            df = pd.read_csv(fullPath)
            length = len(df.index)
            data = df[name].values

            train_x_target = np.zeros([length - 7*24, 7*24, 1])
            train_y_target = np.zeros([length - 7*24, 1*24])

            for i in range(length - 7):
                train_x_target[i, :, 0] = data[i:i+6*24]
                train_y_target[i, :] = data[i+6*24:i+7*24]


            train_x = np.concatenate((train_x, train_x_target), axis=0)
            train_y = np.concatenate((train_y, train_y_target), axis=0)

    return train_x, train_y


def test_data(name: str):
    assert name in ['generation', 'consumption'], 'input name not in list'

    testing_data_path = f'./sample_data/{name}.csv'

    df = pd.read_csv(testing_data_path)
    data = df[name].values
    length = len(df.index)

    test_x = np.zeros([length - 7, 7, 1])
    test_y = np.zeros([length - 7, 1])

    for i in range(length - 7):
        test_x[i, :, 0] = data[i:i+7]
        test_y[i, :] = data[i+7]

    return test_x, test_y


data = {}
for stage in ['train', 'test']:
    data[stage] = {}
    for itemName in ['generation', 'consumption']:
        data[stage][itemName] = {}
        for label in ['x', 'y']:
            data[stage][itemName][label] = None

name = 'consumption'

x, y = train_data(name)
data['train'][name]['x'] = x
data['train'][name]['y'] = y
del x, y
x, y = test_data(name)
data['test'][name]['x'] = x
data['test'][name]['y'] = y
del x, y

#################################################################################
# MODEL

model = keras.Sequential()

model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=False), merge_mode='concat'))
model.add(layers.Dense(12))
model.add(layers.Dense(1))

# model.summary()


#################################################################################
# TRAIN

model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer="adam",
    metrics=["accuracy"],
)



model.fit(
    data['train'][name]['x'], data['train'][name]['y'], batch_size=512, epochs=10, verbose=1, shuffle=False
)
  

#################################################################################
# TEST
model.reset_states()
model.predict(data['train'][name]['x'], batch_size=512)

predictions = []
expecteds = []
for i in range(len(data['test'][name]['x'])):
    # make one-step forecast
    X = data['test'][name]['x'][i]
    X = X.reshape(1, 7, 1)
    yhat = model.predict(X)[0,0]
    
    # store forecast
    predictions.append(yhat)
    expecteds.append(data['test'][name]['y'][i,0])
    



plt.plot(predictions)
plt.plot(expecteds, '--')
plt.show()
