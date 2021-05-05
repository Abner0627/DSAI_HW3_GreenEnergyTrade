import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#################################################################################
# DATA


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


name = 'consumption'

train_x, train_y = train_data(name)
test_x, test_y = test_data(name)


#################################################################################
# MODEL

model = keras.Sequential()

model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True), merge_mode='concat'))
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
    train_x, train_y, batch_size=512, epochs=1, verbose=1, shuffle=True
)
  

#################################################################################
# TEST


yhat = model.predict(train_x[-1:])
plt.plot(yhat.flatten())
plt.plot(train_y[-1].flatten(), '--')
plt.show()
exit()



yhat = model.predict(test_x)

print(yhat.shape)


predictions = yhat.flatten()


plt.plot(predictions)
# plt.plot(expecteds, '--')
plt.show()
