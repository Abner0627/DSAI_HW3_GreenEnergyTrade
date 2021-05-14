import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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


name = 'consumption'

train_x, train_y = train_data(name)


#################################################################################
# MODEL

model = keras.Sequential()

model.add(layers.Bidirectional(layers.LSTM(
    256, return_sequences=True), merge_mode='concat'))
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
    train_x, train_y, batch_size=512, epochs=10, verbose=1, shuffle=True
)


#################################################################################
# TEST


yhat = model.predict(train_x[-1:])

plt.title(f'{name} 0-24hr')

plt.plot(yhat.flatten(), '-', label='predict')
plt.plot(train_y[-1].flatten(), '--', label='real')

plt.xticks(ticks=range(24), labels=[f'{i+1}' for i in range(24)])
plt.legend(loc=1)
plt.show()

model.save(f"{name}_model.h5")
