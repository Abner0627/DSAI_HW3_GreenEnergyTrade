import os
import numpy as np
from numpy.core.fromnumeric import mean
import pandas as pd
import matplotlib.pyplot as plt


def get_data(name: str):

    x = np.zeros([0, 24])

    training_data_path = './training_data'
    for dirPath, dirNames, fileNames in os.walk(training_data_path):
        for f in fileNames:
            fullPath = os.path.join(dirPath, f)

            df = pd.read_csv(fullPath)
            length = len(df.index)
            data = df[name].values

            x_target = np.zeros([length//24+1, 24])

            for i in range(24, length, 24):

                x_target[i//24, :] = data[i-24:i].reshape(1, -1)

            x = np.concatenate((x, x_target), axis=0)

    return x


name = 'generation'

x = get_data(name)

print(x.shape)

train_mean = np.mean(x, axis=0)
train_std = np.std(x, axis=0)

plt.figure(figsize=(10,6))
plt.title(name)
plt.plot(range(24), train_mean, label='mean', color='orange')
plt.fill_between(range(24), train_mean-train_std, train_mean+train_std, alpha=0.3, label='std_range', color='orange')

plt.legend(loc=1)
plt.show()
