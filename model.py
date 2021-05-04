#%% Import packages
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

#%% Sturcture
class m01(keras.Model):
    def __init__(self):
        super(m01, self).__init__()
        self.CvLSTM = keras.layers.ConvLSTM2D(filters=16, kernel_size=3, strides=(1, 1), padding='valid', activation='tanh', recurrent_activation='relu')

    def call(self, x):
        y = self.CvLSTM(x)
        return y

#%% Test
if __name__ == "__main__":
    IN = np.random.rand((32,168))
    F = m01()
    Gen = F(IN)
    print('Gen >>', Gen.shape)

    # 預測漲/跌