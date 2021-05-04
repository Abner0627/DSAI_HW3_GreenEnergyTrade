#%% Import packages
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

#%% Sturcture
class m01(nn.Module):
    def __init__(self, in_sz, out_sz, tap, hid, bid=False):
        super(m01, self).__init__()
        self.GRU = nn.GRU(in_sz, out_sz, hid, bidirectional=bid)

        if bid:
            sz = int(out_sz*2)
        else:
            sz = out_sz

        self.Conv = nn.Sequential(
            nn.Conv1d(tap, 8, kernel_size=7),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=7, dilation=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=1)            
        )
        self.FC_gen = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),            
            nn.Linear(16, 1)
        )
        self.FC_con = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),            
            nn.Linear(16, 1)
        )        
    def forward(self, x):
        x_GRU, hn = self.GRU(x)
        xtt = self.Conv(x_GRU)
        print(xtt.size())
        y_gen = self.FC_gen(xtt)
        y_con = self.FC_con(xtt)
        return y_gen, y_con

class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.layer = keras.Sequential([
            keras.layers.Conv2D(filters=6, kernel_size=(5, 5)),
            keras.layers.Activation(tf.nn.sigmoid),
            keras.layers.AveragePooling2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),
            keras.layers.Conv2D(filters=6, kernel_size=(5, 5)),
            keras.layers.Activation(tf.nn.sigmoid),
            keras.layers.AveragePooling2D(pool_size=(
                2, 2), strides=(2, 2), padding='valid'),

            keras.layers.Flatten(),
            keras.layers.Dense(120),
            keras.layers.Activation(tf.nn.sigmoid),
            keras.layers.Dense(84),
            keras.layers.Activation(tf.nn.sigmoid),
            keras.layers.Dense(43)
        ])

    def call(self, inputs):
        x = self.layer(inputs)

        return x

#%% Test
if __name__ == "__main__":
    IN = torch.randn(32,7,2)
    F = m01(2, 24, 7, hid=3, bid=True)
    Gen, Con = F(IN)
    print('Gen >>', Gen.size())
    print('Con >>', Con.size())

    # 預測漲/跌