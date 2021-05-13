#%% Import packages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

#%% Sturcture
# class m01(keras.Model):
#     def __init__(self, out_sz):
#         super(m01, self).__init__()
#         self.GRU = keras.layers.GRU(out_sz)
#         self.Cv = keras.Sequential([
#             keras.layers.Conv1D(8, kernel_size=5),
#             keras.layers.BatchNormalization(),
#             keras.layers.ReLU(),
#             keras.layers.Conv1D(16, kernel_size=5, dilation_rate=3),
#             keras.layers.BatchNormalization(),
#             keras.layers.ReLU(),    
#             keras.layers.Conv1D(8, kernel_size=1)
#         ])
#         self.FC = keras.Sequential([
#             keras.layers.Flatten(),
#             keras.layers.Dense(32),
#             keras.layers.ReLU(),
#             keras.layers.Dense(24)
#         ])

#     def call(self, x):
#         x = tf.reshape(x, (-1, 7, 24))
#         y1 = self.GRU(x)
#         y1 = tf.reshape(y1, (-1, 24, 1))
#         y2 = self.Cv(y1)
#         y = self.FC(y2)
#         return y

# class m02(keras.Model):
#     def __init__(self, out_sz):
#         super(m02, self).__init__()
#         FL = keras.layers.GRU(out_sz)
#         BL = keras.layers.GRU(out_sz, go_backwards=True)
#         self.GRU = keras.layers.Bidirectional(FL, backward_layer=BL)
#         self.FC = keras.Sequential([
#             keras.layers.Flatten(),
#             keras.layers.Dense(64),
#             keras.layers.ReLU(),
#             keras.layers.Dense(24)
#         ])

#     def call(self, x):
#         x = tf.reshape(x, (-1, 7, 24))
#         y1 = self.GRU(x)
#         y = self.FC(y1)
#         return y 

class m03(keras.Model):
    def __init__(self, out_sz):
        super(m03, self).__init__()
        FL = keras.layers.GRU(out_sz)
        BL = keras.layers.GRU(out_sz, go_backwards=True)
        self.GRU = keras.layers.Bidirectional(FL, backward_layer=BL)
        self.FC = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(64),
            keras.layers.ReLU(),
            keras.layers.Dense(1)
        ])

    def call(self, x):
        y1 = self.GRU(x)
        y = self.FC(y1)
        return y   

class m04(keras.Model):
    def __init__(self, out_sz):
        super(m04, self).__init__()
        FL = keras.layers.GRU(out_sz, return_sequences=True)
        BL = keras.layers.GRU(out_sz, go_backwards=True, return_sequences=True)
        FL2 = keras.layers.GRU(out_sz//2, return_sequences=True)
        BL2 = keras.layers.GRU(out_sz//2, go_backwards=True, return_sequences=True)        
        self.GRU = keras.layers.Bidirectional(FL, backward_layer=BL)
        self.GRU2 = keras.layers.Bidirectional(FL2, backward_layer=BL2)
        self.Cv = keras.Sequential([
            keras.layers.Conv1D(32, kernel_size=5),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),   
            keras.layers.Conv1D(16, kernel_size=1)
        ])        
        self.FC = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(256),
            keras.layers.ReLU(),            
            keras.layers.Dense(64),
            keras.layers.ReLU(),
            keras.layers.Dense(24)
        ])       

    def call(self, x):
        y1 = self.GRU(x)
        y2 = self.GRU2(y1)
        y3 = tf.transpose(y2, perm=[0, 2, 1])
        y4 = self.Cv(y3)
        y = self.FC(y4)
        return y              

#%% Test
if __name__ == "__main__":
    IN = np.random.rand(1,7,48)
    F = m04(128)
    Gen = F(IN)
    print('Gen >>', Gen.shape)

    # 預測漲/跌