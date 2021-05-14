
#%% You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--train", default=False, help="training model or not")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

args = config()

#%% Packages
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import tensorflow.keras as keras
import time
import func
import model

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")

#%% Load
if args.train:
    dpath = './training_data'
    data_list = os.listdir(dpath)
    Ndata, Nlabel = [], []
    for i in range(len(data_list)):
    # i=0
        data_ag = np.array(pd.read_csv(os.path.join(dpath, data_list[i]), header=None))[1:,1:]
        ag = np.stack(data_ag).astype(None)
        gen, con = ag[:,0], ag[:,1]
        gen_data, con_data = func._pack(gen)[:-24, :], func._pack(con)[:-24, :]
        gen_label, con_label = func._pack(gen[7*24:], win=24), func._pack(con[7*24:], win=24)

        t_data = np.concatenate((gen_data, con_data), axis=-1)
        t_data_n = func._norm(t_data)
        t_label = gen_label - con_label

        Ndata.append(t_data_n)
        Nlabel.append(t_label)
    
    ndata = np.reshape(np.vstack(Ndata), (-1,7,48))
    label = np.vstack(Nlabel)
    # Glabel2 = np.sum(np.vstack(Glabel), axis=-1)[:, np.newaxis]
    # Clabel2 = np.sum(np.vstack(Clabel), axis=-1)[:, np.newaxis]

    model = model.m04(128)
    optim = keras.optimizers.SGD(learning_rate=1e-1)
    model.compile(optimizer=optim, loss=keras.losses.MeanSquaredError())

    print("=====model=====")
    history = model.fit(ndata, label, batch_size=32, epochs=40, verbose=1, shuffle=True)
    loss = np.array(history.history['loss'])

    model.save('model')

    with open('loss.npy', 'wb') as f:          
        np.save(f, loss)

else:
#%% val pred
    tStart = time.time()
    model = keras.models.load_model('model')
    # Cmodel = keras.models.load_model('Cmodel')
    # G_path = args.generation
    # C_path = args.consumption
    G_path = "./sample_data/generation_25.csv"
    C_path = "./sample_data/consumption_25.csv"   
    GVal = np.array(pd.read_csv(G_path, header=None))[1:,1:]
    GVal = np.stack(GVal).astype(None)[:,0]
    CVal = np.array(pd.read_csv(C_path, header=None))[1:,1:]
    CVal = np.stack(CVal).astype(None)[:,0] 

    GVdata, CVdata = GVal[:168][np.newaxis, :], CVal[:168][np.newaxis, :]
    label = GVal[168:168+24] - CVal[168:168+24]
    # GVlabel, CVlabel = np.sum(GVal[168:168+24]), np.sum(CVal[168:168+24])
    tVdata = np.concatenate((GVdata, CVdata), axis=-1)
    nVdata = np.reshape(func._norm(tVdata), (-1,7,48))

    pred = model.predict(nVdata)
#%%   
    Gs = rmse(pred, label)
    print("RMSE >> ", Gs)
    tEnd = time.time()
    print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))

#%%
    # fig, ax = plt.subplots(1, 1, figsize = (15,5))
    # ax.plot(Gpred[0,:], color='dodgerblue', label='Pred')
    # ax.plot(GVlabel, color='darkorange', label='Label')
    # ax.legend(fontsize=10, loc=4)
    # plt.title('Generation', fontsize=30) 
    # plt.tight_layout()

    # fig, ax = plt.subplots(1, 1, figsize = (15,5))
    # ax.plot(Cpred[0,:], color='dodgerblue', label='Pred')
    # ax.plot(CVlabel, color='darkorange', label='Label')
    # ax.legend(fontsize=10, loc=4)
    # plt.title('Consumption', fontsize=30) 
    # plt.tight_layout()
    # plt.show()  
   
# else:
#     Gmodel = keras.models.load_model('Gmodel')
#     Cmodel = keras.models.load_model('Cmodel')
#     G_path = args.generation
#     C_path = args.consumption 
#     GVal = np.array(pd.read_csv(G_path, header=None))[1:,1:]
#     GVal = np.stack(GVal).astype(None)[:,0]
#     CVal = np.array(pd.read_csv(C_path, header=None))[1:,1:]
#     CVal = np.stack(CVal).astype(None)[:,0] 
#     date_pre = np.array(pd.read_csv(C_path, header=None))[1:,-1]
    
#     GVdata, CVdata = GVal[np.newaxis, :], CVal[np.newaxis, :]
#     GVlabel, CVlabel = GVal, CVal

#     GnVdata = np.reshape((GVdata), (-1,7,24))
#     CnVdata = np.reshape((CVdata), (-1,7,24))

#     Gpred = Gmodel.predict(GnVdata)
#     Cpred = Cmodel.predict(CnVdata)   

#     vol, act = func._comp(Gpred, Cpred)
#     func._output(args.output, vol, act, date_pre) 
