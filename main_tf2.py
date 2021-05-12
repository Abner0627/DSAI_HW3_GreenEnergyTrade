
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
import tensorflow.keras as keras
import time
import func
import model

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#%% Load
if args.train:
    dpath = './training_data'
    data_list = os.listdir(dpath)
    Gdata, Cdata, Glabel, Clabel = [], [], [], []
    for i in range(len(data_list)):
    # i=0
        data_ag = np.array(pd.read_csv(os.path.join(dpath, data_list[i]), header=None))[1:,1:]
        ag = np.stack(data_ag).astype(None)
        gen, con = ag[:,0], ag[:,1]
        gen_data, con_data = func._pack(gen)[:-24, :], func._pack(con)[:-24, :]
        gen_label, con_label = func._pack(gen[7*24:], win=24), func._pack(con[7*24:], win=24)

        gen_data_n = func._norm(gen_data)
        con_data_n = func._norm(con_data)

        Gdata.append(gen_data_n)
        Cdata.append(con_data_n)
        Glabel.append(gen_label)
        Clabel.append(con_label)
    
    Gndata = np.reshape(np.vstack(Gdata), (-1,7,24))
    Cndata = np.reshape(np.vstack(Cdata), (-1,7,24))
    ndata = np.concatenate((Gndata, Cndata), axis=1)
    label = (np.vstack(Glabel) - np.vstack(Clabel))[:, np.newaxis]
    # Glabel2 = np.sum(np.vstack(Glabel), axis=-1)[:, np.newaxis]
    # Clabel2 = np.sum(np.vstack(Clabel), axis=-1)[:, np.newaxis]

    model = model.m04(128)
    optim = keras.optimizers.SGD(learning_rate=1e-1)
    model.compile(optimizer=optim, loss=keras.losses.Huber())

    print("=====model=====")
    history = model.fit(ndata, label, batch_size=128, epochs=40, verbose=2, shuffle=True)
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
    GnVdata = np.reshape(func._norm(GVdata), (-1,7,24))
    CnVdata = np.reshape(func._norm(CVdata), (-1,7,24))
    nVdata = np.concatenate((GnVdata, CnVdata), axis=1)

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

#     GnVdata = np.reshape(func._norm(GVdata), (-1,7,24))
#     CnVdata = np.reshape(func._norm(CVdata), (-1,7,24))

#     Gpred = Gmodel.predict(GnVdata)
#     Cpred = Cmodel.predict(CnVdata)   

#     vol, act = func._comp(Gpred, Cpred)
#     func._output(args.output, vol, act, date_pre) 
