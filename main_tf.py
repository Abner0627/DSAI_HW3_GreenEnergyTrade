
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
    Glabel = np.sum(np.vstack(Glabel), axis=-1)[:, np.newaxis]
    Clabel = np.sum(np.vstack(Clabel), axis=-1)[:, np.newaxis]

    Gmodel = model.m03(128)
    Cmodel = model.m03(128)
    optim_G = keras.optimizers.SGD(learning_rate=1e-1)
    optim_C = keras.optimizers.SGD(learning_rate=1e-1)

    Gmodel.compile(optimizer=optim_G, loss=keras.losses.Huber())
    Cmodel.compile(optimizer=optim_C, loss=keras.losses.Huber())

    print("=====Gmodel=====")
    history_G = Gmodel.fit(Gndata, Glabel, batch_size=128, epochs=40, verbose=2, shuffle=True)
    print("=====Cmodel=====")
    history_C = Cmodel.fit(Cndata, Clabel, batch_size=128, epochs=40, verbose=2, shuffle=True)
    loss_G = np.array(history_G.history['loss'])
    loss_C = np.array(history_C.history['loss'])

    Gmodel.save('Gmodel')
    Cmodel.save('Cmodel')

    with open('loss.npy', 'wb') as f:          
        np.save(f, loss_G)
        np.save(f, loss_C)

else:
#%% val pred
    tStart = time.time()
    Gmodel = keras.models.load_model('Gmodel')
    Cmodel = keras.models.load_model('Cmodel')
    # G_path = args.generation
    # C_path = args.consumption
    G_path = "./sample_data/generation_25.csv"
    C_path = "./sample_data/consumption_25.csv"   
    GVal = np.array(pd.read_csv(G_path, header=None))[1:,1:]
    GVal = np.stack(GVal).astype(None)[:,0]
    CVal = np.array(pd.read_csv(C_path, header=None))[1:,1:]
    CVal = np.stack(CVal).astype(None)[:,0] 
    
    GVdata, CVdata = GVal[:168][np.newaxis, :], CVal[:168][np.newaxis, :]
    GVlabel, CVlabel = np.sum(GVal[168:168+24]), np.sum(CVal[168:168+24])

    GnVdata = np.reshape(func._norm(GVdata), (-1,7,24))
    CnVdata = np.reshape(func._norm(CVdata), (-1,7,24))

    Gpred = Gmodel.predict(GnVdata)
    Cpred = Cmodel.predict(CnVdata)
#%%   
    Gs = (Gpred-GVlabel)**2/2
    Cs = (Cpred-CVlabel)**2/2
    print("Gs >> ", Gs)
    print("Cs >> ", Cs)
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
