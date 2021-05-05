
#%% You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--train", default=True, help="training model or not")
    parser.add_argument("--valid", default=False, help="training model or not")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

args = config()

#%% Packages
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib
from sklearn import linear_model
import func

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
        gen_data, con_data = func._pack2(gen)[:-24, :], func._pack2(con)[:-24, :]
        gen_label, con_label = func._pack2(gen[7*24:], win=24), func._pack2(con[7*24:], win=24)
        exit()
        gen_data_n = func._norm(gen_data)
        con_data_n = func._norm(con_data)

        Gdata.append(gen_data_n)
        Cdata.append(con_data_n)
        Glabel.append(gen_label)
        Clabel.append(con_label)
    
    Gndata = np.vstack(Gdata)
    Cndata = np.vstack(Cdata)
    Glabel = np.vstack(Glabel)
    Clabel = np.vstack(Clabel)    
    
    Gmodel = linear_model.MultiTaskLasso(alpha=1e-2)
    Cmodel = linear_model.MultiTaskElasticNet(alpha=1e-4) 
    Gmodel.fit(Gndata, Glabel)
    Cmodel.fit(Cndata, Clabel)
    joblib.dump(Gmodel, 'Gmodel')
    joblib.dump(Cmodel, 'Cmodel')

elif args.valid:
#%% val pred
    Gmodel = joblib.load('Gmodel')
    Cmodel = joblib.load('Cmodel')
    # G_path = args.generation
    # C_path = args.consumption
    G_path = "./sample_data/generation_25.csv"
    C_path = "./sample_data/consumption_25.csv"   
    GVal = np.array(pd.read_csv(G_path, header=None))[1:,1:]
    GVal = np.stack(GVal).astype(None)[:,0]
    CVal = np.array(pd.read_csv(C_path, header=None))[1:,1:]
    CVal = np.stack(CVal).astype(None)[:,0] 
    
    GVdata, CVdata = GVal[:168][np.newaxis, :], CVal[:168][np.newaxis, :]
    GVlabel, CVlabel = GVal[168:168+24], CVal[168:168+24]

    GnVdata = func._norm(GVdata)
    CnVdata = func._norm(CVdata)

    Gpred = Gmodel.predict(GnVdata)
    Cpred = Cmodel.predict(CnVdata)
        

#%%
    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    ax.plot(Gpred[0,:], color='dodgerblue', label='Pred')
    ax.plot(GVlabel, color='darkorange', label='Label')
    ax.legend(fontsize=10, loc=4)
    plt.title('Generation', fontsize=30) 
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    ax.plot(Cpred[0,:], color='dodgerblue', label='Pred')
    ax.plot(CVlabel, color='darkorange', label='Label')
    ax.legend(fontsize=10, loc=4)
    plt.title('Consumption', fontsize=30) 
    plt.tight_layout()
    plt.show()  
   
else:
    Gmodel = joblib.load('Gmodel')
    Cmodel = joblib.load('Cmodel')
    G_path = args.generation
    C_path = args.consumption 
    GVal = np.array(pd.read_csv(G_path, header=None))[1:,1:]
    GVal = np.stack(GVal).astype(None)[:,0]
    CVal = np.array(pd.read_csv(C_path, header=None))[1:,1:]
    CVal = np.stack(CVal).astype(None)[:,0] 
    
    GVdata, CVdata = GVal[np.newaxis, :], CVal[np.newaxis, :]
    GVlabel, CVlabel = GVal, CVal

    GnVdata = func._norm(GVdata)
    CnVdata = func._norm(CVdata)

    Gpred = Gmodel.predict(GnVdata)
    Cpred = Cmodel.predict(CnVdata)   

    vol, act = func._comp(Gpred, Cpred)
    func._output(args.output, vol, act) 
