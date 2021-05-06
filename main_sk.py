
#%% You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--train", default=False, help="training model or not")
    parser.add_argument("--valid", default=False, help="training model or not")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

args = config()

#%% Packages
import numpy as np
import pandas as pd
import os
import time
import joblib
from sklearn import linear_model
import func

#%% Load
if args.train:
    tStart = time.time()
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
    
    Gndata = np.vstack(Gdata)
    Cndata = np.vstack(Cdata)
    Glabel = np.sum(np.vstack(Glabel), axis=-1)[:, np.newaxis]
    Clabel = np.sum(np.vstack(Clabel), axis=-1)[:, np.newaxis]
    
    Gmodel = linear_model.Lasso(alpha=1e-2)
    Cmodel = linear_model.ElasticNet(alpha=1e-2) 
    Gmodel.fit(Gndata, Glabel)
    Cmodel.fit(Cndata, Clabel)
    joblib.dump(Gmodel, 'Gmodel')
    joblib.dump(Cmodel, 'Cmodel')

    tEnd = time.time()
    print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))

elif args.valid:
    tStart = time.time()
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
    GVlabel, CVlabel = np.sum(GVal[168:168+24]), np.sum(CVal[168:168+24])

    GnVdata = func._norm(GVdata)
    CnVdata = func._norm(CVdata)

    Gpred = Gmodel.predict(GnVdata)
    Cpred = Cmodel.predict(CnVdata)

    Gs = (Gpred-GVlabel)**2/2
    Cs = (Cpred-CVlabel)**2/2
    print("Gs >> ", Gs)
    print("Cs >> ", Cs)
    tEnd = time.time()
    print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))


