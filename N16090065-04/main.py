import time
tStart = time.time()
#%% You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

args = config()

#%% Packages
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras
import func

Gmodel = keras.models.load_model('Gmodel')
Cmodel = keras.models.load_model('Cmodel')
G_path = args.generation
C_path = args.consumption 
GVal = np.array(pd.read_csv(G_path, header=None))[1:,1:]
GVal = np.stack(GVal).astype(None)[:,0]
CVal = np.array(pd.read_csv(C_path, header=None))[1:,1:]
CVal = np.stack(CVal).astype(None)[:,0] 
date_pre = np.array(pd.read_csv(C_path, header=None))[-1,0]

GVdata, CVdata = GVal[np.newaxis, :], CVal[np.newaxis, :]
GVlabel, CVlabel = GVal, CVal

GnVdata = np.reshape(func._norm(GVdata), (-1,7,24))
CnVdata = np.reshape(func._norm(CVdata), (-1,7,24))

Gpred = Gmodel.predict(GnVdata)
Cpred = Cmodel.predict(CnVdata)   

vol, act = func._comp(Gpred, Cpred)
func._output(args.output, vol, act, date_pre, Gpred)
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))

