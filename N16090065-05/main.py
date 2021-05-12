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
import joblib
from sklearn import linear_model
import func

model = joblib.load('model')
G_path = args.generation
C_path = args.consumption 
GVal = np.array(pd.read_csv(G_path, header=None))[1:,1:]
GVal = np.stack(GVal).astype(None)[:,0]
CVal = np.array(pd.read_csv(C_path, header=None))[1:,1:]
CVal = np.stack(CVal).astype(None)[:,0] 
date_pre = np.array(pd.read_csv(C_path, header=None))[-1,0]

GVdata, CVdata = GVal[np.newaxis, :], CVal[np.newaxis, :]
nVdata = np.concatenate((GVdata, CVdata), axis=1)

pred = model.predict(nVdata) 

act = func._comp2(pred)
D = func._output2(pred, act, date_pre)
df = pd.DataFrame(D, columns=["time", "action", "target_price", "target_volume"])
df.to_csv(args.output, index=False)
tEnd = time.time()
print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))

