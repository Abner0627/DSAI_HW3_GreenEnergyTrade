
# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--train", default=True, help="training model or not")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

args = config()

#%% Packages
import numpy as np
import pandas as pd
import os
import random
import model
import torch
import torch.optim as optim
import torch.nn as nn

#%% Load
if args.train:
    dpath = '../training_data'
    data_list = os.listdir(dpath)
    for i in range(len(data_list)):
        data_ag = np.array(pd.read_csv(os.path.join(dpath, data_list[i]), header=None))
        if i == 0:
            ag = data_ag[1:,:]
        else:
            ag = np.concatenate((ag, data_ag[1:,:]))
        print(ag.shape)

'''
df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
df.to_csv(path, index=False)



if __name__ == "__main__":

    data = [["2018-01-01 00:00:00", "buy", 2.5, 3],
            ["2018-01-01 01:00:00", "sell", 3, 5]]
    output(args.output, data)
'''