
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
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
import random
import func


#%% Load
if args.train:
    dpath = './training_data'
    data_list = os.listdir(dpath)
    for i in range(len(data_list)):
    # i=0
        data_ag = np.array(pd.read_csv(os.path.join(dpath, data_list[i]), header=None))[1:,1:]
        ag = np.stack(data_ag).astype(None)
        gen, con = ag[:,0], ag[:,1]
        gen_data, con_data = func._pack(gen), func._pack(con)
        gen_label, con_label = func._label(gen), func._label(con)

        if i==0:
            Gdata, Cdata = gen_data, con_data
            Glabel, Clabel = gen_label, con_label
        else:
            Gdata, Cdata = np.concatenate((Gdata, gen_data)), np.concatenate((Cdata, con_data))
            Glabel, Clabel = np.concatenate((Glabel, gen_label)), np.concatenate((Clabel, con_label))

    Gndata, Gmu, Gstd = func._nor(Gdata, train=True)
    Cndata, Cmu, Cstd = func._nor(Cdata, train=True)

    Gmodel = linear_model.Lasso(alpha=1e-2, positive=True)
    Cmodel = linear_model.ElasticNet(alpha=1e-4, positive=True)
    Gmodel.fit(Gndata, Glabel)
    Cmodel.fit(Cndata, Clabel)


    #%% val pred
    Cp = './sample_data/consumption_org.csv'
    Gp = './sample_data/generation_org.csv'
    GVal = np.array(pd.read_csv(Gp, header=None))[1:,1:]
    GVal = np.stack(GVal).astype(None)[:,0]
    CVal = np.array(pd.read_csv(Cp, header=None))[1:,1:]
    CVal = np.stack(CVal).astype(None)[:,0]    
    GVdata, CVdata = func._pack(GVal), func._pack(CVal)
    GVlabel, CVlabel = func._label(GVal), func._label(CVal)

    GnVdata, _, _ = func._nor(GVdata, Gmu, Gstd)
    CnVdata, _, _ = func._nor(CVdata, Cmu, Cstd)

    Gpred = Gmodel.predict(GnVdata)
    Cpred = Cmodel.predict(CnVdata)

    #%%
    # fig, ax = plt.subplots(1, 1, figsize = (15,5))
    # ax.plot(Gpred, color='dodgerblue', label='Pred')
    # ax.plot(GVlabel, color='darkorange', label='Label')
    # ax.legend(fontsize=10, loc=4)
    # plt.title('Generation', fontsize=30) 
    # plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    ax.plot(Cpred, color='dodgerblue', label='Pred')
    ax.plot(CVlabel, color='darkorange', label='Label')
    ax.legend(fontsize=10, loc=4)
    plt.title('Consumption', fontsize=30) 
    plt.tight_layout()
    plt.show()    
   

    


'''
df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
df.to_csv(path, index=False)



if __name__ == "__main__":

    data = [["2018-01-01 00:00:00", "buy", 2.5, 3],
            ["2018-01-01 01:00:00", "sell", 3, 5]]
    output(args.output, data)
'''