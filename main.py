
#%% You should not modify this part.
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
seed_v = 25
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import tensorflow.keras as keras
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
    
    Gndata = np.vstack(Gdata)
    Cndata = np.vstack(Cdata)
    Glabel = np.vstack(Glabel)
    Clabel = np.vstack(Clabel)    
    
    Gmodel = model.m02(128)
    Cmodel = model.m02(128)
    optim_G = keras.optimizers.Adam(learning_rate=1e-3)
    optim_C = keras.optimizers.Adam(learning_rate=1e-3)

    Gmodel.compile(optimizer=optim_G, loss='mse')
    Cmodel.compile(optimizer=optim_C, loss='mse')

    print("=====Gmodel=====")
    history_G = Gmodel.fit(Gndata, Glabel, batch_size=32, epochs=40, verbose=2, shuffle=True)
    print("=====Cmodel=====")
    history_C = Cmodel.fit(Cndata, Clabel, batch_size=32, epochs=40, verbose=2, shuffle=True)
    loss_G = np.array(history_G.history['loss'])
    loss_C = np.array(history_C.history['loss'])

    Gmodel.save('Gmodel')
    Cmodel.save('Cmodel')

    with open('loss.npy', 'wb') as f:          
        np.save(f, loss_G)
        np.save(f, loss_C)

else:
#%% val pred
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
    GVlabel, CVlabel = GVal[168:168+24][np.newaxis, :], CVal[168:168+24][np.newaxis, :]

    GnVdata = func._norm(GVdata)
    CnVdata = func._norm(CVdata)

    Gpred = Gmodel.predict(GVdata)
    Cpred = Cmodel.predict(CVdata)
        

#%%
    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    ax.plot(Gpred[0,:], color='dodgerblue', label='Pred')
    ax.plot(GVlabel[0,:], color='darkorange', label='Label')
    ax.legend(fontsize=10, loc=4)
    plt.title('Generation', fontsize=30) 
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize = (15,5))
    ax.plot(Cpred[0,:], color='dodgerblue', label='Pred')
    ax.plot(CVlabel[0,:], color='darkorange', label='Label')
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
# %%
