import numpy as np
import datetime
import pandas as pd

def _pack(x, win=7*24):
    out = []
    leng = len(x)
    for i in range(leng):
        if i+win>leng:
            break
        else:
            wind = (x[i:i+win])[np.newaxis, :]
            out.append(wind)
    out = np.vstack(out)
    return out

def _norm(x, Z=True):
    x_n = np.zeros_like(x)
    for i in range(x.shape[0]):
        x_mu = np.mean(x[i,:], axis=0)
        if Z:
            x_std = np.std(x[i,:], axis=0)
            x_n[i,:] = (x[i,:]-x_mu)/x_std
        else:
            x_min = np.min(x[i,:], axis=0)
            x_max = np.max(x[i,:], axis=0)
            x_n[i,:] = (x[i,:]-x_mu)/(x_max-x_min)
    return x_n

def _comp(Gpred, Cpred):
    G = np.sum(Gpred)
    C = np.sum(Cpred)
    profit = G - C
    if profit<0:
        act = 1
        vol = abs(profit)
    elif profit>0:
        act = -1
        vol = profit
    else:
        act = 0
        vol = 0
    vol = round(vol, 1)
    return vol, act

def _output(path, vol, act):
    date = datetime.datetime.now()+datetime.timedelta(days=1)
    date = date.strftime("%Y-%m-%d")
    if act==-1:
        data = [[date+" 03:00:00", "sell", 2.0, vol-2]]
    elif act==1:
        data = [[date+" 03:00:00", "buy", 2.5, vol-2]]
    else:
        data = [[date+" 03:00:00", "buy", 0, 0]]

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)
    return