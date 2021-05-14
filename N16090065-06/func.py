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

def _pack2(x, win=7*24):
    out = []
    leng = len(x)
    for i in range(leng):
        if i==0:
            r, c = 0, win
        else:
            r = i*win+1
            c = r+win
        if r>leng or c>leng:
            break
        else:
            wind = (x[r:c])[np.newaxis, :]
            out.append(wind)
    out = np.vstack(out)
    return out    

def _norm(x, Z=True):
    x_n = np.zeros_like(x)
    for i in range(x.shape[0]):
        xs = x[i,:]
        x_mu = np.mean(xs)
        if Z:
            x_std = np.std(xs)
            x_n[i,:] = (xs-x_mu)/x_std
        else:
            x_min = np.min(xs)
            x_max = np.max(xs)
            x_n[i,:] = (xs-x_mu)/(x_max-x_min)
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

def _comp2(pred):
    A = []
    for i in range(24):
        if pred[:, i]<0:
            A.append(1)
        elif pred[:, i]>0:
            A.append(-1)
        else:
            A.append(0)
    return np.vstack(A)

def _output2(vol, act, date_pre):
    D = []
    date_pre = datetime.datetime.strptime(date_pre, "%Y-%m-%d %H:%M:%S")
    date = date_pre + datetime.timedelta(days=1)
    date = date.strftime("%Y-%m-%d")
    trad_v = (abs(np.round(vol, 2))).squeeze()
    ini_time = datetime.datetime.strptime("00:00:00", "%H:%M:%S")
    for i in range(len(act)):
        time = ini_time + datetime.timedelta(hours=i)
        if act[i]==-1:
            data = [date + " " + str(time.strftime("%H:%M:%S")), "sell", 2.0, float(trad_v[i])]
        elif act[i]==1:
            data = [date + " " + str(time.strftime("%H:%M:%S")), "buy", 2.2, float(trad_v[i])]
        else:
            data = [date + " " + str(time.strftime("%H:%M:%S")), "buy", 0, 0]
        D.append(data) 
      
    return D



def _output(path, vol, act, date_pre, Gpred):
    date_pre = datetime.datetime.strptime(date_pre, "%Y-%m-%d %H:%M:%S")
    G = np.sum(Gpred)
    date = date_pre + datetime.timedelta(days=1)
    date = date.strftime("%Y-%m-%d")

    trad_v = round(vol, 2)

    if trad_v<0:
        trad_v = 0
    
    if act==-1:
        data = [[date+" 03:00:00", "sell", 2.0, trad_v]]
    elif act==1:
        data = [[date+" 03:00:00", "buy", 2.2, trad_v]]
    else:
        data = [[date+" 03:00:00", "buy", 0, 0]]

    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)
    return