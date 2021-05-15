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


def _output3(vol, act, date_pre, out_buy, out_sell):
    D = []
    date_pre = datetime.datetime.strptime(date_pre, "%Y-%m-%d %H:%M:%S")
    date = date_pre + datetime.timedelta(days=1)
    date = date.strftime("%Y-%m-%d")
    trad_v = (abs(np.round(vol, 2))).squeeze()
    ini_time = datetime.datetime.strptime("00:00:00", "%H:%M:%S")
    for i in range(len(act)):
        time = ini_time + datetime.timedelta(hours=i)
        if act[i]==-1:
            data = [date + " " + str(time.strftime("%H:%M:%S")), "sell", out_sell, float(trad_v[i])]
        elif act[i]==1:
            data = [date + " " + str(time.strftime("%H:%M:%S")), "buy", out_buy, float(trad_v[i])]
        else:
            data = [date + " " + str(time.strftime("%H:%M:%S")), "buy", 0, 0]
        D.append(data) 
      
    return D    

def _wavg(trade):
    cost = np.array(trade[:,4], dtype=np.float)
    vol = np.array(trade[:,5], dtype=np.float)
    cv = np.sum(cost*vol)
    w = np.sum(vol)
    out = cv/w
    return out

def _bid(path, Cnpy):
    with open(Cnpy, 'rb') as f:
        out_buy = np.load(f)
        out_sell = np.load(f)   

    A = np.array(pd.read_csv(path, header=None))
    val = len(A)

    if val==1:
        print('init')
        out_buy = 2.2
        out_sell = 2
    else:
        buy = A[A[:,1]=='buy']
        sell = A[A[:,1]=='sell']
        buy_trade = buy[buy[:,6]!='未成交']
        sell_trade = sell[sell[:,6]!='未成交']

        if len(buy_trade)<5:
            out_buy = out_buy + 0.01
            print("buy bid <5")
        else:
            out_buy = round(_wavg(buy_trade), 2)
            print("buy avg")
        if len(sell_trade)<5:
            out_sell = out_sell - 0.01
            print("sell bid <5")
        else:
            out_sell = round(_wavg(sell_trade), 2) - 0.25
            print("sell avg")     

        if out_buy>=2.4:
            out_buy = 2.4
            print("buy limit")
        if out_sell<=2:
            out_sell = 2
            print("sell limit")

    out_buy = round(out_buy, 2)
    out_sell = round(out_sell, 2)
    print("out_buy", out_buy)
    print("out_sell", out_sell)
    return out_buy, out_sell

#%% Test
if __name__ == "__main__":
    path = "../sample_data/bidresult-68.csv"
    Cnpy = 'cost.npy'
    out_buy, out_sell = _bid(path, Cnpy)
