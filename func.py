import numpy as np

def _pack(x, day=7):
    leng = len(x)
    for i in range(leng):
        if i+day>leng:
            break
        else:
            wind = (x[i:i+day])[np.newaxis, :]
            if i==0:
                out = wind
            else:
                out = np.concatenate((out, wind))
    return out[:-1,:]

def _label(x, day=7):
    return x[day:]

def _nor(x, x_mean=None, x_std=None, train=False):
    if train:
        x_mean = np.mean(x)
        x_std = np.std(x)
        x_nor = (x-x_mean) / x_std
    else:
        x_nor = (x-x_mean) / x_std
    return x_nor, x_mean, x_std

def _comp(GVal, CVal):
    if GVal>CVal:
        # In
        num = GVal - CVal
        act = -1
    elif GVal<CVal:
        # OUt
        num = CVal - GVal 
        act = 1
    else:
        num = 0
        act = 0
    return num, act

