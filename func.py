import numpy as np

def _pack(x, win=7*24):
    leng = len(x)
    for i in range(leng):
        if i+win>leng:
            break
        else:
            wind = (x[i:i+win])[np.newaxis, :]
            if i==0:
                out = wind
            else:
                out = np.concatenate((out, wind))
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

