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
