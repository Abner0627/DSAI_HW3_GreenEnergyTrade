'''
不太型
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from fbprophet import Prophet

# PARAM
predictNum = 1

#
df = pd.read_csv('./training_data/target0.csv')
df = df.rename(columns={'consumption': 'y','time':'ds'})



m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=predictNum)
forecast = m.predict(future)

fig1 = m.plot(forecast)
plt.show()

fig2 = m.plot_components(forecast)
plt.show()

# output = forecast[['yhat']][-predictNum:].to_numpy().flatten()


