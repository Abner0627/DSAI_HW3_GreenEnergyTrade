import pandas as pd
import os
from termcolor import colored

# PARAM
PATH = {
    'consumption': './sample_data/consumption.csv',  # 過去七天歷史用電資料
    'generation': './sample_data/generation.csv',  # 過去七天產電資料
    'bidresult': './sample_data/bidresult.csv',  # 過去七天自己的投標資料
    'output': './sample_data/output.csv',  # 　未來一天投標資訊
    'train': './training_data',  # train
}

for dirPath, dirNames, fileNames in os.walk(PATH['train']):
    for f in fileNames:
        fullPath = os.path.join(dirPath, f)
        
        df = pd.read_csv(fullPath)

        diff = (df['generation'] - df['consumption']).values

        for d in diff:
            if d < 0:
                # ACTION = BUY
                print(colored(d,'red'))
            elif d > 0:
                # ACTION = SOLD
                print(colored(d,'green'))
            else:
                # ACTION = NONE
                print(d)

        
        exit()


