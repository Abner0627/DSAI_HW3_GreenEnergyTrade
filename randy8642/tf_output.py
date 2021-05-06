import numpy as np
import pandas as pd
import datetime
import tensorflow.keras as keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    df_consumption = pd.read_csv('./sample_data/consumption.csv')
    df_generation = pd.read_csv('./sample_data/generation.csv')

    init_date = datetime.datetime.strptime(
        df_consumption['time'].values[-1], "%Y-%m-%d %H:%M:%S")

    data = {
        'time': [],
        'action': [],
        'target_price': [],
        'target_volume': []
    }

    predict = get_predict()
    electric_delta = predict['generation'] - predict['consumption']

    for hour in range(24):
        # 00:00 - 23:00
        hours_added = datetime.timedelta(hours=hour+1)
        future_date_and_time = init_date + hours_added

        if electric_delta[hour] > 0:
            # SELL
            trend_time = future_date_and_time.strftime("%Y-%m-%d %H:%M:%S")
            target_price = 4.9
            target_volume = np.round(np.abs(electric_delta[hour]), decimals=2)
            data['time'].append(trend_time)
            data['action'].append('sell')
            data['target_price'].append(str(target_price))
            data['target_volume'].append(str(target_volume))
            pass
        elif electric_delta[hour] < 0:
            # BUY
            trend_time = future_date_and_time.strftime("%Y-%m-%d %H:%M:%S")
            target_price = 4.9
            target_volume = np.round(np.abs(electric_delta[hour]), decimals=2)
            data['time'].append(trend_time)
            data['action'].append('buy')
            data['target_price'].append(str(target_price))
            data['target_volume'].append(str(target_volume))
            pass
        else:
            pass

    df = pd.DataFrame.from_dict(data).to_csv('./output.csv')


def get_predict():

    def test_data(name: str):
        assert name in ['generation', 'consumption'], 'input name not in list'

        testing_data_path = f'./sample_data/{name}.csv'

        df = pd.read_csv(testing_data_path)
        data = df[name].values
        length = len(df.index)

        test_x = np.zeros([1, 24, 7])
        test_y = np.zeros([1, 24, 1])

        test_x = data.reshape(1, 24, 7)
        # test_y[i, :, :] = data[i+7*24:i+8*24].reshape(1, 24, 1)

        return test_x, test_y

    yhat = {}

    for name in ['consumption', 'generation']:
        # LSTM
        path = f'./model_save/{name}_model.h5'
        model = keras.models.load_model(path)

        test_x, test_y = test_data(name)

        yhat[name] = model.predict(test_x).flatten()

    return yhat


if __name__ == '__main__':
    main()
