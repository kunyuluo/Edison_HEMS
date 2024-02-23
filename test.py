import pandas as pd
import numpy as np
import datetime as dt
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from Helper import DefaultValueFiller, DataProcessor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from Visualization import VisualizeData

# Load data
# *******************************************************************************
# data = pd.read_csv('Data/infinity_wallcontrol_power_data.csv', low_memory=False)
# print(data)

# Get data from specific column
# *******************************************************************************
# features_name = ['power_true_zone_W', 'oat', 'zone_clsp']
features_name = ['vfd_ac_line_power', 'oat', 'zone_rt']
# features_name = ['oat']

# filler = DefaultValueFiller(data, features_name, 1)
# values = filler.calc_default_value_monthly('oat')
# print(values[2023][1][0])
# feature_data = filler.get_feature_data()
# print(feature_data)
# new_df = filler.fill_missing_value()
# print(new_df)
# new_df.to_csv('Data/Edison_Oct2Mar_new_0223.csv', index=False)

# data = pd.read_csv('Data/spike_test.csv', low_memory=False)
# data = data.values.flatten()
# DefaultValueFiller.spike_smoother(data)
#
# plt.plot(data)
# plt.ylim(0, 1400)
# plt.show()

# Prepare the data
# *************************************************************************
file_path = 'Data/Edison_Oct2Mar_new_0223.csv'
n_input, n_output = 48, 48
time_feature = False

sc = MinMaxScaler(feature_range=(0, 1))
sc2 = StandardScaler()
data = DataProcessor(
    file_path,
    feature_names=features_name,
    test_size=0.2,
    val_size=144,
    start_date=(2022, 10, 1),
    end_date=(2023, 1, 17),
    # hour_range=(7, 20),
    group_freq=30,
    n_input=n_input,
    n_output=n_output,
    date_column='timestamp',
    time_zone_transfer=False,
    add_time_features=time_feature,)
    # scaler=sc)

period_data = data.get_period_data()[0]
# period_data.to_csv('Data/period_data.csv')
print(period_data)

# train = data.train
# test = data.test
# val = data.val
# print(len(train))
# print(len(test))
# print(len(val))

VisualizeData.plot_variable_no_time(period_data, 'vfd_ac_line_power')
# VisualizeData.plot_variable(period_data, 'power_true_zone_W')
# VisualizeData.check_linearity(period_data, 'power_true_zone_W', 'zone_clsp', True)

# before = pd.read_csv('Data/model_power_Edison_clean.csv', low_memory=False)
# before['timestamp'] = pd.to_datetime(before['timestamp'])
# # before['timestamp'] = before['timestamp'].dt.strftime('%Y/%m/%d %H:%M')
# before.set_index('timestamp', inplace=True)
# # print(before)
#
# after = pd.read_csv('Data/new_data_0206.csv', low_memory=False)
# after['timestamp'] = pd.to_datetime(after['timestamp'])
# # after['timestamp'] = after['timestamp'].dt.strftime('%Y/%m/%d %H:%M')
# after.set_index('timestamp', inplace=True)
# # print(after)
#
# index = after.index
# # print(index)
# before_filled = pd.DataFrame()
# before_filled.index = index


# before.loc[date]['power_true_zone_W']

# power = []
# for date in index:
#     if date in before.index:
#         print(date)
#         power.append(before.loc[date]['power_true_zone_W'])
#     else:
#         power.append(np.nan)
#
# print(power)
# # print(len(power))
# before_filled['power_true_zone_W'] = power
# # before_filled.to_csv('test.csv')
# print(before_filled)

# plt.plot(power)
# plt.show()
# VisualizeData.plot_variable_no_time(before_filled, 'power_true_zone_W')
