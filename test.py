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
# data = pd.read_csv('Data/model_power_Edison_clean.csv', low_memory=False)
# print(data)

# Get data from specific column
# *******************************************************************************
features_name = ['power_true_zone_W', 'oat', 'zone_clsp']

# filler = DefaultValueFiller(data, features_name, 1)
# feature_data = filler.get_feature_data()
# print(feature_data)
# new_df = filler.fill_missing_value()
# print(new_df)
# new_df.to_csv('Data/new_data_0206.csv', index=False)

# data = pd.read_csv('Data/spike_test.csv', low_memory=False)
# data = data.values.flatten()
# DefaultValueFiller.spike_smoother(data)
#
# plt.plot(data)
# plt.ylim(0, 1400)
# plt.show()

# Prepare the data
# *************************************************************************
file_path = 'Data/new_data_0206.csv'
n_input, n_output = 6, 6
time_feature = False

sc = MinMaxScaler(feature_range=(0, 1))
sc2 = StandardScaler()
data = DataProcessor(
    file_path,
    feature_names=features_name,
    test_size=0.2,
    start_date=(2023, 8, 8),
    end_date=(2023, 8, 10),
    hour_range=(7, 20),
    group_freq=5,
    n_input=n_input,
    n_output=n_output,
    date_column='timestamp',
    time_zone_transfer=True,
    add_time_features=time_feature,)
    # scaler=sc)

period_data = data.get_period_data()[0]
print(period_data)

# VisualizeData.plot_variable_no_time(period_data, 'power_true_zone_W')
VisualizeData.plot_variable(period_data, 'power_true_zone_W', False)
# VisualizeData.check_linearity(period_data, 'power_true_zone_W', 'zone_clsp', True)
