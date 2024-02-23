import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Helper import DataProcessor, PredictAndForecast, Evaluate
from Helper import inverse_transform_prediction
from Visualization import VisualizeData

# Load the model
# *************************************************************************
with open('models/model_lstm.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare the data
# *************************************************************************
file_path = '../Data/Edison_Oct2Mar_new.csv'
features_name = ['vfd_ac_line_power', 'oat', 'zone_rt']
# features_name = ['power_true_zone_W', 'oat', 'zone_clsp']
n_input, n_output = 48, 48
time_feature = False

sc = MinMaxScaler(feature_range=(0, 1))
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
    add_time_features=time_feature,
    scaler=sc)

train = data.train
test = data.test
val = data.val

# Predict for the entire test set:
# *************************************************************************
prediction = PredictAndForecast(model, test, val, n_input=n_input, n_output=n_output)
# predict_values = prediction.get_predictions()
# actual_values = prediction.updated_test()
# predict_values = inverse_transform_prediction(prediction.get_predictions(), len(features_name), sc)
# actual_values = sc.inverse_transform(prediction.updated_test())

# Walk-Forward Predict for certain length of step:
# *************************************************************************
walk_forward = prediction.walk_forward_validation(144, 0)
predict_values = inverse_transform_prediction(walk_forward[0], len(features_name), sc)
actual_values = inverse_transform_prediction(walk_forward[1], len(features_name), sc)

# df = pd.DataFrame({'pred': predict_values.flatten(), 'actual': actual_values[:, 0]})
# print(df)
# df.to_csv('results.csv')

# Evaluate the prediction
# *************************************************************************
evals = Evaluate(actual_values, predict_values)
print('LSTM Model\'s mape is: {}%'.format(round(evals.mape*100, 1)))
print('LSTM Model\'s var ratio is: {}%'.format(round(evals.var_ratio*100, 1)))

# Visualize the results
# *************************************************************************
# VisualizeData.plot_results(actual_values, predict_values)
VisualizeData.plot_results(actual_values, predict_values)
# VisualizeData.plot_sample_results(actual_values, predict_values)