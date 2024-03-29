import pandas as pd
from Helper import DataPreprocessorTiDE
from Visualization import VisualizeData
from darts.models import TiDEModel
from sklearn.preprocessing import MinMaxScaler
from darts.metrics import mape
from darts import concatenate
from darts import TimeSeries
from Helper import PredictAndForecastTiDE
from Helper import inverse_transform_prediction
from Helper import Evaluate

# Load the model
# *******************************************************************************
model = TiDEModel.load('models/model_opt.pkl')

# Get data from specific column
# *******************************************************************************
file_path = '../Data/Edison_Jun2Aug_new.csv'
target_names = ['power_true_zone_W']
dynamic_cov_names = ['oat', 'zone_clsp']
n_input, n_output = 89, 6

sc = MinMaxScaler(feature_range=(0, 1))

data_loader = DataPreprocessorTiDE(
    file_path,
    target_names=target_names,
    dynamic_cov_names=dynamic_cov_names,
    # start_date=(2023, 12, 31),
    # end_date=(2024, 1, 2),
    # hour_range=(6, 20),
    group_freq=5,
    test_size=0.2,
    val_size=100,
    date_column='timestamp',
    add_time_features=False,
    scaler=sc
)

train_target, train_past_covs = data_loader.train_series()
test_target, test_past_covs = data_loader.test_series()
val_target, val_past_covs = data_loader.val_series()
merged_series = concatenate([train_target, test_target], axis=0, ignore_time_axis=True)
merged_past_covs = concatenate([train_past_covs, test_past_covs], axis=0, ignore_time_axis=True)
# print(merged_series.pd_dataframe())
# print(val_target.pd_dataframe())
# future_covs = dataset.future_series(cov_names, n_extend) if use_future_covs else None

# Predict for the entire test set:
# *******************************************************************************
# predictions = PredictAndForecastTiDE(model, n_input, n_output, data_loader)
# preds = inverse_transform_prediction(
#     predictions.get_predictions().values, len(dynamic_cov_names) + len(target_names), sc)
# actuals = inverse_transform_prediction(
#     predictions.updated_test().values, len(dynamic_cov_names) + len(target_names), sc)

# Predict for one sample from the validation set:
# *************************************************************************
pred = model.predict(
    n=12,
    series=merged_series,
    past_covariates=merged_past_covs)

print(pred.pd_dataframe())

actuals = inverse_transform_prediction(
    val_target[:n_output].pd_dataframe().values, len(dynamic_cov_names) + len(target_names), sc)
preds = inverse_transform_prediction(
    pred.pd_dataframe().values, len(dynamic_cov_names) + len(target_names), sc)

# Evaluate the prediction
# *******************************************************************************
evals = Evaluate(actuals, preds)
print('TiDE Model\'s mape is: {}%'.format(round(evals.mape*100, 1)))

# Visualize the results
# *******************************************************************************
VisualizeData.plot_results(actuals, preds)
