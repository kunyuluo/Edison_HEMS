import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from darts.dataprocessing.transformers import (Scaler, MissingValuesFiller, Mapper, InvertibleMapper,)
from darts.metrics import *
from darts.models import *
from darts import TimeSeries


def data_processing(df):
    df_a = df.copy()
    df_a['vfd_ac_line_power'] = df_a['vfd_ac_line_power'].fillna(df['vfd_ac_line_power'].shift(7 * 48))
    df_a['zone_htsp'] = df_a['zone_htsp'].fillna(df_a['zone_htsp'].shift(7 * 48))
    df_a['zone_clsp'] = df_a['zone_clsp'].fillna(df_a['zone_clsp'].shift(7 * 48))
    df_a['oat'] = df_a['oat'].fillna(df_a['oat'].shift(7 * 48))
    df_a['zone_rt'] = df_a['zone_rt'].fillna(df_a['zone_rt'].shift(7 * 48))

    df_a['vfd_ac_line_power'] = df_a['vfd_ac_line_power'].fillna(df['vfd_ac_line_power'].shift(7 * 48))
    df_a['zone_htsp'] = df_a['zone_htsp'].fillna(df_a['zone_htsp'].shift(7 * 48))
    df_a['zone_clsp'] = df_a['zone_clsp'].fillna(df_a['zone_clsp'].shift(7 * 48))
    df_a['oat'] = df_a['oat'].fillna(df_a['oat'].shift(7 * 48))
    df_a['zone_rt'] = df_a['zone_rt'].fillna(df_a['zone_rt'].shift(7 * 48))

    df_a["dToht"] = df_a["oat"] - df_a["zone_htsp"]
    df_a["dTocl"] = df_a["oat"] - df_a["zone_clsp"]

    return df_a


def model_run(model, train, test, train_data, forecast_period, future_covariates=None, past_covariates=None):
    n_samples = 48
    df_train = train_data[:-forecast_period]

    def eval_model(model, train, future_covariates=None, past_covariates=None):
        model.fit(train, future_covariates=future_covariates, past_covariates=past_covariates)
        forecast = model.predict(forecast_period, future_covariates=future_covariates)
        # forecast = scaler.inverse_transform(forecast)
        forecast = forecast.pd_dataframe().reset_index()
        forecast['vfd_ac_line_power'] = np.where(forecast['vfd_ac_line_power'] < 0, 1, forecast['vfd_ac_line_power'])
        return forecast

    # scaler = Scaler()
    # train = scaler.fit_transform(train)

    # Evalulating Model
    model = eval_model(model, train, future_covariates)

    # Adding Test samples for dataframe
    model['acutal_vfd_ac_line_power'] = test

    # Plotting
    # trace_train_data = go.Scatter(x=df_train[-1000:]['timestamp'], y=df_train[-1000:]['vfd_ac_line_power'],
    #                               mode='lines', name='Train Data')
    # trace_test_data = go.Scatter(x=model['timestamp'], y=model['acutal_vfd_ac_line_power'], mode='lines',
    #                              name='Test Data')
    #
    # trace = go.Scatter(
    #     x=model['timestamp'],
    #     y=model['vfd_ac_line_power'],
    #     mode='lines',
    #     name='Forecast'
    # )
    #
    # # Create the figure with all the traces
    # layout = go.Layout(
    #     title='Actual vs Forecasted Energy Sum',
    #     xaxis=dict(title='Date'),
    #     yaxis=dict(title='Energy Use'),
    #     legend=dict(orientation="h", x=0.7, y=1.1)
    # )
    #
    # fig = go.Figure(data=[trace_train_data, trace_test_data, trace], layout=layout)
    # fig.show()

    # converting test and train to darts timeseries objects
    acutal = TimeSeries.from_dataframe(model, "timestamp", "acutal_vfd_ac_line_power")
    forecast = TimeSeries.from_dataframe(model, "timestamp", "vfd_ac_line_power")

    # subset for day 1,2,3
    day_1_actual = acutal[:n_samples]
    day_2_actual = acutal[n_samples:n_samples * 2]
    day_3_actual = acutal[-n_samples:]

    # subet forecast data for day1,2,3
    day_1_foreacast = forecast[:n_samples]
    day_2_foreacast = forecast[n_samples:n_samples * 2]
    day_3_foreacast = forecast[-n_samples:]

    # subset for 4-9 PM forecast and actual for 4-9 metric calcualtion
    start_time = pd.Timestamp('16:00:00').time()
    end_time = pd.Timestamp('21:00:00').time()

    # Day1
    day_1_actual_4to9PM = day_1_actual.pd_dataframe().between_time(start_time, end_time)
    day_1_actual_4to9PM = TimeSeries.from_dataframe(day_1_actual_4to9PM)

    day_1_foreacast_4to9PM = day_1_foreacast.pd_dataframe().between_time(start_time, end_time)
    day_1_foreacast_4to9PM = TimeSeries.from_dataframe(day_1_foreacast_4to9PM)

    # Day2
    day_2_actual_4to9PM = day_2_actual.pd_dataframe().between_time(start_time, end_time)
    day_2_actual_4to9PM = TimeSeries.from_dataframe(day_2_actual_4to9PM)

    day_2_foreacast_4to9PM = day_2_foreacast.pd_dataframe().between_time(start_time, end_time)
    day_2_foreacast_4to9PM = TimeSeries.from_dataframe(day_2_foreacast_4to9PM)

    # Day3
    day_3_actual_4to9PM = day_3_actual.pd_dataframe().between_time(start_time, end_time)
    day_3_actual_4to9PM = TimeSeries.from_dataframe(day_3_actual_4to9PM)

    day_3_foreacast_4to9PM = day_3_foreacast.pd_dataframe().between_time(start_time, end_time)
    day_3_foreacast_4to9PM = TimeSeries.from_dataframe(day_3_foreacast_4to9PM)

    # Calculating Metrics
    metrics_day1 = {
        'SMAPE': smape(day_1_actual, day_1_foreacast),
        'RMSE': rmse(day_1_actual, day_1_foreacast),
        'OPE': ope(day_1_actual, day_1_foreacast),
        'R-squared': r2_score(day_1_actual, day_1_foreacast),
        'CV': coefficient_of_variation(day_1_actual, day_1_foreacast)
        , 'Forecast Horizon': str(model['timestamp'].min().date()) + " to " + str(model['timestamp'].max().date())
        , 'Training Horizon': str(df_train['timestamp'].min().date()) + " to " + str(df_train['timestamp'].max().date())
        , 'SMAPE_4to9': smape(day_1_actual_4to9PM, day_1_foreacast_4to9PM),
        'RMSE_4to9': rmse(day_1_actual_4to9PM, day_1_foreacast_4to9PM),
        'OPE_4to9': ope(day_1_actual_4to9PM, day_1_foreacast_4to9PM),
        'R-squared_4to9': r2_score(day_1_actual_4to9PM, day_1_foreacast_4to9PM),
        'CV_4to9': coefficient_of_variation(day_1_actual_4to9PM, day_1_foreacast_4to9PM)
    }

    metrics_day2 = {
        'SMAPE': smape(day_2_actual, day_2_foreacast),
        'RMSE': rmse(day_2_actual, day_2_foreacast),
        'OPE': ope(day_2_actual, day_2_foreacast),
        'R-squared': r2_score(day_2_actual, day_2_foreacast),
        'CV': coefficient_of_variation(day_2_actual, day_2_foreacast)
        , 'Forecast Horizon': str(model['timestamp'].min().date()) + " to " + str(model['timestamp'].max().date())
        , 'Training Horizon': str(df_train['timestamp'].min().date()) + " to " + str(df_train['timestamp'].max().date())
        , 'SMAPE_4to9': smape(day_2_actual_4to9PM, day_2_foreacast_4to9PM),
        'RMSE_4to9': rmse(day_2_actual_4to9PM, day_2_foreacast_4to9PM),
        'OPE_4to9': ope(day_2_actual_4to9PM, day_2_foreacast_4to9PM),
        'R-squared_4to9': r2_score(day_2_actual_4to9PM, day_2_foreacast_4to9PM),
        'CV_4to9': coefficient_of_variation(day_2_actual_4to9PM, day_2_foreacast_4to9PM)

    }

    metrics_day3 = {
        'SMAPE': smape(day_3_actual, day_3_foreacast),
        'RMSE': rmse(day_3_actual, day_3_foreacast),
        'OPE': ope(day_3_actual, day_3_foreacast),
        'R-squared': r2_score(day_3_actual, day_3_foreacast),
        'CV': coefficient_of_variation(day_3_actual, day_3_foreacast)
        , 'Forecast Horizon': str(model['timestamp'].min().date()) + " to " + str(model['timestamp'].max().date())
        , 'Training Horizon': str(df_train['timestamp'].min().date()) + " to " + str(df_train['timestamp'].max().date())
        , 'SMAPE_4to9': smape(day_3_actual_4to9PM, day_3_foreacast_4to9PM),
        'RMSE_4to9': rmse(day_3_actual_4to9PM, day_3_foreacast_4to9PM),
        'OPE_4to9': ope(day_3_actual_4to9PM, day_3_foreacast_4to9PM),
        'R-squared_4to9': r2_score(day_3_actual_4to9PM, day_3_foreacast_4to9PM),
        'CV_4to9': coefficient_of_variation(day_3_actual_4to9PM, day_3_foreacast_4to9PM)

    }
    return metrics_day1, metrics_day2, metrics_day3


infinity_wallcontrol_power_data = pd.read_csv('../Data/infinity_wallcontrol_power_data.csv')
infinity_wallcontrol_power_data['timestamp'] = pd.to_datetime(infinity_wallcontrol_power_data['timestamp'],utc=True)
infinity_wallcontrol_power_data['timestamp'] = infinity_wallcontrol_power_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
infinity_wallcontrol_power_data = infinity_wallcontrol_power_data[infinity_wallcontrol_power_data['timestamp']<'2023-01-16`']
infinity_wallcontrol_power_data = infinity_wallcontrol_power_data[infinity_wallcontrol_power_data['timestamp']>'2022-10-01`']


df = infinity_wallcontrol_power_data.copy()

df_org = infinity_wallcontrol_power_data.copy()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Define resampling rule
resampling_rule = '30T'  # 30 minutes

# Define aggregation functions
agg_funcs = {
    'oat': 'mean',
    'zone_rt': 'mean',
    'zone_htsp': 'mean',
    'zone_clsp': 'mean',
    'vfd_ac_line_power': 'mean',
    'vfd_ac_line_power_raw': 'mean',
    'zone_currentActivity': 'first',
    'mode': 'first',
    'zone_conditioningStatus': 'first'
}

df = df.resample(resampling_rule).agg(agg_funcs)

df.reset_index(inplace=True)

df['hour_of_day'] = df['timestamp'].dt.hour
df['minute_of_hour'] = df['timestamp'].dt.minute

df = data_processing(df)
# print(df)
#
# plt.plot(df['vfd_ac_line_power'])
# plt.show()

orginal_data = df.copy()

# Foredcast period
forecast_period = 48 * 3
# how many days of traning data need to be use
# train_subset = orginal_data.copy()
train_subset = orginal_data[-48 * 60:]

# Test and Train Split
df_train = train_subset[:-forecast_period]
df_test = train_subset[-forecast_period:]

# Creating Darts Time series objects
train_series = TimeSeries.from_dataframe(df_train, "timestamp", "vfd_ac_line_power")
test_series = train_subset[-forecast_period:][['vfd_ac_line_power']].reset_index(drop=True)

# Creating timeseries objects for future covairates
future_series_oat = TimeSeries.from_dataframe(train_subset[["timestamp", "oat"]], "timestamp", "oat")
future_series_rt = TimeSeries.from_dataframe(train_subset[["timestamp", "zone_rt"]], "timestamp", "zone_rt")
future_series_dToht = TimeSeries.from_dataframe(train_subset[["timestamp", "dToht"]], "timestamp", "dToht")
future_series_zone_htsp = TimeSeries.from_dataframe(train_subset[["timestamp", "zone_htsp"]], "timestamp", "zone_htsp")

future_series_dTocl = TimeSeries.from_dataframe(train_subset[["timestamp", "dTocl"]], "timestamp", "dTocl")
future_series_zone_clsp = TimeSeries.from_dataframe(train_subset[["timestamp", "zone_clsp"]], "timestamp", "zone_clsp")

future_series_hour_of_day = TimeSeries.from_dataframe(train_subset[["timestamp", "hour_of_day"]], "timestamp",
                                                      "hour_of_day")
future_series_minute_of_hour = TimeSeries.from_dataframe(train_subset[["timestamp", "minute_of_hour"]], "timestamp",
                                                         "minute_of_hour")

# future_cov stacking which model need to be used
future_cov = future_series_oat  # .stack(future_series_rt)

# future_cov = future_cov.stack(future_series_hour_of_day)
# future_cov = future_cov.stack(future_series_minute_of_hour)
# future_cov = future_cov.stack(future_series_dToht)

# create covaraites varaiables to log
# covaraites = 'future_series_oat,future_series_hour_of_day,future_series_zone_htsp'
covaraites = 'future_series_oat'

# logging outputs
Forecast_Horizon = str(df_test['timestamp'].min().date()) + " to " + str(df_test['timestamp'].max().date())
Training_Horizon = str(df_train['timestamp'].min().date()) + " to " + str(df_train['timestamp'].max().date())

# Model Paramters
random_state = 42
lags = 48 * 7
output_chunk_length = 48
lags_future_covariates = (48, 0)
n_estimators = 80

# Intiating Model
model = RandomForest(random_state=random_state, n_estimators=n_estimators, lags=lags,
                     output_chunk_length=output_chunk_length,
                     lags_future_covariates=lags_future_covariates)

# calling model_run method
metrics_day1, metrics_day2, metrics_day3 = model_run(model, train_series, test_series, train_subset,
                                                     forecast_period=3 * 48, future_covariates=future_cov,
                                                     past_covariates=None)

df_metric_output = pd.DataFrame([metrics_day1, metrics_day2, metrics_day3])
print(df_metric_output)
