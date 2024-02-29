import optuna
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller, Mapper, InvertibleMapper
from darts import concatenate
from darts.models import TiDEModel
from darts.metrics import mape

# Prepare the data
# *********************************************************************************************
data = pd.read_csv('../Data/infinity_wallcontrol_power_data.csv', low_memory=True)

data['timestamp'] = pd.to_datetime(data['timestamp'])
data['timestamp'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

data_new = data[['timestamp', 'oat', 'zone_rt', 'vfd_ac_line_power']]
data_new = data_new.set_index('timestamp')
data_new.index = pd.to_datetime(data_new.index)
data_new_series = TimeSeries.from_dataframe(data_new, fill_missing_dates=True, freq='5min')

transformer = MissingValuesFiller()
data_new_series = transformer.transform(data_new_series)
data_new_series.pd_dataframe().to_csv('data_new.csv')

train, val = data_new_series.split_before(pd.Timestamp("20230308"))
valex, train = train.split_after(pd.Timestamp("20230101"))

# Standardization
scaler = Scaler()  # default uses sklearn's MinMaxScaler
train = scaler.fit_transform(train)
val = scaler.transform(val)

validation = val
target = train['vfd_ac_line_power']
past_cov = concatenate([train['zone_rt'], train['oat']], axis=1)
# *********************************************************************************************


# Define objectuve function.
# *********************************************************************************************
def objective(trial):
    # Define the hyperparameters to be optimized:
    # Based on the type of the variable, use "suggest_int", "suggest_float", or "suggest_categorical" etc.
    # Define lower bound, higher bound and step for each variable
    in_len = trial.suggest_int("in_len", 10, 100)
    decoder_output_dim = trial.suggest_int("decoder_output_dim", 8, 24, step=4)
    hidden_size = trial.suggest_int("hidden_size", 64, 256, step=64)
    use_rin = trial.suggest_categorical("use_rin", [True, False])

    tide_model = TiDEModel(
        input_chunk_length=in_len,
        output_chunk_length=24,
        random_state=42,
        n_epochs=10,
        decoder_output_dim=decoder_output_dim,
        hidden_size=hidden_size,
        use_reversible_instance_norm=use_rin)

    tide_model.fit(target, past_covariates=past_cov)
    tide_pred = tide_model.predict(24)
    error = mape(validation['vfd_ac_line_power'], tide_pred['vfd_ac_line_power'])

    return error if error is not None else float("inf")


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the MAPE on the validation set
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[print_callback])
