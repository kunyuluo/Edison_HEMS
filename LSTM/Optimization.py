import optuna
from Helper import DataProcessor
from Helper import PredictAndForecast, Evaluate
# from Helper import inverse_transform_prediction
from LSTM import build_lstm_1
from sklearn.preprocessing import MinMaxScaler

# Prepare the data
# *************************************************************************
file_path = '../Data/Edison_Oct2Mar_new.csv'
features_name = ['vfd_ac_line_power', 'oat', 'zone_rt']
# features_name = ['power_true_zone_W', 'oat', 'zone_clsp']
sc = MinMaxScaler(feature_range=(0, 1))
n_input, n_output = 48, 48


# Define objectuve function.
def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 8, 48, step=4)
    # out_len = trial.suggest_int("out_len", 8, 10)
    # lstm_dim = trial.suggest_int("lstm_dim", 100, 350, step=10)
    # dense_dim = trial.suggest_int("dense_dim", 50, 150, step=10)
    # batch_size = trial.suggest_int("batch_size", 16, 32, step=16)
    # use_time_covs = trial.suggest_categorical("use_time_covs", [True, False])
    lstm_dim = 220
    dense_dim = 60

    data_loader = DataProcessor(
        file_path,
        feature_names=features_name,
        test_size=0.2,
        val_size=144,
        start_date=(2022, 10, 1),
        end_date=(2023, 1, 17),
        # hour_range=(6, 20),
        group_freq=30,
        n_input=in_len,
        n_output=n_output,
        date_column='timestamp',
        scaler=sc,
        add_time_features=False)

    model = build_lstm_1(
        data_loader,
        epochs=100,
        batch_size=16,
        lstm_dim=lstm_dim,
        dense_dim=dense_dim)

    # Evaluate how good it is on the validation set, using MAPE
    # train = data_loader.train
    test = data_loader.test
    valid = data_loader.val

    # Predict for the entire test set:
    # *******************************************************************************
    prediction = PredictAndForecast(model[0], test, valid, n_input=in_len, n_output=n_output)
    preds = prediction.get_predictions()
    actuals = prediction.updated_test()
    # preds = inverse_transform_prediction(prediction.get_predictions(), len(features_name), sc)
    # actuals = sc.inverse_transform(prediction.updated_test())

    evals = Evaluate(actuals, preds)
    mape = round(evals.mape() * 100, 2)

    return mape if mape is not None else float("inf")


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the MAPE on the validation set
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[print_callback])
