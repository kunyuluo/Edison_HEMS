import optuna
from Helper import DataPreprocessorTiDE
from Helper import inverse_transform_prediction
from Helper import Evaluate
from Helper import PredictAndForecastTiDE
# from darts.metrics import mape
from darts.models import TiDEModel
from sklearn.preprocessing import MinMaxScaler
from darts import concatenate
from TiDE import build_tide_1

# Get data from specific column
# *******************************************************************************
file_path = '../Data/Edison_Jun2Aug_new.csv'
target_names = ['power_true_zone_W']
dynamic_cov_names = ['oat', 'zone_clsp']
sc = MinMaxScaler(feature_range=(0, 1))
n_output = 48


# Define objectuve function.
def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 10, 100)
    # out_len = trial.suggest_int("out_len", 7, 10)
    decoder_output_dim = trial.suggest_int("decoder_output_dim", 8, 24, step=4)
    hidden_size = trial.suggest_int("hidden_size", 64, 256, step=64)
    use_rin = trial.suggest_categorical("use_rin", [True, False])
    use_time_covs = trial.suggest_categorical("use_time_covs", [True, False])

    data_loader = DataPreprocessorTiDE(
        file_path,
        target_names=target_names,
        dynamic_cov_names=dynamic_cov_names,
        group_freq=30,
        test_size=0.2,
        val_size=100,

        date_column='timestamp',
        add_time_features=use_time_covs,
        scaler=sc
    )

    model = build_tide_1(
        data_loader,
        input_chunk_length=in_len,
        output_chunk_length=n_output,
        epochs=50,
        batch_size=32,
        decoder_output_dim=decoder_output_dim,
        hidden_size=hidden_size,
        use_rin=use_rin,
        use_future_covs=False)

    # Evaluate how good it is on the validation set, using MAPE
    # train_target, train_past_covs = data_loader.train_series()
    # test_target, test_past_covs = data_loader.test_series()
    # val_target, val_past_covs = data_loader.val_series()
    # merged_series = concatenate([train_target, test_target], axis=0, ignore_time_axis=True)
    # merged_past_covs = concatenate([train_past_covs, test_past_covs], axis=0, ignore_time_axis=True)

    # Predict for one sample from the validation set:
    # *************************************************************************
    # pred = model.predict(
    #     n=10,
    #     series=merged_series,
    #     past_covariates=merged_past_covs)
    #
    # actuals = inverse_transform_prediction(val_target_series[:10].pd_dataframe().values, 4, sc)
    # preds = inverse_transform_prediction(pred.pd_dataframe().values, 4, sc)

    # Predict for the entire test set:
    # *******************************************************************************
    predictions = PredictAndForecastTiDE(model, in_len, n_output, data_loader)
    preds = inverse_transform_prediction(
        predictions.get_predictions().values, len(dynamic_cov_names) + len(target_names), sc)
    actuals = inverse_transform_prediction(
        predictions.updated_test().values, len(dynamic_cov_names) + len(target_names), sc)

    evals = Evaluate(actuals, preds)
    mape = round(evals.mape * 100, 2)

    return mape if mape is not None else float("inf")


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the MAPE on the validation set
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, callbacks=[print_callback])
