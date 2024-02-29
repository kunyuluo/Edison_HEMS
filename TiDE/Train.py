from Helper import DataPreprocessorTiDE
from sklearn.preprocessing import MinMaxScaler
from TiDE import build_tide_1

# Get data from specific column
# *******************************************************************************
file_path = '../Data/Edison_Oct2Mar_new.csv'
target_names = ['vfd_ac_line_power']
dynamic_cov_names = ['oat', 'zone_rt']
n_input, n_output = 32, 48

sc = MinMaxScaler(feature_range=(0, 1))

data_loader = DataPreprocessorTiDE(
    file_path,
    target_names=target_names,
    dynamic_cov_names=dynamic_cov_names,
    start_date=(2022, 10, 1),
    end_date=(2023, 1, 17),
    # hour_range=(6, 20),
    group_freq=30,
    test_size=0.2,
    val_size=48,
    date_column='timestamp',
    add_time_features=False,
    scaler=sc
)

# data = data_loader.get_period_data()
# print(data)
# future = data_loader.future_series('downstream_chwsstpt', 30)
# print(future.pd_dataframe())

# Build the model
# *******************************************************************************
model = build_tide_1(
    data_loader,
    input_chunk_length=n_input,
    output_chunk_length=n_output,
    epochs=50,
    batch_size=32,
    decoder_output_dim=12,
    hidden_size=192,
    use_rin=True,
    use_future_covs=False)

# Save models
# *******************************************************************************
model.save('models/model_opt.pkl')
