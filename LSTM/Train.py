from Helper import DataProcessor
from Helper import PredictAndForecast, Evaluate
from Visualization import VisualizeData
from LSTM import build_lstm_1
from sklearn.preprocessing import MinMaxScaler
import pickle

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

# Build the model
# *************************************************************************
epochs = 100
batch_size = 16

baseline = build_lstm_1(data, epochs=epochs, batch_size=batch_size, lstm_dim=160, dense_dim=70)

model = baseline[0]
history = baseline[1]

# Save models
# *************************************************************************
with open('models/model_lstm.pkl', 'wb') as f:
    pickle.dump(model, f)

# Check metrics
# *************************************************************************
VisualizeData.plot_metrics(history, epochs=epochs)
