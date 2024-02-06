from Helper import DataProcessor
from Visualization import VisualizeData
from LSTM import build_lstm_1
from sklearn.preprocessing import MinMaxScaler
import pickle

# Prepare the data
# *************************************************************************
file_path = '../Data/new_data_0206.csv'
features_name = ['power_true_zone_W', 'oat', 'zone_clsp']
n_input, n_output = 6, 6
time_feature = False

sc = MinMaxScaler(feature_range=(0, 1))
data = DataProcessor(
    file_path,
    feature_names=features_name,
    test_size=0.2,
    val_size=200,
    # start_date=(2023, 8, 18),
    # end_date=(2023, 10, 16),
    # hour_range=(7, 20),
    group_freq=5,
    n_input=n_input,
    n_output=n_output,
    date_column='timestamp',
    add_time_features=time_feature,)
    # scaler=sc)

# Build the model
# *************************************************************************
epochs = 40
batch_size = 32

baseline = build_lstm_1(data, epochs=epochs, batch_size=batch_size, lstm_dim=250, dense_dim=50)

model = baseline[0]
history = baseline[1]

# Save models
# *************************************************************************
with open('models/model_lstm.pkl', 'wb') as f:
    pickle.dump(model, f)

# Check metrics
# *************************************************************************
VisualizeData.plot_metrics(history, epochs=epochs)
