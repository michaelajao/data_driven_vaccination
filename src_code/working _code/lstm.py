import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
# Adjust the list of features to exclude the missing columns
adjusted_features = [
    "covidOccupiedMVBeds",
    "covidOccupiedMVBeds_lag_1",
    "covidOccupiedMVBeds_lag_2",
    "covidOccupiedMVBeds_lag_3",
    "covidOccupiedMVBeds_lag_5",
    "covidOccupiedMVBeds_lag_7",
    "covidOccupiedMVBeds_lag_14",
    "covidOccupiedMVBeds_lag_21",
    "month",
    "day",
    "day_of_week",
    # "susceptible",
    # "exposed",
    # "active_cases",
    # "recovered",
    # "cumulative_deceased"
]

# Load the data (assuming the merged_data is already loaded as a DataFrame)
# merged_data = pd.read_csv('data/processed/england_data.csv')

# Initialize the scaler
scaler = MinMaxScaler()

# Fit and transform the data
data_scaled = pd.DataFrame(scaler.fit_transform(data[adjusted_features]), columns=adjusted_features)

# Create windowed sequences for training and testing
def create_windowed_dataset(data, target_column, window_size=7):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size].drop(columns=[target_column]).values)
        y.append(data[target_column].values[i + window_size])
    return np.array(X), np.array(y)

# Prepare the data
target_column = "covidOccupiedMVBeds"
window_size = 7

# Create windowed sequences
X_windowed, y_windowed = create_windowed_dataset(data_scaled, target_column, window_size)

# Split the data into training and testing sets since it is a time series data
X_train, X_test, y_train, y_test = train_test_split(X_windowed, y_windowed, test_size=0.2, shuffle=False)

# Reshape the data to fit the RNN input format (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], window_size, X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], window_size, X_test.shape[2]))

# Build the enhanced LSTM model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(LSTM(100, activation='relu', return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(LSTM(100, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=2)

# Evaluate the model
train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"Train Loss: {train_loss}, Train MAE: {train_mae}")
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

def rescale_predictions(predictions, X, scaler, original_features):
    # Create a placeholder array for rescaling
    rescaled_predictions = np.zeros((predictions.shape[0], len(original_features)))
    rescaled_predictions[:, 0] = predictions[:, 0]  # Put predictions in the first column
    # Use the remaining columns from the input features to rescale properly
    rescaled_predictions[:, 1:X.shape[2]] = X[:, 0, 1:]
    # Inverse transform using the scaler
    rescaled_predictions = scaler.inverse_transform(rescaled_predictions)
    return rescaled_predictions[:, 0]  # Return only the first column (rescaled predictions)

# Original features used for scaling
original_features = adjusted_features

# # Function to rescale the predicted values
# def rescale_predictions(predictions, X, scaler, original_features):
#     num_features = len(original_features)
#     rescaled_predictions = np.zeros((predictions.shape[0], num_features))
#     rescaled_predictions[:, 0] = predictions[:, 0]
#     rescaled_predictions[:, 1:] = X[:, -1, :-1]
#     rescaled_predictions_full = scaler.inverse_transform(rescaled_predictions)
#     return rescaled_predictions_full[:, 0]

# # Original features used for scaling
# original_features = adjusted_features

# Rescale the predictions and actual values
y_train_pred_rescaled = rescale_predictions(y_train_pred, X_train, scaler, original_features)
y_test_pred_rescaled = rescale_predictions(y_test_pred, X_test, scaler, original_features)
y_train_rescaled = rescale_predictions(y_train.reshape(-1, 1), X_train, scaler, original_features)
y_test_rescaled = rescale_predictions(y_test.reshape(-1, 1), X_test, scaler, original_features)

# Calculate MAE and MSE on the original scale
train_mae_original = mean_absolute_error(y_train_rescaled, y_train_pred_rescaled)
test_mae_original = mean_absolute_error(y_test_rescaled, y_test_pred_rescaled)
train_mse_original = mean_squared_error(y_train_rescaled, y_train_pred_rescaled)
test_mse_original = mean_squared_error(y_test_rescaled, y_test_pred_rescaled)

print(f"Train MAE (original scale): {train_mae_original}")
print(f"Test MAE (original scale): {test_mae_original}")
print(f"Train MSE (original scale): {train_mse_original}")
print(f"Test MSE (original scale): {test_mse_original}")

# Plot the actual vs predicted values for the test set
plt.figure(figsize=(14, 7))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(y_test_pred_rescaled, label='Predicted')
plt.title('Actual vs Predicted covidOccupiedMVBeds')
plt.xlabel('Time')
plt.ylabel('covidOccupiedMVBeds')
plt.legend()
plt.show()
