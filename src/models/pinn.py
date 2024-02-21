import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set the default style
plt.style.use("seaborn-v0_8-white")
plt.rcParams.update(
    {
        "lines.linewidth": 2,
        "font.family": "serif",
        "axes.titlesize": 20,
        "axes.labelsize": 14,
        "figure.figsize": [15, 8],
        "figure.autolayout": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.75",
        "legend.fontsize": "medium",
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
path = "../../data/raw/pickle/covid19_data.pkl"
data = pd.read_pickle(path)

# select one of the region
region = "North East England"
data = data[data["region"] == region]

# Convert the date to datetime
data["date"] = pd.to_datetime(data["date"])

min_date = data["date"].min()
max_date = data["date"].max()

data_range = max_date - min_date
train_end = min_date + pd.Timedelta(days=data_range.days * 0.70)
val_end = train_end + pd.Timedelta(days=data_range.days * 0.15)

# Split the data into train, validation and test
train = data[data['date'] < train_end]
val = data[(data['date'] >= train_end) & (data['date'] < val_end)]
test = data[data['date'] >= val_end]

total_sample = len(data)
train_percent = len(train) / total_sample * 100
val_percent = len(val) / total_sample * 100
test_percent = len(test) / total_sample * 100

print(f"Train: {len(train)} samples ({train_percent:.2f}%)")
print(f"Validation: {len(val)} samples ({val_percent:.2f}%)")
print(f"Test: {len(test)} samples ({test_percent:.2f}%)")