# Experiment 2 Script for COVID-19 Forecasting

# Change path to the root directory of the project
import os

os.chdir("../../")

# Description: This script contains the code for the second experiment in the project,
# forecasting COVID-19 MVBeds using various RNN models and hyperparameter tuning with Simulated Annealing.

# Imports for handling data
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import cycle

# Imports for machine learning
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_squared_error as mse,
    mean_absolute_percentage_error as mape,
)

# Imports for visualization
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Progress bar
from tqdm.autonotebook import tqdm

tqdm.pandas()

# Local imports for data loaders and models
from src.utils import plotting_utils
from src.dl.dataloaders import TimeSeriesDataModule
from src.dl.multivariate_models import (
    SingleStepRNNConfig,
    SingleStepRNNModel,
    Seq2SeqConfig,
    Seq2SeqModel,
    RNNConfig,
)
from src.transforms.target_transformations import AutoStationaryTransformer

# Set seeds for reproducibility
pl.seed_everything(42)
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set float32 matrix multiplication precision
torch.set_float32_matmul_precision("high")

# Set default Plotly template
pio.templates.default = "simple_white"

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# Set logging configuration
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Utility Functions
def format_plot(
    fig, legends=None, xlabel="Time", ylabel="Value", title="", font_size=15
):
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t: t.update(name=next(names)))
    fig.update_layout(
        autosize=False,
        width=1200,
        height=600,
        title_text=title,
        title={"x": 0.5, "xanchor": "center", "yanchor": "top"},
        titlefont={"size": 20},
        legend_title=None,
        legend=dict(
            font=dict(size=font_size),
            orientation="h",
            yanchor="bottom",
            y=0.98,
            xanchor="right",
            x=1,
        ),
        yaxis=dict(
            title_text=ylabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
            showgrid=False
        ),
        xaxis=dict(
            title_text=xlabel,
            titlefont=dict(size=font_size),
            tickfont=dict(size=font_size),
            showgrid=False
        ),
    )
    return fig


def mase(actual, predicted, insample_actual):
    mae_insample = np.mean(np.abs(np.diff(insample_actual)))
    mae_outsample = np.mean(np.abs(actual - predicted))
    return mae_outsample / mae_insample


def forecast_bias(actual, predicted):
    return np.mean(predicted - actual)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def nrmse(y_true, y_pred):
    """Calculate the Normalized Root Mean Square Error (NRMSE)."""
    return np.sqrt(mse(y_true, y_pred)) / (
        np.max(y_true) - np.min(y_true)
    )


def plot_forecast(
    pred_df,
    forecast_columns,
    selected_area,
    forecast_display_names=None,
    save_path=None,
):
    if forecast_display_names is None:
        forecast_display_names = forecast_columns
    else:
        assert len(forecast_columns) == len(forecast_display_names)

    mask = ~pred_df[forecast_columns[0]].isnull()
    colors = px.colors.qualitative.D3  # Use a colorblind-friendly palette
    act_color = colors[0]
    colors = cycle(colors[1:])

    fig = go.Figure()

    # Actual data plot
    fig.add_trace(
        go.Scatter(
            x=pred_df[mask].index,
            y=pred_df[mask].covidOccupiedMVBeds,
            mode="lines+markers",
            marker=dict(size=8, opacity=0.7, symbol="circle"),
            line=dict(color=act_color, width=3),
            name="Actual COVID-19 MVBeds trends",
        )
    )

    # Predicted data plot
    line_styles = ["solid", "dash", "dot", "dashdot"]
    markers = ["circle", "square", "diamond", "cross", "x", "triangle-up"]
    for col, display_col, line_style, marker in zip(
        forecast_columns, forecast_display_names, cycle(line_styles), cycle(markers)
    ):
        fig.add_trace(
            go.Scatter(
                x=pred_df[mask].index,
                y=pred_df.loc[mask, col],
                mode="lines+markers",
                marker=dict(size=6, symbol=marker),
                line=dict(color=next(colors), width=2, dash=line_style),
                name=display_col,
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="COVID-19 MVBeds",
        xaxis=dict(title_font=dict(size=15), tickfont=dict(size=12), showgrid=False),
        yaxis=dict(title_font=dict(size=15), tickfont=dict(size=12), showgrid=False),
        legend=dict(
            font=dict(size=12),
            orientation="h",
            yanchor="top",
            y=1.12,  # Move the legend down to separate it from the title
            xanchor="center",
            x=0.5,
        ),
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=40, t=60, b=40),  # Adjust left margin for better alignment
        width=1200,
        height=600,
    )

    if save_path:
        pio.write_image(fig, f"{save_path}.png")
        pio.write_image(fig, f"{save_path}.pdf")
    return fig


def highlight_abs_min(s, props=""):
    return np.where(s == np.nanmin(np.abs(s.values)), props, "")


# Function to convert data to float32
def to_float32(df):
    for col in df.columns:
        df[col] = df[col].astype("float32")
    return df

# Load and Prepare Data
data_path = Path("data/processed/england_data.csv")
data = pd.read_csv(data_path)
data["date"] = pd.to_datetime(data["date"])

# Select and Process Data
selected_area = "England"
data.head()

# Add time-lagged features
def add_lags(data, lags, features):
    added_features = []
    for feature in features:
        for lag in lags:
            new_feature = feature + f"_lag_{lag}"
            data[new_feature] = data[feature].shift(lag)
            added_features.append(new_feature)
    return data, added_features


lags = [1, 2, 3, 5, 7, 14, 21]
data, added_features = add_lags(data, lags, ["covidOccupiedMVBeds"])
data.dropna(inplace=True)


# Create temporal features
def create_temporal_features(df, date_column):
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day
    df["day_of_week"] = df[date_column].dt.dayofweek
    return df


data = create_temporal_features(data, "date")
data = data.set_index("date")
data.info()

# Add rolling features
def add_rolling_features(df, window_size, columns, agg_funcs=None):
    if agg_funcs is None:
        agg_funcs = ["mean", "std", "min", "max"]
    added_features = {}
    for column in columns:
        for func in agg_funcs:
            roll_col_name = f"{column}_rolling_{window_size}_{func}"
            df[roll_col_name] = df[column].rolling(window_size).agg(func)
            if column not in added_features:
                added_features[column] = []
            added_features[column].append(roll_col_name)
    df.dropna(inplace=True)
    return df, added_features


window_size = 7
columns_to_roll = [
    "hospitalCases",
    "newAdmissions",
    "new_confirmed",
    "new_deceased",
]

agg_funcs = ["mean"]

merged_data, added_features = add_rolling_features(
    data, window_size, columns_to_roll, agg_funcs
)

for column, features in added_features.items():
    logging.info(f"{column}: {', '.join(features)}")

# plot cumulative_deceased
fig = px.line(
    merged_data,
    x=merged_data.index,
    y="hospitalCases_rolling_7_mean",
    title="Hospital Cases Rolling Mean",
)
fig.show()
fig.write_image("reports/figures/hospital_cases_rolling_mean.png")

# Filter data between the specified dates
start_date = "2020-04-27"
end_date = "2021-12-31"

merged_data = merged_data[start_date:end_date]

# Calculate the range of dates
date_range = merged_data.index[-1] - merged_data.index[0]
logging.info(f"Data ranges from {merged_data.index[0]} to {merged_data.index[-1]} ({date_range.days} days)")

# drop columns that are not needed
merged_data.drop(
    columns=[
        "areaName",
        "population",
        "openstreetmap_id",
        "latitude",
        "longitude",
    ],
    inplace=True,
)

# Calculate split points
total_days = date_range.days
train_end = merged_data.index[0] + pd.Timedelta(days=int(total_days * 0.70))
val_end = train_end + pd.Timedelta(days=int(total_days * 0.20))

# Split the data into training, validation, and testing sets
train = merged_data[merged_data.index <= train_end]
val = merged_data[(merged_data.index > train_end) & (merged_data.index <= val_end)]
test = merged_data[merged_data.index > val_end]

# Calculate the percentage of dates in each dataset
total_sample = len(merged_data)
train_sample = len(train) / total_sample * 100
val_sample = len(val) / total_sample * 100
test_sample = len(test) / total_sample * 100

print(
    f"Train: {train_sample:.2f}%, Validation: {val_sample:.2f}%, Test: {test_sample:.2f}%"
)
print(
    f"Train: {len(train)} samples, Validation: {len(val)} samples, Test: {len(test)} samples"
)
print(
    f"Max date in train: {train.index.max()}, Min date in validation: {val.index.min()}, Max date in test: {test.index.max()}"
)

# Concatenate the DataFrames
sample_df = pd.concat([train, val, test])

# Convert all the feature columns to float32
sample_df = to_float32(sample_df)

pinn_features = [
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
]

sample_df = sample_df[pinn_features]
cols = list(sample_df.columns)
cols.remove("covidOccupiedMVBeds")
sample_df = sample_df[cols + ["covidOccupiedMVBeds"]]
sample_df.head()

# Prepare DataModule for PyTorch Lightning
datamodule = TimeSeriesDataModule(
    data=sample_df,
    n_val=val.shape[0],
    n_test=test.shape[0],
    window=7,  # 7 days window
    horizon=1,  # single step
    normalize="global",  # normalizing the data
    batch_size=32,
    num_workers=0,
)
datamodule.setup()

# Initialize pred_df
pred_df = test.copy()

# Function to train and evaluate models
def train_and_evaluate(model, rnn_config, datamodule, test, train, algorithm_name, metric_record, pred_df, cross_validate=False):
    if cross_validate:
        metrics = cross_validate(model.__class__, rnn_config, sample_df, n_folds=5)
        logging.info(f"Cross-Validation Metrics: {metrics}")
        return metric_record, pred_df
    
    trainer = pl.Trainer(
        min_epochs=5,
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="valid_loss", patience=20)],
    )
    trainer.fit(model, datamodule)

    predictions = trainer.predict(model, datamodule.test_dataloader())
    predictions = torch.cat(predictions).squeeze().detach().numpy()
    predictions = predictions * datamodule.train.std + datamodule.train.mean
    actuals = test["covidOccupiedMVBeds"].values

    assert actuals.shape == predictions.shape, "Mismatch in shapes between actuals and predictions"

    metrics = {
        "Algorithm": algorithm_name,
        "MAE": mae(actuals, predictions),
        "MAPE": mape(actuals, predictions),
        "RMSE": rmse(actuals, predictions),
        "NRMSE": nrmse(actuals, predictions),
        "MSE": mse(actuals, predictions),
        "MASE": mase(actuals, predictions, train["covidOccupiedMVBeds"].values),
        "Forecast Bias": forecast_bias(actuals, predictions),
    }

    value_formats = ["{}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.4f}", "{:.2f}"]
    metrics = {
        key: format_.format(value)
        for key, value, format_ in zip(metrics.keys(), metrics.values(), value_formats)
    }

    pred_df_ = pd.DataFrame({algorithm_name: predictions}, index=test.index)
    pred_df = pred_df.join(pred_df_)
    metric_record.append(metrics)
    logging.info(metrics)
    return metric_record, pred_df

# Train Vanilla RNN
rnn_config = SingleStepRNNConfig(
    rnn_type="RNN",
    input_size=len(pinn_features),
    hidden_size=32,
    num_layers=5,
    bidirectional=False,
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)
algorithm_name = "Vanilla RNN"
metric_record, pred_df = train_and_evaluate(model, rnn_config, datamodule, test, train, algorithm_name, metric_record=[], pred_df=pred_df)

# Train Vanilla LSTM
rnn_config = SingleStepRNNConfig(
    rnn_type="LSTM",
    input_size=len(pinn_features),
    hidden_size=32,
    num_layers=5,
    bidirectional=True,
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)
algorithm_name = "Vanilla LSTM"
metric_record, pred_df = train_and_evaluate(model, rnn_config, datamodule, test, train, algorithm_name, metric_record, pred_df)

# Train Vanilla GRU
rnn_config = SingleStepRNNConfig(
    rnn_type="GRU",
    input_size=len(pinn_features),
    hidden_size=32,
    num_layers=5,
    bidirectional=True,
    learning_rate=1e-3,
)
model = SingleStepRNNModel(rnn_config)
algorithm_name = "Vanilla GRU"
metric_record, pred_df = train_and_evaluate(model, rnn_config, datamodule, test, train, algorithm_name, metric_record, pred_df)

# plot the forecast
forecast_columns = ["Vanilla RNN", "Vanilla LSTM", "Vanilla GRU"]
forecast_display_names = ["Vanilla RNN", "Vanilla LSTM", "Vanilla GRU"]
fig = plot_forecast(
    pred_df, forecast_columns, selected_area, forecast_display_names=forecast_display_names,
    save_path="reports/figures/vanilla_rnn_lstm_gru_forecast"
)
fig.show()

param_bounds = {
    "rnn_type": ["RNN", "GRU", "LSTM"],
    "hidden_size": (32, 128),
    "num_layers": (5, 100),
    "bidirectional": [True, False]
}

initial_params = ["RNN", 32, 5, True]
initial_temp = 10

def objective(params):
    rnn_type, hidden_size, num_layers, bidirectional = params
    rnn_config = SingleStepRNNConfig(
        rnn_type=rnn_type,
        input_size=len(pinn_features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        learning_rate=1e-3
    )
    model = SingleStepRNNModel(rnn_config)
    model.float()

    trainer = pl.Trainer(
        min_epochs=5,
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="valid_loss", patience=20)],
    )
    trainer.fit(model, datamodule)

    predictions = trainer.predict(model, datamodule.test_dataloader())
    predictions = torch.cat(predictions).squeeze().detach().numpy()
    predictions = predictions * datamodule.train.std + datamodule.train.mean

    actuals = test["covidOccupiedMVBeds"].values

    assert actuals.shape == predictions.shape, "Mismatch in shapes between actuals and predictions"

    return np.mean(np.abs(actuals - predictions))

def neighbor(params):
    if len(params) == 4:
        rnn_type, hidden_size, num_layers, bidirectional = params
        hidden_size = np.random.randint(*param_bounds["hidden_size"])
        num_layers = np.random.randint(*param_bounds["num_layers"])
        rnn_type = np.random.choice(param_bounds["rnn_type"])
        bidirectional = bool(np.random.choice(param_bounds["bidirectional"]))
        return [rnn_type, hidden_size, num_layers, bidirectional]
    elif len(params) == 6:
        encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden = params
        hidden_size = np.random.randint(*param_bounds["hidden_size"])
        num_layers = np.random.randint(*param_bounds["num_layers"])
        encoder_type = np.random.choice(param_bounds["encoder_type"])
        decoder_type = np.random.choice(param_bounds["decoder_type"])
        bidirectional = bool(np.random.choice(param_bounds["bidirectional"]))
        decoder_use_all_hidden = bool(np.random.choice(param_bounds["decoder_use_all_hidden"]))
        return [encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden]

def simulated_annealing(objective, initial_params, initial_temp, neighbor, n_iter, cooling_rate=0.20, verbose=True):
    current_params = initial_params
    current_cost = objective(current_params)
    best_params = current_params
    best_cost = current_cost
    temp = initial_temp
    cost_history = []

    for i in range(n_iter):
        candidate_params = neighbor(current_params)
        candidate_cost = objective(candidate_params)

        acceptance_probability = np.exp(-abs(candidate_cost - current_cost) / temp)

        if candidate_cost < current_cost or np.random.uniform() < acceptance_probability:
            current_params = candidate_params
            current_cost = candidate_cost

            if current_cost < best_cost:
                best_params = current_params
                best_cost = current_cost

        temp *= cooling_rate
        cost_history.append(best_cost)

        logging.info(f"Iteration: {i+1}, Best Cost: {best_cost:.4f}, Current Cost: {current_cost:.4f}, Temperature: {temp:.4f}")
        
        # Break early if the minimum cost has been constant for over 10 iterations
        if i > 10 and np.all(np.isclose(cost_history[-10:], cost_history[-1])):
            break

    return best_cost, best_params, cost_history

n_iter = 100
cooling_rate = 0.90

best_cost, best_params, cost_history = simulated_annealing(
    objective, initial_params, initial_temp, neighbor, n_iter, cooling_rate
)

logging.info(f"Best Parameters: {best_params}, Best Cost: {best_cost}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1, len(cost_history) + 1), y=cost_history, mode="lines"))
fig.update_layout(
    title="Simulated Annealing Optimization for Hyperparameter Tuning",
    xaxis_title="Iteration",
    yaxis_title="Best Cost",
    template="plotly_white",
)

fig.write_image("reports/figures/simulated_annealing_cost.png")
fig.write_image("reports/figures/simulated_annealing_cost.pdf")
fig.show()

nn_config = SingleStepRNNConfig(
    rnn_type=best_params[0],
    input_size=len(pinn_features),
    hidden_size=best_params[1],
    num_layers=best_params[2],
    bidirectional=best_params[3],
    learning_rate=1e-3,
)

model = SingleStepRNNModel(rnn_config)
algorithm_name = f"Optimized {rnn_config.rnn_type} (SA)"
metric_record, pred_df = train_and_evaluate(model, rnn_config, datamodule, test, train, algorithm_name, metric_record, pred_df)

HORIZON = 1
WINDOW = 7

encoder_config = RNNConfig(
    input_size=len(pinn_features),
    hidden_size=32,
    num_layers=5,
    bidirectional=True
)

decoder_config = RNNConfig(
    input_size=len(pinn_features),
    hidden_size=32,
    num_layers=5,
    bidirectional=True
)

rnn2fc_config = Seq2SeqConfig(
    encoder_type="LSTM",
    decoder_type="FC",
    encoder_params=encoder_config,
    decoder_params={"window_size": WINDOW, "horizon": HORIZON},
    decoder_use_all_hidden=True,
    learning_rate=1e-3,
)

model = Seq2SeqModel(rnn2fc_config)
algorithm_name = "Seq2Seq LSTM"
metric_record, pred_df = train_and_evaluate(model, rnn2fc_config, datamodule, test, train, algorithm_name, metric_record, pred_df)

param_bounds = {
    "encoder_type": ["RNN", "GRU", "LSTM"],
    "decoder_type": ["FC"],
    "hidden_size": (32, 128),
    "num_layers": (5, 100),
    "bidirectional": [True, False],
    "decoder_use_all_hidden": [True, False],
}

initial_params = ["RNN", "FC", 32, 5, True, True]
initial_temp = 10

def objective(params):
    encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden = params

    encoder_config = RNNConfig(
        input_size=len(pinn_features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional
    )

    decoder_config = RNNConfig(
        input_size=len(pinn_features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional
    )

    rnn2fc_config = Seq2SeqConfig(
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        encoder_params=encoder_config,
        decoder_params={"window_size": WINDOW, "horizon": HORIZON},
        decoder_use_all_hidden=decoder_use_all_hidden,
        learning_rate=1e-3,
    )

    model = Seq2SeqModel(rnn2fc_config)

    trainer = pl.Trainer(
        min_epochs=5,
        max_epochs=200,
        accelerator="gpu",
        devices=1,
        callbacks=[EarlyStopping(monitor="valid_loss", patience=20)],
    )

    trainer.fit(model, datamodule)

    predictions = trainer.predict(model, datamodule.test_dataloader())
    predictions = torch.cat(predictions).squeeze().detach().numpy()
    predictions = predictions * datamodule.train.std + datamodule.train.mean

    actuals = test["covidOccupiedMVBeds"].values

    assert actuals.shape == predictions.shape, "Mismatch in shapes between actuals and predictions"

    return np.mean(np.abs(actuals - predictions))

best_cost, best_params, cost_history = simulated_annealing(
    objective, initial_params, initial_temp, neighbor, n_iter, cooling_rate
)

logging.info(f"Best Parameters: {best_params}, Best Cost: {best_cost}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(1, len(cost_history) + 1), y=cost_history, mode="lines"))
fig.update_layout(
    title="Simulated Annealing Optimization Seq2Seq for Hyperparameter Tuning",
    xaxis_title="Iteration",
    yaxis_title="Best Cost",
    template="plotly_white",
)

fig.write_image("reports/figures/simulated_annealing_seq2seq_cost.png")
fig.write_image("reports/figures/simulated_annealing_seq2seq_cost.pdf")
fig.show()

encoder_type, decoder_type, hidden_size, num_layers, bidirectional, decoder_use_all_hidden = best_params

encoder_config = RNNConfig(
    input_size=len(pinn_features),
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=bidirectional
)

decoder_config = RNNConfig(
    input_size=len(pinn_features),
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=bidirectional
)

rnn2fc_config = Seq2SeqConfig(
    encoder_type=encoder_type,
    decoder_type=decoder_type,
    encoder_params=encoder_config,
    decoder_params={"window_size": WINDOW, "horizon": HORIZON},
    decoder_use_all_hidden=decoder_use_all_hidden,
    learning_rate=1e-3,
)

model = Seq2SeqModel(rnn2fc_config)
algorithm_name = f"Optimized Seq2Seq {encoder_type} (SA)"
metric_record, pred_df = train_and_evaluate(model, rnn2fc_config, datamodule, test, train, algorithm_name, metric_record, pred_df)

shutil.rmtree("lightning_logs")

# Save metrics
metric_df = pd.DataFrame(metric_record)
metric_df[["MAE", "MAPE", "RMSE", "MSE", "MASE", "Forecast Bias"]] = metric_df[
    ["MAE", "MAPE", "RMSE", "MSE", "MASE", "Forecast Bias"]
].astype("float32")

final_metric_file = f"reports/metrics/{selected_area}_final_metrics.csv"
metric_df.to_csv(final_metric_file, index=False)

# Save predictions
forecast_columns = [
    "Vanilla RNN", 
    "Vanilla LSTM", 
    "Vanilla GRU", 
    "Optimized GRU (SA)", 
    "Seq2Seq LSTM", 
    "Optimized Seq2Seq RNN (SA)",
]

forecast_display_names = [
    "Vanilla RNN", 
    "Vanilla LSTM", 
    "Vanilla GRU", 
    "Optimized GRU (SA)", 
    "Seq2Seq LSTM", 
    "Optimized Seq2Seq RNN (SA)",
]

comparison_fig = plot_forecast(
    pred_df,
    forecast_columns,
    selected_area,
    forecast_display_names,
    save_path=f"reports/figures/{selected_area}_forecast_comparison",
)
comparison_fig.show()

pred_df.head(2)

pred_df.to_csv(f"reports/results/{selected_area}_predictions.csv")
