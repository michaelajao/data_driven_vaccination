# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from tqdm import trange  # Import tqdm for progress bar

# Set up directories and random seed
def setup_environment(seed=42):
    """Set up directories and random seed for reproducibility."""
    directories = ["../../models", "../../reports/figures", "../../reports/results"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

setup_environment()


# Device setup
def get_device():
    """Get the appropriate device (GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
print(f"Using device: {device}")

# Configure plots
def configure_plots():
    """Configure matplotlib for consistent plotting."""
    sns.set_style("darkgrid")
    plt.rcParams.update(
        {
            "font.size": 12,
            "figure.figsize": (12, 8),
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.dpi": 300,
        }
    )

configure_plots()

# Utility functions
def normalized_root_mean_square_error(y_true, y_pred):
    """Calculate the Normalized Root Mean Square Error (NRMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (
        np.max(y_true) - np.min(y_true)
    )

def mean_absolute_scaled_error(y_true, y_pred, y_train):
    """Calculate the Mean Absolute Scaled Error (MASE)."""
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d if d != 0 else np.inf

def calculate_errors(y_true, y_pred, y_train):
    """Calculate and return various error metrics."""
    return {
        "NRMSE": normalized_root_mean_square_error(y_true, y_pred),
        "MASE": mean_absolute_scaled_error(y_true, y_pred, y_train),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }

def save_metrics(metrics, filename):
    """Save metrics to a CSV file."""
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f"../../reports/results/{filename}.csv", index=False)
    print(f"Metrics saved to ../../reports/results/{filename}.csv")

# Load and preprocess data
def load_and_preprocess_data(
    filepath, rolling_window=7, start_date=None, end_date=None
):
    """Load and preprocess the data."""
    df = pd.read_csv(filepath)

    # Parse the 'date' column
    df["date"] = pd.to_datetime(df["date"])  # Automatically infer date format

    required_columns = [
        "date",
        "population",
        "cumulative_confirmed",
        "cumulative_deceased",
        "new_confirmed",
        "new_deceased",
        "cumAdmissions",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "newAdmissions",
    ]
    df = df[required_columns].copy()

    # Compute daily new values
    df["daily_confirmed"] = df["cumulative_confirmed"].diff().clip(lower=0).fillna(0)
    df["daily_deceased"] = df["cumulative_deceased"].diff().clip(lower=0).fillna(0)
    df["daily_hospitalized"] = df["cumAdmissions"].diff().clip(lower=0).fillna(0)

    # Apply rolling average
    rolling_cols = [
        "daily_confirmed",
        "daily_deceased",
        "daily_hospitalized",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "newAdmissions",
    ]
    df[rolling_cols] = (
        df[rolling_cols].rolling(window=rolling_window, min_periods=1).mean()
    )

    # Filter by date
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    df.reset_index(drop=True, inplace=True)
    return df

data = load_and_preprocess_data(
    filepath="../../data/processed/england_data.csv",
    rolling_window=7,
    start_date="2020-05-01",
    end_date="2021-12-31",
)

# Verify that dates are parsed correctly
data.head()

# Split data into training and validation sets
features = [
    "daily_confirmed",
    "daily_hospitalized",
    "covidOccupiedMVBeds",
    "daily_deceased",
]

train_data = data[:-7]
val_data = data[-7:]

# Scaling
def scale_data(train_df, val_df, features):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[features])
    val_scaled = scaler.transform(val_df[features])
    return train_scaled, val_scaled, scaler

train_scaled, val_scaled, scaler = scale_data(train_data, val_data, features)

# Prepare tensors
def prepare_tensors(scaled_data, device):
    """Prepare tensors for training."""
    data_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)
    return data_tensor

train_data_tensor = prepare_tensors(train_scaled, device)

# Prepare time stamps and normalize
time_stamps = torch.tensor(train_data.index.values, dtype=torch.float32).view(-1, 1)
t_min = time_stamps.min()
t_max = time_stamps.max()
time_stamps_normalized = (time_stamps - t_min) / (t_max - t_min)
time_stamps_normalized = time_stamps_normalized.to(device)
time_stamps_normalized.requires_grad_(True)  # Ensure requires_grad=True before using t

# Model definitions
class ResidualBlock(nn.Module):
    """Residual Block with Tanh activation."""

    def __init__(self, hidden_neurons):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
        )
        self.activation = nn.Tanh()
        self.init_weights()

    def forward(self, x):
        return self.activation(x + self.block(x))

    def init_weights(self):
        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  # Initialize biases to 0.01

        self.apply(initialize_weights)


class EpiNet(nn.Module):
    """Physics-Informed Neural Network for epidemiological modeling."""

    def __init__(self, num_layers=5, hidden_neurons=20, output_size=8):
        super().__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers):
            block = ResidualBlock(hidden_neurons)
            layers.append(block)
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.network = nn.Sequential(*layers)
        self.init_weights()

        # Constants as learnable parameters
        self._rho = nn.Parameter(torch.rand(1))
        self._alpha = nn.Parameter(torch.rand(1))
        self._ds = nn.Parameter(torch.rand(1))
        self._da = nn.Parameter(torch.rand(1))
        self._dH = nn.Parameter(torch.rand(1))

    def forward(self, t):
        out = self.network(t)
        return out  # Do not apply activation functions here

    @property
    def rho(self):
        return torch.sigmoid(self._rho)

    @property
    def alpha(self):
        return torch.sigmoid(self._alpha)

    @property
    def ds(self):
        return torch.sigmoid(self._ds)

    @property
    def da(self):
        return torch.sigmoid(self._da)

    @property
    def dH(self):
        return torch.sigmoid(self._dH)

    def init_weights(self):
        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  # Initialize biases to 0.01

        self.apply(initialize_weights)


class ParameterNet(nn.Module):
    """Neural Network to predict time-varying parameters."""

    def __init__(self, num_layers=3, hidden_neurons=32, output_size=6):
        super().__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.network = nn.Sequential(*layers)
        self.init_weights()

    def forward(self, t):
        raw_parameters = self.network(t)
        # Apply transformations
        beta = torch.exp(raw_parameters[:, 0])  # Positive and unbounded
        gamma_c = torch.sigmoid(raw_parameters[:, 1])  # Between 0 and 1
        delta_c = torch.sigmoid(raw_parameters[:, 2])
        eta = torch.sigmoid(raw_parameters[:, 3])
        mu = torch.sigmoid(raw_parameters[:, 4])
        omega = torch.sigmoid(raw_parameters[:, 5])
        return beta, gamma_c, delta_c, eta, mu, omega

    def init_weights(self):
        def initialize_weights(m):
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(m.weight, gain=gain)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  # Initialize biases to 0.01

        self.apply(initialize_weights)

# Define SEIRD model
def seird_model(y, N, beta, alpha, rho, ds, da, omega, dH, mu, gamma_c, delta_c, eta):
    S, E, Is, Ia, H, C, R, D = y

    dSdt = -beta * (Is + Ia) / N * S + eta * R
    dEdt = beta * (Is + Ia) / N * S - alpha * E
    dIsdt = alpha * rho * E - ds * Is
    dIadt = alpha * (1 - rho) * E - da * Ia
    dHdt = ds * omega * Is - dH * H - mu * H
    dCdt = dH * (1 - omega) * H - gamma_c * C - delta_c * C
    dRdt = ds * (1 - omega) * Is + da * Ia + gamma_c * C - eta * R
    dDdt = mu * H + delta_c * C

    return dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt

# Define loss function
def pinn_loss(model_output, data_tensor, parameters, t, constants):
    """Calculate the physics-informed neural network loss."""
    # Apply activations to constrain variables
    S_pred = torch.sigmoid(model_output[:, 0])  # Between 0 and 1
    E_pred = torch.relu(model_output[:, 1])  # Non-negative
    Is_pred = torch.relu(model_output[:, 2])  # Non-negative
    Ia_pred = torch.relu(model_output[:, 3])  # Non-negative
    H_pred = torch.relu(model_output[:, 4])  # Non-negative
    C_pred = torch.relu(model_output[:, 5])  # Non-negative
    R_pred = torch.relu(model_output[:, 6])  # Non-negative
    D_pred = torch.relu(model_output[:, 7])  # Non-negative

    Is_data, H_data, C_data, D_data = data_tensor.T

    N = 1  # Normalized population
    rho, alpha, ds, da, dH = constants
    beta_pred, gamma_c_pred, delta_c_pred, eta_pred, mu_pred, omega_pred = parameters

    # Ensure t is correctly shaped and has requires_grad=True
    if not t.requires_grad:
        t.requires_grad_(True)

    # Compute time derivatives
    S_t = grad(
        S_pred.sum(), t, create_graph=True, retain_graph=True
    )[0]
    E_t = grad(
        E_pred.sum(), t, create_graph=True, retain_graph=True
    )[0]
    Is_t = grad(
        Is_pred.sum(), t, create_graph=True, retain_graph=True
    )[0]
    Ia_t = grad(
        Ia_pred.sum(), t, create_graph=True, retain_graph=True
    )[0]
    H_t = grad(
        H_pred.sum(), t, create_graph=True, retain_graph=True
    )[0]
    C_t = grad(
        C_pred.sum(), t, create_graph=True, retain_graph=True
    )[0]
    R_t = grad(
        R_pred.sum(), t, create_graph=True, retain_graph=True
    )[0]
    D_t = grad(
        D_pred.sum(), t, create_graph=True, retain_graph=True
    )[0]

    # Ensure that gradients are not None
    derivatives = [S_t, E_t, Is_t, Ia_t, H_t, C_t, R_t, D_t]
    for i, deriv in enumerate(derivatives):
        if deriv is None:
            raise RuntimeError(
                f"Gradient of state variable {i} is None. Check if outputs depend on t."
            )

    # SEIRD model equations
    dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt = seird_model(
        [S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred],
        N,
        beta_pred,
        alpha,
        rho,
        ds,
        da,
        omega_pred,
        dH,
        mu_pred,
        gamma_c_pred,
        delta_c_pred,
        eta_pred,
    )

    # Compute residuals
    residuals = [
        S_t - dSdt,
        E_t - dEdt,
        Is_t - dIsdt,
        Ia_t - dIadt,
        H_t - dHdt,
        C_t - dCdt,
        R_t - dRdt,
        D_t - dDdt,
    ]

    residual_loss = sum(torch.mean(r**2) for r in residuals)

    # Compute predicted new cases and deaths (incidence)
    new_cases_pred = alpha * rho * E_pred  # New symptomatic infections
    new_deaths_pred = mu_pred * H_pred + delta_c_pred * C_pred  # New deaths
    new_hospital_admissions_pred = ds * omega_pred * Is_pred  # New hospital admissions

    # Data loss
    data_loss = (
        torch.mean((new_cases_pred - Is_data) ** 2)
        + torch.mean((new_hospital_admissions_pred - H_data) ** 2)
        + torch.mean((C_pred - C_data) ** 2)
        + torch.mean((new_deaths_pred - D_data) ** 2)
    )

    # Initial condition loss
    initial_cost = (
        torch.mean((Is_pred[0] - Is_data[0]) ** 2)
        + torch.mean((H_pred[0] - H_data[0]) ** 2)
        + torch.mean((C_pred[0] - C_data[0]) ** 2)
        + torch.mean((D_pred[0] - D_data[0]) ** 2)
    )

    # Total loss
    loss = data_loss + residual_loss + initial_cost
    return loss

# Early stopping
class EarlyStopping:
    """Early stopping to stop the training when the loss does not improve."""

    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.counter = 0

# Training loop
def train_model(
    model,
    parameter_net,
    optimizer,
    scheduler,
    time_stamps,
    data_tensor,
    num_epochs=1000,
    early_stopping=None,
):
    """Train the model with early stopping and learning rate scheduling."""
    train_losses = []

    progress_bar = trange(num_epochs, desc="Training", unit="epoch")
    for epoch in progress_bar:
        model.train()
        parameter_net.train()
        optimizer.zero_grad()

        # Forward pass
        model_output = model(time_stamps)
        parameters = parameter_net(time_stamps)
        constants = (model.rho, model.alpha, model.ds, model.da, model.dH)

        # Compute loss
        loss = pinn_loss(model_output, data_tensor, parameters, time_stamps, constants)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_losses.append(train_loss)
        scheduler.step(train_loss)

        # Update progress bar
        progress_bar.set_postfix(loss=train_loss)

        if early_stopping:
            early_stopping(train_loss)
            if early_stopping.early_stop:
                progress_bar.write("Early stopping triggered.")
                break

    return train_losses

# Initialize models, optimizer, scheduler
model = EpiNet(num_layers=5, hidden_neurons=20, output_size=8).to(device)
parameter_net = ParameterNet(num_layers=3, hidden_neurons=32, output_size=6).to(device)

optimizer = optim.Adam(
    list(model.parameters()) + list(parameter_net.parameters()), lr=1e-4
)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=50, verbose=True
)

early_stopping = EarlyStopping(patience=100, verbose=True)

# Ensure time_stamps have requires_grad=True before training
time_stamps_normalized.requires_grad_(True)

# Train the model
with torch.autograd.set_detect_anomaly(True):
    train_losses = train_model(
        model,
        parameter_net,
        optimizer,
        scheduler,
        time_stamps_normalized,
        train_data_tensor,
        num_epochs=50000,
        early_stopping=early_stopping,
    )

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.yscale("log")
plt.savefig("../../reports/figures/training_loss.png")
plt.show()

# Save the model
torch.save(model.state_dict(), "../../models/epinet_model.pth")
torch.save(parameter_net.state_dict(), "../../models/parameter_net.pth")

# Load the model (optional, for future use)
model.load_state_dict(torch.load("../../models/epinet_model.pth"))
parameter_net.load_state_dict(torch.load("../../models/parameter_net.pth"))

# Evaluate the model on training data
model.eval()
parameter_net.eval()
with torch.no_grad():
    model_output = model(time_stamps_normalized)
    parameters = parameter_net(time_stamps_normalized)

# Extract model outputs
S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred = model_output.T
rho, alpha, ds, da, dH = model.rho, model.alpha, model.ds, model.da, model.dH
beta_pred, gamma_c_pred, delta_c_pred, eta_pred, mu_pred, omega_pred = parameters

# Apply activations to outputs
S_pred = torch.sigmoid(S_pred)
E_pred = torch.relu(E_pred)
Is_pred = torch.relu(Is_pred)
Ia_pred = torch.relu(Ia_pred)
H_pred = torch.relu(H_pred)
C_pred = torch.relu(C_pred)
R_pred = torch.relu(R_pred)
D_pred = torch.relu(D_pred)

# Compute predicted incidence
new_cases_pred = alpha * rho * E_pred  # New symptomatic infections
new_deaths_pred = mu_pred * H_pred + delta_c_pred * C_pred  # New deaths
new_hospital_admissions_pred = ds * omega_pred * Is_pred  # New hospital admissions

# Prepare observed and predicted data for plotting
observed_data = train_data[features].reset_index(drop=True)
predicted_data = np.stack(
    [
        new_cases_pred.cpu().detach().numpy(),
        new_hospital_admissions_pred.cpu().detach().numpy(),
        C_pred.cpu().detach().numpy(),
        new_deaths_pred.cpu().detach().numpy(),
    ],
    axis=1,
)

# Inverse transform predicted data
predicted_data_inverse = scaler.inverse_transform(predicted_data)
predicted_df = pd.DataFrame(predicted_data_inverse, columns=features)

# Plot observed vs predicted
def plot_results(dates, observed, predicted, variables, filename):
    """Plot observed vs predicted data."""
    num_vars = len(variables)
    fig, axs = plt.subplots(
        nrows=(num_vars + 1) // 2, ncols=2, figsize=(15, num_vars * 3)
    )
    axs = axs.flatten()

    for idx, var in enumerate(variables):
        axs[idx].plot(dates, observed[var], label="Observed", color="blue")
        axs[idx].plot(
            dates, predicted[var], label="Predicted", linestyle="--", color="red"
        )
        axs[idx].set_title(var)
        axs[idx].legend()
        axs[idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{filename}.png")
    plt.show()

dates = train_data["date"]
plot_results(dates, observed_data, predicted_df, features, "observed_vs_predicted")

# Compute error metrics on training data
train_data_values = [train_data[var].values for var in features]
metrics = []

for var, y_true, y_pred in zip(features, train_data_values, predicted_data_inverse.T):
    errors = calculate_errors(y_true, y_pred, y_train=y_true)
    metrics.append({"Variable": var, **errors})

save_metrics(metrics, "training_metrics")


# Evaluate on validation data
val_time_stamps = torch.tensor(val_data.index.values, dtype=torch.float32).view(-1, 1)
val_time_stamps_normalized = (val_time_stamps - t_min) / (t_max - t_min)
val_time_stamps_normalized = val_time_stamps_normalized.to(device)
val_time_stamps_normalized.requires_grad_(True)

# After loading the model outputs and parameters for validation data
with torch.no_grad():
    val_model_output = model(val_time_stamps_normalized)
    val_parameters = parameter_net(val_time_stamps_normalized)

# Extract model outputs
(S_pred_val, E_pred_val, Is_pred_val, Ia_pred_val, H_pred_val, C_pred_val, R_pred_val, D_pred_val) = val_model_output.T

# Extract validation parameters
(beta_pred_val, gamma_c_pred_val, delta_c_pred_val, eta_pred_val, mu_pred_val, omega_pred_val) = val_parameters

# Apply activations to outputs (remains the same)
S_pred_val = torch.sigmoid(S_pred_val)
E_pred_val = torch.relu(E_pred_val)
Is_pred_val = torch.relu(Is_pred_val)
Ia_pred_val = torch.relu(Ia_pred_val)
H_pred_val = torch.relu(H_pred_val)
C_pred_val = torch.relu(C_pred_val)
R_pred_val = torch.relu(R_pred_val)
D_pred_val = torch.relu(D_pred_val)

# Compute predicted incidence using validation parameters
new_cases_pred_val = alpha * rho * E_pred_val  # Constants can be used directly
new_deaths_pred_val = mu_pred_val * H_pred_val + delta_c_pred_val * C_pred_val
new_hospital_admissions_pred_val = ds * omega_pred_val * Is_pred_val


# Prepare observed and predicted data for plotting
observed_val_data = val_data[features].reset_index(drop=True)
predicted_val_data = np.stack(
    [
        new_cases_pred_val.cpu().detach().numpy(),
        new_hospital_admissions_pred_val.cpu().detach().numpy(),
        C_pred_val.cpu().detach().numpy(),
        new_deaths_pred_val.cpu().detach().numpy(),
    ],
    axis=1,
)

# Inverse transform predicted data
predicted_val_data_inverse = scaler.inverse_transform(predicted_val_data)
predicted_val_df = pd.DataFrame(predicted_val_data_inverse, columns=features)

# Plot validation results
dates_val = val_data["date"]
plot_results(
    dates_val, observed_val_data, predicted_val_df, features, "validation_results"
)

# Compute error metrics on validation data
val_data_values = [val_data[var].values for var in features]
metrics_val = []

for var, y_true, y_pred in zip(features, val_data_values, predicted_val_data_inverse.T):
    errors = calculate_errors(y_true, y_pred, y_train=train_data[var].values)
    metrics_val.append({"Variable": var, **errors})

save_metrics(metrics_val, "validation_metrics")

# Plot time-varying parameters
def plot_parameters(dates, parameters, filename):
    """Plot time-varying parameters."""
    params = ["beta", "gamma_c", "delta_c", "eta", "mu", "omega"]
    num_params = len(params)
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
    axs = axs.flatten()

    for idx, (param, values) in enumerate(zip(params, parameters)):
        axs[idx].plot(dates, values.cpu().numpy(), label=param, color=f"C{idx}")
        axs[idx].set_title(f"Time-varying {param}")
        axs[idx].set_xlabel("Date")
        axs[idx].set_ylabel(param)
        axs[idx].tick_params(axis="x", rotation=45)
        axs[idx].legend()

    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{filename}.png")
    plt.show()

parameters_np = [p.cpu().detach() for p in parameters]
plot_parameters(dates, parameters_np, "time_varying_parameters")

# Plot parameters for validation data
parameters_val_np = [p.cpu().detach() for p in val_parameters]
plot_parameters(dates_val, parameters_val_np, "time_varying_parameters_validation")