# ================================================================
# Import Necessary Libraries
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # For better plotting styles
from tqdm.notebook import tqdm  # For progress bars
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

# ================================================================
# Setup and Configuration
# ================================================================

# Ensure the folders exist for saving outputs
os.makedirs("../../models", exist_ok=True)
os.makedirs("../../reports/figures", exist_ok=True)
os.makedirs("../../reports/results", exist_ok=True)
os.makedirs("../../reports/parameters", exist_ok=True)

# Device setup for CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# ================================================================
# Plotting Configuration
# ================================================================

plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams.update(
    {
        "font.size": 14,
        "font.weight": "bold",
        "figure.figsize": [10, 4],
        "text.usetex": False,
        "figure.facecolor": "white",
        "figure.autolayout": True,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.titlesize": 18,
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "legend.loc": "best",
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "xtick.labelsize": 12,
        "xtick.direction": "in",
        "xtick.top": False,
        "ytick.labelsize": 12,
        "ytick.direction": "in",
        "ytick.right": False,
        "grid.color": "grey",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "errorbar.capsize": 4,
        "figure.subplot.wspace": 0.4,
        "figure.subplot.hspace": 0.4,
        "image.cmap": "viridis",
    }
)

# ================================================================
# Error Metrics Functions
# ================================================================

def normalized_root_mean_square_error(y_true, y_pred):
    """Calculate the Normalized Root Mean Square Error (NRMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

def safe_mean_absolute_scaled_error(y_true, y_pred, y_train, epsilon=1e-10):
    """Calculate the Mean Absolute Scaled Error (MASE) safely."""
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    d = max(d, epsilon)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

def calculate_errors(y_true, y_pred, y_train, train_size, area_name):
    """Calculate and print various error metrics."""
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    nrmse = normalized_root_mean_square_error(y_true, y_pred)
    mase = safe_mean_absolute_scaled_error(y_true, y_pred, y_train)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"Normalized Root Mean Square Error (NRMSE): {nrmse:.4f}")
    print(f"Mean Absolute Scaled Error (MASE): {mase:.4f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    return mape, nrmse, mase, rmse, mae, mse

# ================================================================
# SEIRD Model Definition
# ================================================================

def seird_model(y, t, N, beta, alpha, rho, ds, da, omega, dH, mu, gamma_c, delta_c, eta):
    """Define the SEIRD model differential equations."""
    S, E, Is, Ia, H, C, R, D = y

    dSdt = -beta * (Is + Ia) / N * S + eta * R
    dEdt = beta * (Is + Ia) / N * S - alpha * E
    dIsdt = alpha * rho * E - ds * Is
    dIadt = alpha * (1 - rho) * E - da * Ia
    dHdt = ds * omega * Is - dH * H - mu * H
    dCdt = dH * (1 - omega) * H - gamma_c * C - delta_c * C
    dRdt = ds * (1 - omega) * Is + da * Ia + dH * (1 - mu) * H + gamma_c * C - eta * R
    dDdt = mu * H + delta_c * C

    return [dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt]

# ================================================================
# Data Loading and Preprocessing
# ================================================================

area_name = "East of England"

def load_preprocess_data(filepath, area_name, rolling_window=7, start_date="2020-04-01", end_date=None):
    """Load and preprocess the COVID-19 data."""
    df = pd.read_csv(filepath)

    # Select the data for the specified area
    df = df[df["areaName"] == area_name].reset_index(drop=True)

    # Convert the date column to datetime format
    df["date"] = pd.to_datetime(df["date"])

    # Compute daily new values from cumulative values
    df["daily_confirmed"] = df["cumulative_confirmed"].diff().fillna(0)
    df["daily_deceased"] = df["cumulative_deceased"].diff().fillna(0)
    df["daily_hospitalized"] = df["cumAdmissions"].diff().fillna(0)

    # Ensure no negative values
    df["daily_confirmed"] = df["daily_confirmed"].clip(lower=0)
    df["daily_deceased"] = df["daily_deceased"].clip(lower=0)
    df["daily_hospitalized"] = df["daily_hospitalized"].clip(lower=0)

    required_columns = [
        "date", "population", "cumulative_confirmed", "cumulative_deceased",
        "new_confirmed", "new_deceased", "cumAdmissions", "daily_confirmed",
        "daily_deceased", "daily_hospitalized", "hospitalCases",
        "covidOccupiedMVBeds", "newAdmissions"
    ]

    # Select required columns
    df = df[required_columns]

    # Apply rolling average to smooth out data
    for col in required_columns[2:]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    # Filter data between start_date and end_date
    mask = df["date"] >= start_date
    if end_date:
        mask &= df["date"] <= end_date
    df = df.loc[mask].reset_index(drop=True)

    return df

# Load and preprocess the data
data = load_preprocess_data("../../data/processed/merged_nhs_covid_data.csv", area_name, rolling_window=7, start_date="2020-05-01", end_date="2021-12-31")

# ================================================================
# Data Visualization for Daily Confirmed Cases
# ================================================================
# Plotting new deceased cases over time
plt.plot(data["date"], data["daily_deceased"])
plt.title("New Daily Deceased over time")
plt.xlabel("Date")
plt.ylabel("New Deceased")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================================================
# Data Splitting and Scaling
# ================================================================

# Split data into training and validation sets; validation is the last 7 days
train_data = data[:-7].reset_index(drop=True)
val_data = data[-7:].reset_index(drop=True)

def scale_data(train_data, val_data, features, device):
    """Scale training and validation data using the same scaler."""
    scaler = MinMaxScaler()
    scaler.fit(train_data[features])

    # Scale the training data
    train_data_scaled = scaler.transform(train_data[features])
    train_data_scaled = torch.tensor(train_data_scaled, dtype=torch.float32).to(device)

    # Scale the validation data
    val_data_scaled = scaler.transform(val_data[features])
    val_data_scaled = torch.tensor(val_data_scaled, dtype=torch.float32).to(device)

    return train_data_scaled, val_data_scaled, scaler

# Features to be used in the model
features = [
    "daily_confirmed",       # New confirmed cases per day
    "daily_hospitalized",    # New hospitalizations per day
    "covidOccupiedMVBeds",   # Number of occupied mechanical ventilation beds
    "daily_deceased",        # New deceased cases per day
]

# Scale the data
train_data_scaled, val_data_scaled, scaler = scale_data(train_data, val_data, features, device)

# ================================================================
# Neural Network Model Definitions
# ================================================================

class ResidualBlock(nn.Module):
    """Residual Block with two linear layers and a skip connection."""
    def __init__(self, hidden_neurons):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Linear(hidden_neurons, hidden_neurons),
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(x + self.layer(x))

class EpiNet(nn.Module):
    """Neural network to approximate the solution of the SEIRD model."""
    def __init__(self, num_layers=5, hidden_neurons=20, output_size=8):
        super(EpiNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        # Hidden layers with residual connections
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_neurons))

        # Output layer
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)

        # Initialize parameters with constraints
        self._rho = nn.Parameter(torch.tensor([0.8], device=device), requires_grad=True)
        self._alpha = nn.Parameter(torch.tensor([1 / 5], device=device), requires_grad=True)
        self._ds = nn.Parameter(torch.tensor([1 / 4], device=device), requires_grad=True)
        self._da = nn.Parameter(torch.tensor([1 / 7], device=device), requires_grad=True)
        self._dH = nn.Parameter(torch.tensor([1 / 13.4], device=device), requires_grad=True)

        # Initialize weights using Xavier initialization
        self.init_xavier()

    def forward(self, t):
        return torch.sigmoid(self.net(t))

    # Properties to ensure parameters remain within valid ranges
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

    def get_constants(self):
        """Retrieve the constants for the model."""
        return self.rho, self.alpha, self.ds, self.da, self.dH

    def init_xavier(self):
        """Initialize network weights using Xavier initialization."""
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)
        self.net.apply(init_weights)

class ParameterNet(nn.Module):
    """Neural network to estimate time-varying parameters."""
    def __init__(self, num_layers=5, hidden_neurons=20, output_size=6):
        super(ParameterNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Build the network layers
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self.init_xavier()

    def forward(self, t):
        raw_parameters = self.net(t)

        # Apply activation functions and scaling to specific ranges
        beta = torch.sigmoid(raw_parameters[:, 0])   # Scale beta to (0, 1)
        gamma_c = torch.sigmoid(raw_parameters[:, 1])  # Scale gamma_c to (0, 1)
        delta_c = torch.sigmoid(raw_parameters[:, 2])  # Scale delta_c to (0, 1)
        eta = torch.sigmoid(raw_parameters[:, 3])       # Scale eta to (0, 0.1)
        mu = torch.sigmoid(raw_parameters[:, 4])    # Scale mu to (0, 0.1)
        omega = torch.sigmoid(raw_parameters[:, 5])    # Scale omega to (0, 1)

        return beta, gamma_c, delta_c, eta, mu, omega

    def init_xavier(self):
        """Initialize network weights using Xavier initialization."""
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)
        self.net.apply(init_weights)

# ================================================================
# Loss Function Definition
# ================================================================

def einn_loss(model_output, tensor_data, parameters, t, constants):
    """Calculate the EpiNet loss combining data loss, residual loss, and initial condition loss."""
    # Unpack model outputs
    S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred = (
        model_output[:, 0], model_output[:, 1], model_output[:, 2], model_output[:, 3],
        model_output[:, 4], model_output[:, 5], model_output[:, 6], model_output[:, 7],
    )

    Is_data, H_data, C_data, D_data = tensor_data[:, 0], tensor_data[:, 1], tensor_data[:, 2], tensor_data[:, 3]

    N = 1  # Population normalized to 1
    rho, alpha, ds, da, dH = constants

    beta_pred, gamma_c_pred, delta_c_pred, eta_pred, mu_pred, omega_pred = parameters

    # Calculate time derivatives
    S_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    E_t = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]
    Ia_t = grad(Ia_pred, t, grad_outputs=torch.ones_like(Ia_pred), create_graph=True)[0]
    Is_t = grad(Is_pred, t, grad_outputs=torch.ones_like(Is_pred), create_graph=True)[0]
    H_t = grad(H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    C_t = grad(C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
    R_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    D_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]

    # SEIRD model equations
    dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt = seird_model(
        [S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred],
        t, N, beta_pred, alpha, rho, ds, da, omega_pred, dH, mu_pred, gamma_c_pred, delta_c_pred, eta_pred
    )

    # Data loss: MSE between predictions and actual data
    data_loss = (
        torch.mean((Is_pred - Is_data) ** 2) +
        torch.mean((H_pred - H_data) ** 2) +
        torch.mean((C_pred - C_data) ** 2) +
        torch.mean((D_pred - D_data) ** 2)
    )

    # Residual loss: Enforces that model predictions follow the SEIRD differential equations
    residual_loss = (
        torch.mean((S_t - dSdt) ** 2) +
        torch.mean((E_t - dEdt) ** 2) +
        torch.mean((Is_t - dIsdt) ** 2) +
        torch.mean((Ia_t - dIadt) ** 2) +
        torch.mean((H_t - dHdt) ** 2) +
        torch.mean((C_t - dCdt) ** 2) +
        torch.mean((R_t - dRdt) ** 2) +
        torch.mean((D_t - dDdt) ** 2)
    )

    # Initial condition loss: Ensures that the initial model states match the observed data
    S0 = 1.0  # Assuming total population is normalized to 1
    E0 = 0.0
    Ia0 = 0.0
    R0 = 0.0

    initial_loss = (
        (S_pred[0] - S0) ** 2 +
        (E_pred[0] - E0) ** 2 +
        (Is_pred[0] - Is_data[0]) ** 2 +
        (Ia_pred[0] - Ia0) ** 2 +
        (H_pred[0] - H_data[0]) ** 2 +
        (C_pred[0] - C_data[0]) ** 2 +
        (R_pred[0] - R0) ** 2 +
        (D_pred[0] - D_data[0]) ** 2
    )

    loss = data_loss + residual_loss + initial_loss
    return loss

# ================================================================
# Early Stopping Class
# ================================================================

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=50, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# ================================================================
# Training Function
# ================================================================

def train_model(model, parameter_net, optimizer, scheduler, time_stamps, data_scaled, num_epochs=5000, early_stopping=None):
    """Train the EpiNet model."""
    train_losses = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        parameter_net.train()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        t = time_stamps.to(device).float()
        data = data_scaled.to(device).float()
        model_output = model(t)
        parameters = parameter_net(t)
        constants = model.get_constants()

        # Compute loss
        loss = einn_loss(model_output, data, parameters, t, constants)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log training loss
        train_loss = loss.item()
        train_losses.append(train_loss)

        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}")

        # Check early stopping
        if early_stopping:
            early_stopping(train_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return train_losses

# ================================================================
# Model Initialization and Training
# ================================================================

# Initialize model, optimizer, and scheduler
model = EpiNet(num_layers=6, hidden_neurons=20, output_size=8).to(device)
parameter_net = ParameterNet(num_layers=5, hidden_neurons=20, output_size=6).to(device)

optimizer = optim.Adam(list(model.parameters()) + list(parameter_net.parameters()), lr=1e-4)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.8)

# Early stopping
early_stopping = EarlyStopping(patience=100, verbose=True)

# Create timestamps tensor
time_stamps = torch.tensor(train_data.index.values, dtype=torch.float32).view(-1, 1).to(device).requires_grad_()

# Train the model
train_losses = train_model(model, parameter_net, optimizer, scheduler, time_stamps, train_data_scaled, num_epochs=50000, early_stopping=early_stopping)

# ================================================================
# Training Loss Visualization
# ================================================================

# Plot the loss history
plt.figure(figsize=(10, 6))
plt.plot(np.log10(train_losses), label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Log10(Loss)")
plt.title("Training Loss History")
plt.legend()
plt.show()

# ================================================================
# Model Evaluation and Predictions
# ================================================================

model.eval()
parameter_net.eval()

with torch.no_grad():
    # Prepare the full time stamps tensor for the entire dataset
    time_stamps_full = torch.tensor(data.index.values, dtype=torch.float32).view(-1, 1).to(device).requires_grad_()

    # Get model outputs and parameters
    model_output_full = model(time_stamps_full)
    parameters_full = parameter_net(time_stamps_full)

    # Unpack parameters
    beta_pred, gamma_c_pred, delta_c_pred, eta_pred, mu_pred, omega_pred = parameters_full

    # Prepare DataFrame for inverse scaling
    observed_model_output = pd.DataFrame({
        "daily_confirmed": model_output_full[:, 2].cpu().numpy(),
        "daily_hospitalized": model_output_full[:, 4].cpu().numpy(),
        "covidOccupiedMVBeds": model_output_full[:, 5].cpu().numpy(),
        "daily_deceased": model_output_full[:, 7].cpu().numpy(),
    }, index=data.index)

# Inverse transform the model outputs to original scale
observed_model_output_scaled = scaler.inverse_transform(observed_model_output)
observed_model_output_scaled = pd.DataFrame(
    observed_model_output_scaled,
    columns=["daily_confirmed", "daily_hospitalized", "covidOccupiedMVBeds", "daily_deceased"],
    index=data.index
)

# Define unobserved model outputs (compartments not directly observed)
unobserved_model_output = pd.DataFrame({
    "S": model_output_full[:, 0].cpu().numpy(),
    "E": model_output_full[:, 1].cpu().numpy(),
    "Ia": model_output_full[:, 3].cpu().numpy(),
    "R": model_output_full[:, 6].cpu().numpy(),
}, index=data.index)

# ================================================================
# Validation Metrics Calculation
# ================================================================

# Prepare validation data
val_indices = val_data.index.values
val_observed = val_data[features].reset_index(drop=True)
val_predicted = observed_model_output_scaled.loc[val_indices].reset_index(drop=True)

# Calculate error metrics on validation data
print("\nValidation Metrics:")
for col in features:
    print(f"\nMetrics for {col}:")
    calculate_errors(
        val_observed[col].values,
        val_predicted[col].values,
        train_data[col].values,
        train_size=len(train_data),
        area_name=area_name,
    )

# ================================================================
# Plotting Observed vs Predicted Data
# ================================================================

fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

def plot_observed_vs_predicted(ax, data, observed_model_output_scaled, variable, ylabel):
    ax.plot(data["date"], data[variable], label="Observed", color="blue")
    ax.plot(data["date"], observed_model_output_scaled[variable], label="Predicted", color="red", linestyle="--")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

plot_observed_vs_predicted(axs[0], data, observed_model_output_scaled, "daily_confirmed", "Daily Confirmed Cases")
plot_observed_vs_predicted(axs[1], data, observed_model_output_scaled, "daily_hospitalized", "Daily Hospitalizations")
plot_observed_vs_predicted(axs[2], data, observed_model_output_scaled, "covidOccupiedMVBeds", "Occupied MV Beds")
plot_observed_vs_predicted(axs[3], data, observed_model_output_scaled, "daily_deceased", "Daily Deaths")

fig.suptitle("Observed vs Predicted Data", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ================================================================
# Plotting Unobserved Model Outputs
# ================================================================

fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

def plot_unobserved_data(ax, unobserved_model_output, variable, ylabel):
    ax.plot(data["date"], unobserved_model_output[variable], label="Predicted", color="green")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

plot_unobserved_data(axs[0], unobserved_model_output, "S", "Susceptible (S)")
plot_unobserved_data(axs[1], unobserved_model_output, "E", "Exposed (E)")
plot_unobserved_data(axs[2], unobserved_model_output, "Ia", "Asymptomatic Infectious (Ia)")
plot_unobserved_data(axs[3], unobserved_model_output, "R", "Recovered (R)")

fig.suptitle("Unobserved Data Predictions", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ================================================================
# Time-Varying Parameter Estimation Visualization
# ================================================================

# Create DataFrame for parameter estimations
parameter_estimation = pd.DataFrame({
    "beta": beta_pred.cpu().numpy(),
    "gamma_c": gamma_c_pred.cpu().numpy(),
    "delta_c": delta_c_pred.cpu().numpy(),
    "eta": eta_pred.cpu().numpy(),
    "mu": mu_pred.cpu().numpy(),
    "omega": omega_pred.cpu().numpy(),
}, index=data.index)

# Apply smoothing to parameters for better visualization
parameter_estimation_smooth = parameter_estimation.rolling(window=7, min_periods=1).mean()

# Plotting time-varying parameters in a 3x2 layout
fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

def plot_time_varying_parameters(ax, parameter_estimation, parameter, ylabel):
    ax.plot(data["date"], parameter_estimation[parameter], label="Estimated", color="purple")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

plot_time_varying_parameters(axs[0, 0], parameter_estimation_smooth, "beta", r"$\beta$")
plot_time_varying_parameters(axs[0, 1], parameter_estimation_smooth, "gamma_c", r"$\gamma_c$")
plot_time_varying_parameters(axs[1, 0], parameter_estimation_smooth, "delta_c", r"$\delta_c$")
plot_time_varying_parameters(axs[1, 1], parameter_estimation_smooth, "eta", r"$\eta$")
plot_time_varying_parameters(axs[2, 0], parameter_estimation_smooth, "mu", r"$\mu$")
plot_time_varying_parameters(axs[2, 1], parameter_estimation_smooth, "omega", r"$\omega$")

fig.suptitle("Time-Varying Parameter Estimation", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ================================================================
# Effective and Basic Reproduction Number Calculation and Plotting
# ================================================================

# Detach tensors for calculations
beta_pred = beta_pred.detach()
mu_pred = mu_pred.detach()
ds = model.ds.detach()
da = model.da.detach()
rho = model.rho.detach()

# Calculate Effective Reproduction Number R_t
# For SEIRD model with compartments, R_t can be approximated as:
# R_t = beta * [rho / (ds + mu)] + beta * [(1 - rho) / da]
Rt = beta_pred * (rho / (ds + mu_pred) + (1 - rho) / da)

# Calculate Basic Reproduction Number R_0 (assuming no interventions)
R0 = Rt.clone().detach()  # Since we don't have time-varying interventions modeled, R0 ~ Rt

# Plot R_t over time with threshold line at R=1
plt.figure(figsize=(10, 6))
plt.plot(data["date"], Rt.cpu().numpy(), label="$R_t$", color="orange")
plt.axhline(y=1, color='red', linestyle='--', label='Threshold (R=1)')
plt.ylabel("$R_t$")
plt.xlabel("Date")
plt.title("Effective Reproduction Number Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot R_0 over time with threshold line at R=1
plt.figure(figsize=(10, 6))
plt.plot(data["date"], R0.cpu().numpy(), label="$R_0$", color="blue")
plt.axhline(y=1, color='red', linestyle='--', label='Threshold (R=1)')
plt.ylabel("$R_0$")
plt.xlabel("Date")
plt.title("Basic Reproduction Number Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()