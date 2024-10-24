# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Ensure the necessary directories exist
os.makedirs("../../models", exist_ok=True)
os.makedirs("../../reports/figures", exist_ok=True)
os.makedirs("../../reports/results", exist_ok=True)
os.makedirs("../../reports/parameters", exist_ok=True)

# ================================================================
# Device Setup for CUDA or CPU and Seed Initialization
# ================================================================
# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    device = torch.device("cuda")
    print(f"Using {torch.cuda.device_count()} GPUs.")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# ================================================================
# Matplotlib Style and Parameters
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
        "axes.formatter.limits": (0, 5),
        "axes.formatter.use_mathtext": True,
        "axes.formatter.useoffset": False,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
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
# Define Error Metrics Functions
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

def calculate_errors(y_true, y_pred, y_train):
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
# SEIRD Model Differential Equations
# ================================================================
def seird_model(y, t, N, beta, alpha, rho, ds, da, omega, dH, mu, gamma_c, delta_c, eta):
    """Define the SEIRD model differential equations."""
    S, E, Is, Ia, H, C, R, D = y

    dSdt = -beta * (Is + Ia) / N * S + eta * R
    dEdt = beta * (Is + Ia) / N * S - alpha * E
    dIsdt = alpha * rho * E - ds * Is
    dIadt = alpha * (1 - rho) * E - da * Ia
    dHdt = ds * omega * Is - dH * H - mu * H
    dCdt = dH * (1 - omega) * (H - gamma_c * C) - delta_c * C
    dRdt = ds * (1 - omega) * Is + da * Ia + dH * (1 - mu) * H + gamma_c * C - eta * R
    dDdt = mu * H + delta_c * C

    return [dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt]

# ================================================================
# Data Loading and Preprocessing
# ================================================================
def load_preprocess_data(filepath, area_name, rolling_window=7, start_date="2020-04-01", end_date=None):
    """Load and preprocess the COVID-19 data."""
    df = pd.read_csv(filepath)

    # Select the columns of interest
    df = df[df["areaName"] == area_name].reset_index(drop=True)

    # Convert the date column to datetime
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
        "date", "population", "cumulative_confirmed", "cumulative_deceased", "cumAdmissions",
        "daily_confirmed", "daily_deceased", "daily_hospitalized", "hospitalCases", "covidOccupiedMVBeds", "newAdmissions",
    ]

    # Select required columns
    df = df[required_columns]

    # Apply rolling average to smooth out data (except for date and population)
    for col in required_columns[2:]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1, center=False).mean().fillna(0)

    # Select data from start date to end date
    mask = df["date"] >= start_date
    if end_date:
        mask &= df["date"] <= end_date
    df = df.loc[mask].reset_index(drop=True)

    return df

# Load data
area_name = "East of England"
data = load_preprocess_data(
    "../../data/processed/merged_nhs_covid_data.csv",
    area_name,
    rolling_window=7,
    start_date="2020-05-01",
    end_date="2021-12-31",
)

# ================================================================
# Data Splitting and Scaling
# ================================================================
# Features to consider
features = ["daily_confirmed", "daily_hospitalized", "covidOccupiedMVBeds", "daily_deceased"]

# Split data into training and validation sets; the validation is the last 7 days
train_data = data[:-7].reset_index(drop=True)
val_data = data[-7:].reset_index(drop=True)

# Scaling the data
def scale_data(train_data, val_data, features):
    """Scale training and validation data using the same scaler."""
    scaler = MinMaxScaler()
    scaler.fit(train_data[features])

    # Scale the training data
    train_data_scaled = scaler.transform(train_data[features])

    # Scale the validation data
    val_data_scaled = scaler.transform(val_data[features])

    return train_data_scaled, val_data_scaled, scaler

# Scale the data
train_data_scaled, val_data_scaled, scaler = scale_data(train_data, val_data, features)

# Prepare tensors
time_stamps = torch.tensor(train_data.index.values, dtype=torch.float32).view(-1, 1)
train_data_tensor = torch.tensor(train_data_scaled, dtype=torch.float32)

# ================================================================
# Neural Network Models Definition
# ================================================================
class ResidualBlock(nn.Module):
    """Defines a residual block with Tanh activation."""
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
    """EpiNet model for estimating state variables."""
    def __init__(self, num_layers=3, hidden_neurons=10, output_size=8):
        super(EpiNet, self).__init__()

        # Input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        # Hidden layers with residual connections
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_neurons))

        # Output layer
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)

        # Initialize parameters with constraints
        self._rho = nn.Parameter(torch.tensor([0.8]), requires_grad=True)
        self._alpha = nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self._ds = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self._da = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self._dH = nn.Parameter(torch.tensor([0.05]), requires_grad=True)

        # Initialize weights using Xavier initialization
        self.init_xavier()

    def forward(self, t):
        return torch.sigmoid(self.net(t))

    @property
    def rho(self):
        return 0.1 + 0.9 * torch.sigmoid(self._rho)  # Range [0.1, 1.0]

    @property
    def alpha(self):
        return 0.1 + 0.9 * torch.sigmoid(self._alpha)  # Range [0.1, 1.0]

    @property
    def ds(self):
        return 0.1 + 0.9 * torch.sigmoid(self._ds)  # Range [0.1, 1.0]

    @property
    def da(self):
        return 0.1 + 0.9 * torch.sigmoid(self._da)  # Range [0.1, 1.0]

    @property
    def dH(self):
        return 0.1 + 0.9 * torch.sigmoid(self._dH)  # Range [0.1, 1.0]

    def get_constants(self):
        """Retrieve the constants for the model."""
        return self.rho, self.alpha, self.ds, self.da, self.dH

    def init_xavier(self):
        """Initialize weights using Xavier initialization."""
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)
        self.net.apply(init_weights)

class ParameterNet(nn.Module):
    """ParameterNet model for estimating time-varying parameters."""
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=6):
        super(ParameterNet, self).__init__()

        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])

        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)

        self.init_xavier()

    def forward(self, t):
        raw_parameters = self.net(t)
        # Apply scaling to parameters to constrain them within realistic ranges
        beta = 0.1 + 0.9 * torch.sigmoid(raw_parameters[:, 0])        # Range [0.1, 1.0]
        gamma_c = 0.0 + 0.5 * torch.sigmoid(raw_parameters[:, 1])     # Range [0.0, 0.5]
        delta_c = 0.0 + 0.5 * torch.sigmoid(raw_parameters[:, 2])     # Range [0.0, 0.5]
        eta = 0.0 + 0.2 * torch.sigmoid(raw_parameters[:, 3])         # Range [0.0, 0.2]
        mu = 0.0 + 0.1 * torch.sigmoid(raw_parameters[:, 4])          # Range [0.0, 0.1]
        omega = 0.0 + 0.5 * torch.sigmoid(raw_parameters[:, 5])       # Range [0.0, 0.5]
        return beta, gamma_c, delta_c, eta, mu, omega

    def init_xavier(self):
        """Initialize weights using Xavier initialization."""
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)
        self.net.apply(init_weights)

# ================================================================
# Define the PINN Loss Function
# ================================================================
def pinn_loss(model_output, tensor_data, parameters, t, constants):
    """Calculate the Physics-Informed Neural Network loss."""
    S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred = (
        model_output[:, 0],
        model_output[:, 1],
        model_output[:, 2],
        model_output[:, 3],
        model_output[:, 4],
        model_output[:, 5],
        model_output[:, 6],
        model_output[:, 7],
    )

    Is_data, H_data, C_data, D_data = (
        tensor_data[:, 0],
        tensor_data[:, 1],
        tensor_data[:, 2],
        tensor_data[:, 3],
    )

    N = 1  # Since data is normalized

    rho, alpha, ds, da, dH = constants

    beta_pred, gamma_c_pred, delta_c_pred, eta_pred, mu_pred, omega_pred = parameters

    # Compute gradients with respect to time
    S_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    E_t = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]
    Ia_t = grad(Ia_pred, t, grad_outputs=torch.ones_like(Ia_pred), create_graph=True)[0]
    Is_t = grad(Is_pred, t, grad_outputs=torch.ones_like(Is_pred), create_graph=True)[0]
    H_t = grad(H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    C_t = grad(C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
    R_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    D_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]

    # Compute the right-hand side of the differential equations
    dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt = seird_model(
        [S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred],
        t,
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

    # Compute data loss
    data_loss = (
        torch.mean((Is_pred - Is_data) ** 2)
        + torch.mean((H_pred - H_data) ** 2)
        + torch.mean((C_pred - C_data) ** 2)
        + torch.mean((D_pred - D_data) ** 2)
    )

    # Compute residual loss (physics-informed loss)
    residual_loss = (
        torch.mean((S_t - dSdt) ** 2)
        + torch.mean((E_t - dEdt) ** 2)
        + torch.mean((Is_t - dIsdt) ** 2)
        + torch.mean((Ia_t - dIadt) ** 2)
        + torch.mean((H_t - dHdt) ** 2)
        + torch.mean((C_t - dCdt) ** 2)
        + torch.mean((R_t - dRdt) ** 2)
        + torch.mean((D_t - dDdt) ** 2)
    )

    # Initial condition loss
    Is0, H0, C0, D0 = Is_data[0], H_data[0], C_data[0], D_data[0]
    initial_cost = (
        torch.mean((Is_pred[0] - Is0) ** 2)
        + torch.mean((H_pred[0] - H0) ** 2)
        + torch.mean((C_pred[0] - C0) ** 2)
        + torch.mean((D_pred[0] - D0) ** 2)
    )

    # Total loss
    loss = data_loss + residual_loss + initial_cost
    return loss

# ================================================================
# Early Stopping Class Definition
# ================================================================
class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
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
def train_model(
    model,
    parameter_net,
    optimizer,
    scheduler,
    time_stamps,
    train_data_scaled,
    num_epochs=50000,
    early_stopping=None,
):
    """Train the EpiNet model with tqdm for progress tracking."""
    train_losses = []

    time_stamps = time_stamps.to(device).float().requires_grad_()
    train_data_scaled = train_data_scaled.to(device).float()

    with tqdm(total=num_epochs) as pbar:  # Use tqdm to track progress across epochs
        for epoch in range(num_epochs):
            model.train()
            parameter_net.train()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            model_output = model(time_stamps)
            parameters = parameter_net(time_stamps)

            # Handle multi-GPU access to get_constants()
            if isinstance(model, nn.DataParallel):
                constants = model.module.get_constants()
            else:
                constants = model.get_constants()

            # Compute loss
            loss = pinn_loss(model_output, train_data_scaled, parameters, time_stamps, constants)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Append loss for tracking
            train_losses.append(loss.item())
            scheduler.step(loss.item())

            # Update tqdm progress bar
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
            pbar.update(1)

            # Check early stopping
            if early_stopping:
                early_stopping(loss.item())
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

    return train_losses


# ================================================================
# Model Initialization and Training
# ================================================================
# Initialize models
model = EpiNet(num_layers=5, hidden_neurons=20, output_size=8)
parameter_net = ParameterNet(num_layers=5, hidden_neurons=20, output_size=6)

# Move models to device
model = model.to(device)
parameter_net = parameter_net.to(device)

# Use DataParallel for multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    parameter_net = nn.DataParallel(parameter_net)

# Initialize optimizer and scheduler
optimizer = optim.AdamW(
    list(model.parameters()) + list(parameter_net.parameters()), lr=1e-4
)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.8)

# Early stopping
early_stopping = EarlyStopping(patience=100, verbose=True)

# Prepare tensors for the full dataset
time_stamps = torch.tensor(train_data.index.values, dtype=torch.float32).view(-1, 1)
train_data_scaled_tensor = torch.tensor(train_data_scaled, dtype=torch.float32)

# Train the model
train_losses = train_model(
    model,
    parameter_net,
    optimizer,
    scheduler,
    time_stamps,
    train_data_scaled_tensor,
    num_epochs=50000,
    early_stopping=early_stopping,
)

# ================================================================
# Plot Training Loss History
# ================================================================
plt.figure()
plt.plot(np.log10(train_losses), label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Log10(Loss)")
plt.title("Training Loss History")
plt.legend()
plt.show()

# ================================================================
# Model Evaluation
# ================================================================
# Prepare full time stamps
time_stamps_full = torch.tensor(data.index.values, dtype=torch.float32).view(-1, 1)
time_stamps_full = time_stamps_full.to(device).float().requires_grad_()

# Switch to evaluation mode
model.eval()
parameter_net.eval()

with torch.no_grad():
    model_output_full = model(time_stamps_full)
    parameters_full = parameter_net(time_stamps_full)

# Extract observed model outputs
observed_model_output = pd.DataFrame(
    {
        "daily_confirmed": model_output_full[:, 2].cpu().numpy(),
        "daily_hospitalized": model_output_full[:, 4].cpu().numpy(),
        "covidOccupiedMVBeds": model_output_full[:, 5].cpu().numpy(),
        "daily_deceased": model_output_full[:, 7].cpu().numpy(),
    },
    index=data.index,
)

# Inverse transform to original scale
observed_model_output_scaled = scaler.inverse_transform(observed_model_output)
observed_model_output_scaled = pd.DataFrame(
    observed_model_output_scaled,
    columns=features,
    index=data.index,
)

# Define unobserved model outputs
unobserved_model_output = pd.DataFrame(
    {
        "S": model_output_full[:, 0].cpu().numpy(),
        "E": model_output_full[:, 1].cpu().numpy(),
        "Ia": model_output_full[:, 3].cpu().numpy(),
        "R": model_output_full[:, 6].cpu().numpy(),
    },
    index=data.index,
)

# ================================================================
# Compute Error Metrics on Validation Data
# ================================================================
val_indices = val_data.index.values
val_observed = val_data[features].reset_index(drop=True)
val_predicted = observed_model_output_scaled.loc[val_indices].reset_index(drop=True)

print("\nValidation Metrics:")
for col in features:
    print(f"\nMetrics for {col}:")
    calculate_errors(
        val_observed[col].values,
        val_predicted[col].values,
        train_data[col].values,
    )

# ================================================================
# Plotting Observed vs Predicted Data
# ================================================================
def plot_observed_vs_predicted(ax, data, observed_model_output_scaled, variable, ylabel):
    ax.plot(data["date"], data[variable], label="Observed", color="blue")
    ax.plot(
        data["date"],
        observed_model_output_scaled[variable],
        label="Predicted",
        color="red",
        linestyle="--",
    )
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

plot_observed_vs_predicted(
    axs[0], data, observed_model_output_scaled, "daily_confirmed", r"$I_s$"
)
plot_observed_vs_predicted(
    axs[1], data, observed_model_output_scaled, "daily_hospitalized", r"$H$"
)
plot_observed_vs_predicted(
    axs[2], data, observed_model_output_scaled, "covidOccupiedMVBeds", r"$C$"
)
plot_observed_vs_predicted(
    axs[3], data, observed_model_output_scaled, "daily_deceased", r"$D$"
)

fig.suptitle("Observed vs Predicted Data", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ================================================================
# Plotting Unobserved Data Predictions
# ================================================================
def plot_unobserved_data(ax, unobserved_model_output, variable, ylabel):
    ax.plot(
        data["date"],
        unobserved_model_output[variable],
        label="Predicted",
        color="green",
    )
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

plot_unobserved_data(axs[0], unobserved_model_output, "S", r"$S$")
plot_unobserved_data(axs[1], unobserved_model_output, "E", r"$E$")
plot_unobserved_data(axs[2], unobserved_model_output, "Ia", r"$I_a$")
plot_unobserved_data(axs[3], unobserved_model_output, "R", r"$R$")

fig.suptitle("Unobserved Data Predictions", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ================================================================
# Time-Varying Parameter Estimation Visualization
# ================================================================
parameter_estimation = pd.DataFrame(
    {
        "beta": parameters_full[0].cpu().numpy(),
        "gamma_c": parameters_full[1].cpu().numpy(),
        "delta_c": parameters_full[2].cpu().numpy(),
        "eta": parameters_full[3].cpu().numpy(),
        "mu": parameters_full[4].cpu().numpy(),
        "omega": parameters_full[5].cpu().numpy(),
    },
    index=data.index,
)

# Apply smoothing to parameters for better visualization
parameter_estimation_smooth = parameter_estimation.rolling(window=7, min_periods=1).mean()

def plot_time_varying_parameters(ax, parameter_estimation, parameter, ylabel):
    ax.plot(data["date"], parameter_estimation[parameter], label="Estimated", color="purple")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

fig, axs = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

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
# Effective Reproduction Number R_t Calculation and Plotting
# ================================================================
# After obtaining parameters from the ParameterNet
beta_pred, gamma_c_pred, delta_c_pred, eta_pred, mu_pred, omega_pred = parameters_full

# Detach tensors
beta_pred = beta_pred.detach()
mu_pred = mu_pred.detach()

# Retrieve constant parameters and detach
rho = model.module.rho.detach() if isinstance(model, nn.DataParallel) else model.rho.detach()
alpha = model.module.alpha.detach() if isinstance(model, nn.DataParallel) else model.alpha.detach()
ds = model.module.ds.detach() if isinstance(model, nn.DataParallel) else model.ds.detach()
da = model.module.da.detach() if isinstance(model, nn.DataParallel) else model.da.detach()

# Calculate R_t
Rt = beta_pred * (rho / (ds + mu_pred) + (1 - rho) / da)

# Plot R_t
plt.figure(figsize=(10, 4))
plt.plot(data["date"], Rt.cpu().numpy(), label="$R_t$", color="orange")
plt.axhline(y=1, color='red', linestyle='--', label='Threshold (R=1)')
plt.ylabel("$R_t$")
plt.xlabel("Date")
plt.title("Effective Reproduction Number Over Time")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
