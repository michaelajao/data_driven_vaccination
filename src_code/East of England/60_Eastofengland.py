# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # For improved plotting aesthetics
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

# Ensure the folders exist
os.makedirs("../../models", exist_ok=True)
os.makedirs("../../reports/figures", exist_ok=True)
os.makedirs("../../reports/results", exist_ok=True)
os.makedirs("../../reports/parameters", exist_ok=True)

# Device setup for CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_pytorch():
    """Check PyTorch and CUDA setup."""
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available. PyTorch will run on CPU.")

check_pytorch()

# Set matplotlib style and parameters
plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams.update(
    {
        "font.size": 14,
        "font.weight": "bold",
        "figure.figsize": [8, 4],
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

# Define error metric functions
def normalized_root_mean_square_error(y_true, y_pred):
    """Calculate the Normalized Root Mean Square Error (NRMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (
        np.max(y_true) - np.min(y_true)
    )

def safe_mean_absolute_scaled_error(y_true, y_pred, y_train, epsilon=1e-10):
    """Calculate the Mean Absolute Scaled Error (MASE) safely."""
    n = len(y_train)
    d = np.abs(np.diff(y_train.squeeze())).sum() / (n - 1)
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

def calculate_all_metrics(y_true, y_pred, y_train, label, train_size, area_name):
    """Calculate metrics for each variable."""
    print(f"\nMetrics for {label}:")
    return calculate_errors(y_true, y_pred, y_train, train_size, area_name)

# Define the SEIRD model differential equations
def seird_model(y, t, N, beta, alpha, rho, ds, da, omega, dH, mu, gamma_c, delta_c, eta):
    """Define the SEIRD model differential equations."""
    S, E, Is, Ia, H, C, R, D = y  # Each is a tensor of shape (batch_size,)

    dSdt = -beta * (Is + Ia) / N * S + eta * R
    dEdt = beta * (Is + Ia) / N * S - alpha * E
    dIsdt = alpha * rho * E - ds * Is
    dIadt = alpha * (1 - rho) * E - da * Ia
    dHdt = ds * omega * Is - dH * H - mu * H
    dCdt = dH * (1 - omega) * H - gamma_c * C - delta_c * C
    dRdt = (
        ds * (1 - omega) * Is
        + da * Ia
        + dH * (1 - mu) * H
        + gamma_c * C
        - eta * R
    )
    dDdt = mu * H + delta_c * C

    return dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt

# def load_preprocess_data(
#     filepath,
#     area_name,
#     rolling_window=7,
#     start_date="2020-04-01",
#     end_date=None,
# ):
#     """Load and preprocess the COVID-19 data."""
#     df = pd.read_csv(filepath)

#     # Select the columns of interest
#     df = df[df["areaName"] == area_name].reset_index(drop=True)

#     # Convert the date column to datetime
#     df["date"] = pd.to_datetime(df["date"])

#     # Compute daily new values from cumulative values
#     df["daily_confirmed"] = df["cumulative_confirmed"].diff().fillna(0)
#     df["daily_deceased"] = df["cumulative_deceased"].diff().fillna(0)
#     df["daily_hospitalized"] = df["cumAdmissions"].diff().fillna(0)

#     # Ensure no negative values
#     df["daily_confirmed"] = df["daily_confirmed"].clip(lower=0)
#     df["daily_deceased"] = df["daily_deceased"].clip(lower=0)
#     df["daily_hospitalized"] = df["daily_hospitalized"].clip(lower=0)

#     required_columns = [
#         "date",
#         "population",
#         "cumulative_confirmed",
#         "cumulative_deceased",
#         "new_confirmed",
#         "new_deceased",
#         "cumAdmissions",
#         "daily_confirmed",
#         "daily_deceased",
#         "daily_hospitalized",
#         "hospitalCases",
#         "covidOccupiedMVBeds",
#         "newAdmissions",
#     ]

#     # Select required columns
#     df = df[required_columns]

#     # Apply rolling average to smooth out data (except for date and population)
#     for col in required_columns[2:]:
#         df[col] = (
#             df[col]
#             .rolling(window=rolling_window, min_periods=1, center=False)
#             .mean()
#             .fillna(0)
#         )

#     # Select data from start date to end date
#     mask = df["date"] >= start_date
#     if end_date:
#         mask &= df["date"] <= end_date
#     df = df.loc[mask].reset_index(drop=True)

#     return df

# Data Loading and Preprocessing
def load_preprocess_data(
    filepath,
    area_name,
    recovery_period=21,
    rolling_window=7,
    start_date="2020-05-01",
    end_date="2020-08-31",
):
    """Load and preprocess the COVID-19 data."""
    df = pd.read_csv(filepath)

    # Select the area of interest
    df = df[df["areaName"] == area_name].reset_index(drop=True)

    # Convert the date column to datetime
    df["date"] = pd.to_datetime(df["date"])
    
    df["daily_confirmed"] = df["cumulative_confirmed"].diff().fillna(0)
    df["daily_deceased"] = df["cumulative_deceased"].diff().fillna(0)

    # Calculate recovered cases
    df["recovered"] = (
        df["cumulative_confirmed"].shift(recovery_period)
        - df["cumulative_deceased"].shift(recovery_period)
    )
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)

    # Calculate susceptible cases
    df["susceptible"] = (
        df["population"]
        - df["cumulative_confirmed"]
        - df["cumulative_deceased"]
        - df["recovered"]
    )

    # Apply rolling average to smooth out data
    cols_to_smooth = [
        "susceptible",
        "cumulative_confirmed",
        "cumulative_deceased",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "new_deceased",
        "new_confirmed",
        "newAdmissions",
        "cumAdmissions",
        "recovered",
        "daily_confirmed",
        "daily_deceased",
    ]
    for col in cols_to_smooth:
        df[col] = (
            df[col]
            .rolling(window=rolling_window, min_periods=1)
            .mean()
            .fillna(0)
        )

    # Select data from start date to end date
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df.loc[mask].reset_index(drop=True)

    return df

# Set area name
area_name = "East of England"

# Load and preprocess data
data = load_preprocess_data(
    "../../data/processed/merged_nhs_covid_data.csv",
    area_name,
    rolling_window=7,
    start_date="2020-05-01",
    end_date="2020-08-31",
)

# Plot new deceased cases over time
plt.figure(figsize=(8, 4))
plt.plot(data["date"], data["daily_deceased"], label="daily_deceased")
plt.title("New Deceased over Time")
plt.xlabel("Date")
plt.ylabel("New Deceased")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Prepare tensors for training
def prepare_tensors(data, features, device):
    """Prepare tensors for training."""
    tensors = {}
    for feature in features:
        tensors[feature] = (
            torch.tensor(data[feature].values, dtype=torch.float32)
            .view(-1, 1)
            .to(device)
        )
    return tensors

def split_and_scale_data(data, train_size, features, device):
    """Split and scale data into training and validation sets."""
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    # Scale the data
    scaled_train_data = pd.DataFrame(
        scaler.transform(train_data[features]), columns=features
    )
    scaled_val_data = pd.DataFrame(
        scaler.transform(val_data[features]), columns=features
    )

    # Prepare tensors
    train_tensors = prepare_tensors(scaled_train_data, features, device)
    val_tensors = prepare_tensors(scaled_val_data, features, device)

    return train_tensors, val_tensors, scaler

# Define features
features = ["daily_confirmed", "newAdmissions", "covidOccupiedMVBeds", "daily_deceased", "recovered"]

# Set train size (number of days)
train_size = 60

# Split and scale data
train_tensors, val_tensors, scaler = split_and_scale_data(
    data, train_size, features, device
)

# Define the neural network model
class StateNN(nn.Module):
    """Neural network model for state approximation and parameter estimation."""

    def __init__(
        self,
        inverse=False,
        init_params=None,
        num_layers=4,
        hidden_neurons=20,
        seed=42,
    ):
        super(StateNN, self).__init__()
        torch.manual_seed(seed)

        # Define neural network layers
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 8))  # Output size is 8 (state variables)
        self.net = nn.Sequential(*layers)

        # Initialize parameters for inverse problem
        if inverse:
            # Initialize parameters with given initial values or defaults
            self.beta = nn.Parameter(
                torch.tensor(init_params.get("beta", 0.1), dtype=torch.float32).to(device)
            )
            self.omega = nn.Parameter(
                torch.tensor(init_params.get("omega", 0.01), dtype=torch.float32).to(device)
            )
            self.mu = nn.Parameter(
                torch.tensor(init_params.get("mu", 0.01), dtype=torch.float32).to(device)
            )
            self.gamma_c = nn.Parameter(
                torch.tensor(init_params.get("gamma_c", 0.01), dtype=torch.float32).to(device)
            )
            self.delta_c = nn.Parameter(
                torch.tensor(init_params.get("delta_c", 0.01), dtype=torch.float32).to(device)
            )
            self.eta = nn.Parameter(
                torch.tensor(init_params.get("eta", 0.01), dtype=torch.float32).to(device)
            )
        else:
            self.beta = None
            self.omega = None
            self.mu = None
            self.gamma_c = None
            self.delta_c = None
            self.eta = None

        # Initialize weights
        self.init_weights()

    def forward(self, t):
        return self.net(t)

    def get_parameters(self):
        """Get the estimated parameters with constraints."""
        beta = torch.sigmoid(self.beta) * 0.9 + 0.1
        omega = torch.sigmoid(self.omega) * 0.09 + 0.01
        mu = torch.sigmoid(self.mu) * 0.09 + 0.01
        gamma_c = torch.sigmoid(self.gamma_c) * 0.09 + 0.01
        delta_c = torch.sigmoid(self.delta_c) * 0.09 + 0.01
        eta = torch.sigmoid(self.eta) * 0.09 + 0.01
        return beta, omega, mu, gamma_c, delta_c, eta

    def init_weights(self):
        """Initialize neural network weights."""
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

        self.apply(init_layer)

# Define the loss function
def pinn_loss(
    model_output,
    data_tensors,
    model,
    t,
    constants,
    lambda_reg=1e-4,
):
    """Compute the PINN loss function."""
    # Split the model output into state variables
    S_pred, E_pred, Ia_pred, Is_pred, H_pred, C_pred, R_pred, D_pred = torch.split(
        model_output, 1, dim=1
    )

    # Squeeze the state variables to remove extra dimensions
    S_pred = S_pred.squeeze(-1)
    E_pred = E_pred.squeeze(-1)
    Ia_pred = Ia_pred.squeeze(-1)
    Is_pred = Is_pred.squeeze(-1)
    H_pred = H_pred.squeeze(-1)
    C_pred = C_pred.squeeze(-1)
    R_pred = R_pred.squeeze(-1)
    D_pred = D_pred.squeeze(-1)

    # Extract data tensors
    Is_data = torch.cat((data_tensors["train"]["daily_confirmed"], data_tensors["val"]["daily_confirmed"])).squeeze(-1)
    H_data = torch.cat((data_tensors["train"]["newAdmissions"], data_tensors["val"]["newAdmissions"])).squeeze(-1)
    C_data = torch.cat((data_tensors["train"]["covidOccupiedMVBeds"], data_tensors["val"]["covidOccupiedMVBeds"])).squeeze(-1)
    D_data = torch.cat((data_tensors["train"]["daily_deceased"], data_tensors["val"]["daily_deceased"])).squeeze(-1)
    R_data = torch.cat((data_tensors["train"]["recovered"], data_tensors["val"]["recovered"])).squeeze(-1)

    # Total population (normalized)
    N = 1.0

    # Estimated parameters
    beta, omega, mu, gamma_c, delta_c, eta = model.get_parameters()

    # Constants
    alpha, rho, ds, da, dH = constants

    # Compute time derivatives
    S_t = grad(
        S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True
    )[0].squeeze(-1)
    E_t = grad(
        E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True
    )[0].squeeze(-1)
    Ia_t = grad(
        Ia_pred, t, grad_outputs=torch.ones_like(Ia_pred), create_graph=True
    )[0].squeeze(-1)
    Is_t = grad(
        Is_pred, t, grad_outputs=torch.ones_like(Is_pred), create_graph=True
    )[0].squeeze(-1)
    H_t = grad(
        H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True
    )[0].squeeze(-1)
    C_t = grad(
        C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True
    )[0].squeeze(-1)
    R_t = grad(
        R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True
    )[0].squeeze(-1)
    D_t = grad(
        D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True
    )[0].squeeze(-1)

    # Compute the right-hand side of the differential equations
    dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt = seird_model(
        [S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred],
        t.squeeze(-1),
        N,
        beta,
        alpha,
        rho,
        ds,
        da,
        omega,
        dH,
        mu,
        gamma_c,
        delta_c,
        eta,
    )

    # Compute losses
    data_loss = (
        nn.MSELoss()(Is_pred, Is_data)
        + nn.MSELoss()(H_pred, H_data)
        + nn.MSELoss()(C_pred, C_data)
        + nn.MSELoss()(D_pred, D_data)
        + nn.MSELoss()(R_pred, R_data)
    )

    residual_loss = (
        nn.MSELoss()(S_t, dSdt)
        + nn.MSELoss()(E_t, dEdt)
        + nn.MSELoss()(Ia_t, dIadt)
        + nn.MSELoss()(Is_t, dIsdt)
        + nn.MSELoss()(H_t, dHdt)
        + nn.MSELoss()(C_t, dCdt)
        + nn.MSELoss()(R_t, dRdt)
        + nn.MSELoss()(D_t, dDdt)
    )

    # Initial condition loss
    initial_loss = (
        nn.MSELoss()(S_pred[0], S_pred[0].detach())
        + nn.MSELoss()(E_pred[0], E_pred[0].detach())
        + nn.MSELoss()(Ia_pred[0], Ia_pred[0].detach())
        + nn.MSELoss()(Is_pred[0], Is_pred[0].detach())
        + nn.MSELoss()(H_pred[0], H_pred[0].detach())
        + nn.MSELoss()(C_pred[0], C_pred[0].detach())
        + nn.MSELoss()(R_pred[0], R_pred[0].detach())
        + nn.MSELoss()(D_pred[0], D_pred[0].detach())
    )

    # L2 regularization
    l2_reg = torch.tensor(0.0).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)

    # Total loss
    total_loss = data_loss + residual_loss + initial_loss + lambda_reg * l2_reg

    return total_loss

# Early stopping class
class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=10, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                # Load the best model weights
                model.load_state_dict(self.best_model_state)
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

# Training function
def train_model(
    model,
    optimizer,
    scheduler,
    early_stopping,
    num_epochs,
    t,
    data_tensors,
    constants,
    lambda_reg=1e-4,
):
    """Train the PINN model."""
    loss_history = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        model_output = model(t)

        # Compute loss
        loss = pinn_loss(
            model_output, data_tensors, model, t, constants, lambda_reg
        )

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Record loss
        loss_history.append(loss.item())

        # Early stopping
        early_stopping(loss.item(), model)
        if early_stopping.early_stop:
            if early_stopping.verbose:
                print("Early stopping")
            break

        # Print progress
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

    return loss_history

# Plotting functions
def plot_loss(losses, title="Training Loss", filename=None):
    """Plot the training loss history."""
    plt.plot(np.arange(1, len(losses) + 1), losses, label="Loss", color="black")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, format="pdf", dpi=600)
    plt.show()

def plot_actual_vs_predicted(
    data,
    actual_values,
    predicted_values,
    features,
    train_size,
    area_name,
    filename=None,
):
    """Plot actual vs predicted values."""
    fig, axs = plt.subplots(len(features), 1, figsize=(10, 10), sharex=True)

    for i, feature in enumerate(features):
        axs[i].plot(
            data["date"],
            actual_values[i],
            label="Actual",
            color="black",
            marker="o",
            markersize=3,
            linestyle="None",
        )
        axs[i].plot(
            data["date"],
            predicted_values[i],
            label="Predicted",
            color="red",
            linestyle="--",
            linewidth=2,
        )
        axs[i].axvline(
            data["date"].iloc[train_size],
            color="blue",
            linestyle="--",
            label="Train/Test Split",
        )
        axs[i].set_ylabel(feature.replace("_", " ").title())
        axs[i].legend()
        # axs[i].grid(True, linestyle="--", alpha=0.7)

    axs[-1].set_xlabel("Date")
    plt.suptitle(f"EINN Model Predictions for {area_name}")
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if filename:
        plt.savefig(filename, format="pdf", dpi=600)
    plt.show()

# Define constants
alpha = 1 / 5  # Incubation period (5 days)
rho = 0.80  # Proportion of symptomatic infections
ds = 1 / 4  # Infectious period for symptomatic (4 days)
da = 1 / 7  # Infectious period for asymptomatic (7 days)
dH = 1 / 13.4  # Hospitalization days (13.4 days)
constants = (alpha, rho, ds, da, dH)

# Initialize the model
init_params = {
    "beta": 0.1,
    "omega": 0.01,
    "mu": 0.01,
    "gamma_c": 0.01,
    "delta_c": 0.01,
    "eta": 0.01,
}
model = StateNN(
    inverse=True,
    init_params=init_params,
    num_layers=8,
    hidden_neurons=20,
    seed=seed,
).to(device)

# Time tensor
t = (
    torch.tensor(np.arange(1, len(data) + 1), dtype=torch.float32)
    .view(-1, 1)
    .to(device)
    .requires_grad_(True)
)

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)

# Early stopping
early_stopping = EarlyStopping(patience=100, verbose=True)

# Number of epochs
num_epochs = 50000

# Prepare data tensors
data_tensors = {"train": train_tensors, "val": val_tensors}

# Train the model
loss_history = train_model(
    model,
    optimizer,
    scheduler,
    early_stopping,
    num_epochs,
    t,
    data_tensors,
    constants,
    lambda_reg=1e-4,
)

# Plot loss history
plot_loss(
    loss_history,
    title="EINN Training Loss",
    filename=f"../../reports/figures/loss/{train_size}_{area_name}_loss_history.pdf",
)

# Evaluate the model
model.eval()
with torch.no_grad():
    model_output = model(t).cpu().numpy()

# Split the model output into state variables
S_pred, E_pred, Ia_pred, Is_pred, H_pred, C_pred, R_pred, D_pred = np.split(
    model_output, 8, axis=1
)

# Scale predictions back to original scale
scaled_predictions = np.concatenate(
    [Is_pred, H_pred, C_pred, D_pred, R_pred], axis=1
)
original_scale_predictions = scaler.inverse_transform(scaled_predictions)
Is_pred, H_pred, C_pred, D_pred, R_pred = np.split(
    original_scale_predictions, 5, axis=1
)

# Actual values
actual_values = data[features].values
Is_actual, H_actual, C_actual, D_actual, R_actual = np.split(
    actual_values, 5, axis=1
)

# Convert to lists for plotting
actual_values_list = [Is_actual, H_actual, C_actual, D_actual, R_actual]
predicted_values_list = [Is_pred, H_pred, C_pred, D_pred, R_pred]

# Plot actual vs predicted values
plot_actual_vs_predicted(
    data,
    actual_values_list,
    predicted_values_list,
    features,
    train_size,
    area_name,
    filename=f"../../reports/figures/Pinn/{train_size}_{area_name}_predictions.pdf",
)

# Calculate and print metrics for each feature
metrics = {}
for actual, pred, feature in zip(
    actual_values_list, predicted_values_list, features
):
    metrics[feature] = calculate_all_metrics(
        actual, pred, train_tensors[feature].cpu().numpy(), feature, train_size, area_name
    )

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(
    f"../../reports/results/{train_size}_{area_name}_metrics.csv", index=False
)

# Extract learned parameters
beta, omega, mu, gamma_c, delta_c, eta = model.get_parameters()
beta, omega, mu, gamma_c, delta_c, eta = (
    beta.item(),
    omega.item(),
    mu.item(),
    gamma_c.item(),
    delta_c.item(),
    eta.item(),
)

# Print learned parameters
print("\nLearned Parameters:")
print(f"Beta: {beta:.4f}")
print(f"Omega: {omega:.4f}")
print(f"Mu: {mu:.4f}")
print(f"Gamma_c: {gamma_c:.4f}")
print(f"Delta_c: {delta_c:.4f}")
print(f"Eta: {eta:.4f}")

# Save learned parameters to CSV
learned_params = pd.DataFrame(
    {
        "beta": [beta],
        "omega": [omega],
        "mu": [mu],
        "gamma_c": [gamma_c],
        "delta_c": [delta_c],
        "eta": [eta],
    }
)
learned_params.to_csv(
    f"../../reports/parameters/{train_size}_{area_name}_learned_params.csv",
    index=False,
)
