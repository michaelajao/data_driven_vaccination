# ================================================================
# Import Necessary Libraries
# ================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm.auto import tqdm
from scipy.integrate import odeint
import scienceplots
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Ensure the folders exist
os.makedirs("../../models", exist_ok=True)
os.makedirs("../../reports/figures/loss", exist_ok=True)
os.makedirs("../../reports/figures/Pinn", exist_ok=True)
os.makedirs("../../reports/results", exist_ok=True)
os.makedirs("../../reports/parameters", exist_ok=True)

# ================================================================
# Device Configuration
# ================================================================
# Device setup for CUDA or CPU
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    device = torch.device("cuda")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(seed)

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

# ================================================================
# Matplotlib Configuration
# ================================================================
# Set matplotlib style and parameters
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({
    "font.size": 14,
    "font.weight": "bold",
    "figure.figsize": [10, 6],
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
    "axes.grid": True,
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
})

# ================================================================
# Error Metrics Definitions
# ================================================================
def normalized_root_mean_square_error(y_true, y_pred):
    """Calculate the Normalized Root Mean Square Error (NRMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (np.max(y_true) - np.min(y_true))

def safe_mean_absolute_scaled_error(y_true, y_pred, y_train, epsilon=1e-10):
    """Calculate the Mean Absolute Scaled Error (MASE) safely."""
    n = len(y_train)
    d = np.abs(np.diff(y_train.squeeze())).sum() / (n - 1)
    d = max(d, epsilon)
    errors = np.abs(y_true.squeeze() - y_pred.squeeze())
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
    mape, nrmse, mase, rmse, mae, mse = calculate_errors(y_true, y_pred, y_train, train_size, area_name)
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

area_name = "East of England"

# ================================================================
# Data Loading and Preprocessing
# ================================================================
def load_preprocess_data(filepath, area_name, recovery_period=16, rolling_window=7, start_date="2020-04-01", end_date=None):
    """Load and preprocess the COVID-19 data."""
    df = pd.read_csv(filepath)
    
    # Select the area of interest
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

    # Calculate recovered cases
    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)
    
    # Calculate susceptible individuals
    df["susceptible"] = df["population"] - df["cumulative_confirmed"] - df["cumulative_deceased"] - df["recovered"]

    required_columns = [
        "date",
        "population",
        "susceptible",
        "cumulative_confirmed",
        "cumulative_deceased",
        "daily_confirmed",
        "daily_deceased",
        "daily_hospitalized",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "newAdmissions",
        "cumAdmissions",
        "recovered",
    ]

    # Select required columns
    df = df[required_columns]

    # Apply rolling average to smooth out data (except for date and population)
    for col in required_columns[2:]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1, center=False).mean().fillna(0)

    # Select data from start date to end date
    mask = (df["date"] >= start_date)
    if end_date:
        mask &= (df["date"] <= end_date)
    df = df.loc[mask].reset_index(drop=True)
    
    return df

data = load_preprocess_data("../../data/processed/merged_nhs_covid_data.csv", area_name, recovery_period=21, rolling_window=7, start_date="2020-05-01", end_date="2020-08-31")

# Plotting new deceased cases over time
plt.plot(data["date"], data["daily_deceased"])
plt.title("New Daily Deceased over time")
plt.xlabel("Date")
plt.ylabel("New Deceased")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ================================================================
# Data Preparation and Scaling
# ================================================================
def prepare_tensors(data, features, device):
    """Prepare tensors for training."""
    tensors = {}
    for feature in features:
        tensors[feature] = torch.tensor(data[feature].values, dtype=torch.float32).view(-1, 1).to(device)
    return tensors

def split_and_scale_data(data, train_size, features, device):
    """Split and scale data into training and validation sets."""
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    train_data = data.iloc[:train_size].reset_index(drop=True)
    val_data = data.iloc[train_size:].reset_index(drop=True)

    scaled_train_data = pd.DataFrame(scaler.transform(train_data[features]), columns=features)
    scaled_val_data = pd.DataFrame(scaler.transform(val_data[features]), columns=features)

    train_tensors = prepare_tensors(scaled_train_data, features, device)
    val_tensors = prepare_tensors(scaled_val_data, features, device)

    tensor_data = {
        "train": train_tensors,
        "val": val_tensors,
    }
    
    return tensor_data, scaler, train_data, val_data

features = ["daily_confirmed", "newAdmissions", "covidOccupiedMVBeds", "daily_deceased", "recovered"]

# Set the train size in days
train_size = 60  # Adjust as needed

tensor_data, scaler, train_data, val_data = split_and_scale_data(data, train_size, features, device)

# ================================================================
# Model Definition
# ================================================================
class EpiNet(nn.Module):
    def __init__(self, inverse=False, init_params=None, seed=42, num_layers=4, hidden_neurons=20):
        super(EpiNet, self).__init__()
        torch.manual_seed(seed)
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 8))  # Output: S, E, Is, Ia, H, C, R, D
        self.net = nn.Sequential(*layers)

        if inverse:
            # Initialize parameters
            self._beta = nn.Parameter(torch.tensor([init_params['beta']], dtype=torch.float32).to(device), requires_grad=True)
            self._omega = nn.Parameter(torch.tensor([init_params['omega']], dtype=torch.float32).to(device), requires_grad=True)
            self._mu = nn.Parameter(torch.tensor([init_params['mu']], dtype=torch.float32).to(device), requires_grad=True)
            self._gamma_c = nn.Parameter(torch.tensor([init_params['gamma_c']], dtype=torch.float32).to(device), requires_grad=True)
            self._delta_c = nn.Parameter(torch.tensor([init_params['delta_c']], dtype=torch.float32).to(device), requires_grad=True)
            self._eta = nn.Parameter(torch.tensor([init_params['eta']], dtype=torch.float32).to(device), requires_grad=True)
        else:
            self._beta = None
            self._omega = None
            self._mu = None
            self._gamma_c = None
            self._delta_c = None
            self._eta = None

        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    @property
    def beta(self):
        return torch.sigmoid(self._beta) * 0.9 + 0.1  # Range [0.1, 1.0]

    @property
    def omega(self):
        return torch.sigmoid(self._omega) * 0.5  # Range [0.0, 0.5]

    @property
    def mu(self):
        return torch.sigmoid(self._mu) * 0.1  # Range [0.0, 0.1]

    @property
    def gamma_c(self):
        return torch.sigmoid(self._gamma_c) * 0.5  # Range [0.0, 0.5]

    @property
    def delta_c(self):
        return torch.sigmoid(self._delta_c) * 0.5  # Range [0.0, 0.5]

    @property
    def eta(self):
        return torch.sigmoid(self._eta) * 0.2  # Range [0.0, 0.2]

    def init_xavier(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        self.apply(init_weights)

# ================================================================
# Loss Function Definition
# ================================================================
def einn_loss(model_output, tensor_data, model, t):
    """Compute the loss function for the EpiNet model."""
    # Split the model output into the different compartments
    S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred = torch.split(model_output, 1, dim=1)

    # Extract training data from tensor_data
    train_tensors = tensor_data["train"]
    val_tensors = tensor_data["val"]

    # Combine training and validation data for total data
    Is_total = torch.cat([train_tensors["daily_confirmed"], val_tensors["daily_confirmed"]], dim=0)
    H_total = torch.cat([train_tensors["newAdmissions"], val_tensors["newAdmissions"]], dim=0)
    C_total = torch.cat([train_tensors["covidOccupiedMVBeds"], val_tensors["covidOccupiedMVBeds"]], dim=0)
    D_total = torch.cat([train_tensors["daily_deceased"], val_tensors["daily_deceased"]], dim=0)
    R_total = torch.cat([train_tensors["recovered"], val_tensors["recovered"]], dim=0)

    # Since E, Ia are unobserved, we can use the model's predictions
    E_total = E_pred
    Ia_total = Ia_pred
    S_total = S_pred

    # Constants based on literature
    rho = 0.80  # Proportion of symptomatic infections
    alpha = 1 / 5  # Incubation period (5 days)
    ds = 1 / 4  # Infectious period for symptomatic (4 days)
    da = 1 / 7  # Infectious period for asymptomatic (7 days)
    dH = 1 / 13.4  # Hospitalization days (13.4 days)

    # Learned parameters
    if isinstance(model, nn.DataParallel):
        beta = model.module.beta
        omega = model.module.omega
        mu = model.module.mu
        gamma_c = model.module.gamma_c
        delta_c = model.module.delta_c
        eta = model.module.eta
    else:
        beta = model.beta
        omega = model.omega
        mu = model.mu
        gamma_c = model.gamma_c
        delta_c = model.delta_c
        eta = model.eta

    N = 1  # Normalized population

    # Compute time derivatives
    S_t = grad(S_pred.sum(), t, create_graph=True)[0]
    E_t = grad(E_pred.sum(), t, create_graph=True)[0]
    Ia_t = grad(Ia_pred.sum(), t, create_graph=True)[0]
    Is_t = grad(Is_pred.sum(), t, create_graph=True)[0]
    H_t = grad(H_pred.sum(), t, create_graph=True)[0]
    C_t = grad(C_pred.sum(), t, create_graph=True)[0]
    R_t = grad(R_pred.sum(), t, create_graph=True)[0]
    D_t = grad(D_pred.sum(), t, create_graph=True)[0]

    # Compute the differential equations
    dSdt = -beta * (Is_pred + Ia_pred) / N * S_pred + eta * R_pred
    dEdt = beta * (Is_pred + Ia_pred) / N * S_pred - alpha * E_pred
    dIsdt = alpha * rho * E_pred - ds * Is_pred
    dIadt = alpha * (1 - rho) * E_pred - da * Ia_pred
    dHdt = ds * omega * Is_pred - dH * H_pred - mu * H_pred
    dCdt = dH * (1 - omega) * H_pred - gamma_c * C_pred - delta_c * C_pred
    dRdt = ds * (1 - omega) * Is_pred + da * Ia_pred + dH * (1 - mu) * H_pred + gamma_c * C_pred - eta * R_pred
    dDdt = mu * H_pred + delta_c * C_pred

    # Data loss
    data_loss = (
        torch.mean((Is_pred - Is_total) ** 2) +
        torch.mean((H_pred - H_total) ** 2) +
        torch.mean((C_pred - C_total) ** 2) +
        torch.mean((D_pred - D_total) ** 2) +
        torch.mean((R_pred - R_total) ** 2)
    )

    # Residual loss
    residual_loss = (
        torch.mean((S_t - dSdt) ** 2) +
        torch.mean((E_t - dEdt) ** 2) +
        torch.mean((Ia_t - dIadt) ** 2) +
        torch.mean((Is_t - dIsdt) ** 2) +
        torch.mean((H_t - dHdt) ** 2) +
        torch.mean((C_t - dCdt) ** 2) +
        torch.mean((R_t - dRdt) ** 2) +
        torch.mean((D_t - dDdt) ** 2)
    )

    # Initial condition loss
    initial_loss = (
        (S_pred[0] - S_total[0]) ** 2 +
        (E_pred[0] - E_total[0]) ** 2 +
        (Ia_pred[0] - Ia_total[0]) ** 2 +
        (Is_pred[0] - Is_total[0]) ** 2 +
        (H_pred[0] - H_total[0]) ** 2 +
        (C_pred[0] - C_total[0]) ** 2 +
        (R_pred[0] - R_total[0]) ** 2 +
        (D_pred[0] - D_total[0]) ** 2
    )

    # Total loss
    loss = data_loss + residual_loss + initial_loss

    return loss

# ================================================================
# Early Stopping Definition
# ================================================================
class EarlyStopping:
    def __init__(self, patience=100, verbose=False, delta=0):
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
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# ================================================================
# Model Initialization
# ================================================================
# Initialize model parameters
init_params = {
    'beta': 0.5,
    'omega': 0.1,
    'mu': 0.01,
    'gamma_c': 0.1,
    'delta_c': 0.1,
    'eta': 0.05,
}

model = EpiNet(
    inverse=True,
    init_params=init_params,
    seed=seed,
    num_layers=6,
    hidden_neurons=32
).to(device)

# Wrap model with DataParallel if multiple GPUs are available
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print("Model is wrapped with DataParallel.")

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.9)
early_stopping = EarlyStopping(patience=100, verbose=True)

num_epochs = 10000  # Adjust as needed

# Create time tensor
t = torch.tensor(np.arange(0, len(data)), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)

# ================================================================
# Training Function
# ================================================================
def train_model(model, optimizer, scheduler, early_stopping, num_epochs, t, tensor_data):
    """Train the model."""
    loss_history = []
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        model_output = model(t)
        
        # Compute the loss function
        loss = einn_loss(model_output, tensor_data, model, t)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record the loss
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # Early stopping
        early_stopping(loss_value)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss_value:.6f}")
                
    return loss_history

# ================================================================
# Model Training
# ================================================================
# Train the model
loss_history = train_model(model, optimizer, scheduler, early_stopping, num_epochs, t, tensor_data)

# ================================================================
# Model Evaluation
# ================================================================
# Generate predictions
model.eval()
with torch.no_grad():
    model_output = model(t).cpu().numpy()

# Split the model output into the different compartments
S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred = np.split(model_output, 8, axis=1)

# Scale the predictions back to the original scale
scaled_predictions = np.concatenate([Is_pred, H_pred, C_pred, D_pred, R_pred], axis=1)
original_scale_predictions = scaler.inverse_transform(scaled_predictions)
Is_pred, H_pred, C_pred, D_pred, R_pred = np.split(original_scale_predictions, 5, axis=1)

# Actual values
actual_values = data[features].values
Is_actual, H_actual, C_actual, D_actual, R_actual = np.split(actual_values, 5, axis=1)

# Convert actual and predicted values to lists of arrays for easier plotting
actual_values_list = [Is_actual, H_actual, C_actual, D_actual, R_actual]
predicted_values_list = [Is_pred, H_pred, C_pred, D_pred, R_pred]

# ================================================================
# Plotting Functions
# ================================================================
def plot_loss(losses, title="Training Loss", filename=None):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(losses) + 1), losses, label='Loss', color='black')
    plt.yscale('log')
    plt.title(f"{title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, format='pdf', dpi=600)
    plt.show()

def plot_actual_vs_predicted(data, actual_values, predicted_values, features, train_size, area_name, filename=None):
    fig, ax = plt.subplots(len(features), 1, figsize=(12, 14), sharex=True)

    for i, feature in enumerate(features):
        ax[i].plot(data["date"], actual_values[i], label="Actual", color="blue")
        ax[i].plot(data["date"], predicted_values[i], label="Predicted", color="red", linestyle="--")
        ax[i].axvline(data["date"].iloc[train_size], color="green", linestyle="--", label="Train/Test Split")
        ax[i].set_ylabel(feature.replace("_", " ").title())
        ax[i].legend()
        ax[i].grid(True, linestyle='--', alpha=0.7)
        ax[i].tick_params(axis='x', rotation=45)

    ax[-1].set_xlabel("Date")
    plt.suptitle(f"EpiNet Model Predictions for {area_name}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if filename:
        plt.savefig(filename, format='pdf', dpi=600)
    plt.show()

# Plot loss history
plot_loss(loss_history, title="EpiNet Training Loss", filename=f"../../reports/figures/loss/{train_size}_{area_name}_loss_history.pdf")

# Plot actual vs predicted values
plot_actual_vs_predicted(data, actual_values_list, predicted_values_list, features, train_size, area_name, filename=f"../../reports/figures/Pinn/{train_size}_{area_name}_predictions.pdf")

# ================================================================
# Metrics Calculation
# ================================================================
# Calculate and print the metrics for each state
metrics = {}
metrics["daily_confirmed"] = calculate_all_metrics(Is_actual, Is_pred, tensor_data["train"]["daily_confirmed"].cpu().numpy(), "Daily Confirmed", train_size, area_name)
metrics["newAdmissions"] = calculate_all_metrics(H_actual, H_pred, tensor_data["train"]["newAdmissions"].cpu().numpy(), "New Admissions", train_size, area_name)
metrics["covidOccupiedMVBeds"] = calculate_all_metrics(C_actual, C_pred, tensor_data["train"]["covidOccupiedMVBeds"].cpu().numpy(), "Occupied MV Beds", train_size, area_name)
metrics["daily_deceased"] = calculate_all_metrics(D_actual, D_pred, tensor_data["train"]["daily_deceased"].cpu().numpy(), "Daily Deceased", train_size, area_name)
metrics["recovered"] = calculate_all_metrics(R_actual, R_pred, tensor_data["train"]["recovered"].cpu().numpy(), "Recovered", train_size, area_name)

# Save the metrics as CSV
metrics_df = pd.DataFrame(metrics, index=["MAPE", "NRMSE", "MASE", "RMSE", "MAE", "MSE"])
metrics_df.to_csv(f"../../reports/results/{train_size}_{area_name}_metrics.csv")

# ================================================================
# Extract and Save Learned Parameters
# ================================================================
# Extract the learned parameters
if isinstance(model, nn.DataParallel):
    beta = model.module.beta.cpu().item()
    omega = model.module.omega.cpu().item()
    mu = model.module.mu.cpu().item()
    gamma_c = model.module.gamma_c.cpu().item()
    delta_c = model.module.delta_c.cpu().item()
    eta = model.module.eta.cpu().item()
else:
    beta = model.beta.cpu().item()
    omega = model.omega.cpu().item()
    mu = model.mu.cpu().item()
    gamma_c = model.gamma_c.cpu().item()
    delta_c = model.delta_c.cpu().item()
    eta = model.eta.cpu().item()

# Print the learned parameters
print(f"\nLearned Parameters:")
print(f"Beta: {beta:.4f}")
print(f"Omega: {omega:.4f}")
print(f"Mu: {mu:.4f}")
print(f"Gamma_c: {gamma_c:.4f}")
print(f"Delta_c: {delta_c:.4f}")
print(f"Eta: {eta:.4f}")

# Save learned parameters as CSV
learned_params = pd.DataFrame({
    "beta": [beta],
    "omega": [omega],
    "mu": [mu],
    "gamma_c": [gamma_c],
    "delta_c": [delta_c],
    "eta": [eta]
})

learned_params.to_csv(f"../../reports/parameters/{train_size}_{area_name}_learned_params.csv", index=False)
