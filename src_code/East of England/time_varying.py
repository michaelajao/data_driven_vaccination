import os
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from collections import deque
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

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

# Set matplotlib style and parameters
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({
    "font.size": 14,
    "figure.figsize": [10, 5],
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
})

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

def calculate_errors(y_true, y_pred, y_train, train_size, areaname):
    """Calculate and print various error metrics."""
    mape = mean_absolute_percentage_error(y_true, y_pred)
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

def calculate_all_metrics(actual, predicted, train_data, label, train_size, areaname):
    """Calculate metrics for each state."""
    print(f"\nMetrics for {label}:")
    mape, nrmse, mase, rmse, mae, mse = calculate_errors(actual, predicted, train_data, train_size, areaname)
    return mape, nrmse, mase, rmse, mae, mse

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

areaname = "East of England"
def load_preprocess_data(filepath, areaname, recovery_period=16, rolling_window=7, start_date=None, end_date=None):
    """Load and preprocess the COVID-19 data."""
    df = pd.read_csv(filepath)
    
    # Select the columns of interest
    df = df[df["areaName"] == areaname].reset_index(drop=True)
    
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
        "date",
        "population",
        "cumulative_confirmed",
        "cumulative_deceased",
        "new_confirmed",
        "new_deceased",
        "cumAdmissions",
        "daily_confirmed",
        "daily_deceased",
        "daily_hospitalized",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "newAdmissions",
    ]

    # Select required columns
    df = df[required_columns]

    # Apply 7-day rolling average to smooth out data (except for date and population)
    for col in required_columns[2:]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    # Select data from start date to end date
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df.loc[mask]
    
    return df

data = load_preprocess_data("../../data/processed/merged_nhs_covid_data.csv", areaname, recovery_period=21, rolling_window=7,start_date="2020-05-01", end_date="2020-12-31")

# Plotting new deceased cases over time
plt.plot(data["date"], data["daily_deceased"])
plt.title("New Daily Deceased over time")
plt.xlabel("Date")
plt.ylabel("New Deceased")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Split data into training and validation sets, the validation should be the last 7 days
train_data = data[:-7]
val_data = data[-7:]

def prepare_tensors(data, device):
    """Prepare tensors for training."""
    I = (
        torch.tensor(data["daily_confirmed"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    H = (
        torch.tensor(data["daily_hospitalized"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    C = (
        torch.tensor(data["covidOccupiedMVBeds"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    D = (
        torch.tensor(data["daily_deceased"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    return I, H, C, D

def scale_data(data, features, device):
    """Split and scale data into training and validation sets."""
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    # Scale the data
    data_scaled = scaler.transform(data[features])

    # Convert the data to PyTorch tensors
    data_scaled = torch.tensor(data_scaled, dtype=torch.float32).to(device)

    return data_scaled, scaler

features = [
    "daily_confirmed",
    "daily_hospitalized",
    "covidOccupiedMVBeds",
    "daily_deceased",
]

# Split and scale the data
data_scaled, scaler = scale_data(train_data, features, device)

class ResidualBlock(nn.Module):
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
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=8):
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
        
        self._rho = nn.Parameter(torch.tensor([torch.rand(1)], device=device), requires_grad=True)
        self._alpha = nn.Parameter(torch.tensor([torch.rand(1)], device=device), requires_grad=True)
        self._ds = nn.Parameter(torch.tensor([torch.rand(1)], device=device), requires_grad=True)
        self._da = nn.Parameter(torch.tensor([torch.rand(1)], device=device), requires_grad=True)
        self._dH = nn.Parameter(torch.tensor([torch.rand(1)], device=device), requires_grad=True)

        # Initialize weights using Xavier initialization
        self.init_xavier()

    def forward(self, t):
        return torch.sigmoid(self.net(t))
    
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
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Apply the weight initialization to the network
        self.net.apply(init_weights)

class ParameterNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=6):
        super(ParameterNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        
        #         # Hidden layers with residual connections
        # for _ in range(num_layers):
        #     layers.append(ResidualBlock(hidden_neurons))

        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)

        self.init_xavier()

    def forward(self, t):
        raw_parameters = self.net(t)

        # Apply the sigmoid function to represent constant parameters
        beta = torch.sigmoid(raw_parameters[:, 0]) 
        gamma_c = torch.sigmoid(raw_parameters[:, 1])
        delta_c = torch.sigmoid(raw_parameters[:, 2])
        eta = torch.sigmoid(raw_parameters[:, 3])
        mu = torch.sigmoid(raw_parameters[:, 4])
        omega = torch.sigmoid(raw_parameters[:, 5])

        return beta, gamma_c, delta_c, eta, mu, omega

    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Apply the weight initialization to the network
        self.net.apply(init_weights)

def einn_loss(model_output, tensor_data, parameters, t, constants):
    """Calculate the EpiNet loss."""
    S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred = (
        model_output[:, 0], model_output[:, 1], model_output[:, 2], model_output[:, 3],
        model_output[:, 4], model_output[:, 5], model_output[:, 6], model_output[:, 7]
    )

    Is_data, H_data, C_data, D_data = tensor_data[:, 0], tensor_data[:, 1], tensor_data[:, 2], tensor_data[:, 3]

    N = 1
    rho, alpha, ds, da, dH = constants
    
    beta_pred, gamma_c_pred, delta_c_pred, eta_pred, mu_pred, omega_pred = parameters

    S_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    E_t = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]
    Ia_t = grad(Ia_pred, t, grad_outputs=torch.ones_like(Ia_pred), create_graph=True)[0]
    Is_t = grad(Is_pred, t, grad_outputs=torch.ones_like(Is_pred), create_graph=True)[0]
    H_t = grad(H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    C_t = grad(C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
    R_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    D_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]

    dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt = seird_model(
        [S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred],
        t, N, beta_pred, alpha, rho, ds, da, omega_pred, dH, mu_pred, gamma_c_pred, delta_c_pred, eta_pred
    )

    data_loss = torch.mean((Is_pred - Is_data) ** 2) + torch.mean((H_pred - H_data) ** 2) + torch.mean((C_pred - C_data) ** 2) + torch.mean((D_pred - D_data) ** 2)
    
    residual_loss = torch.mean((S_t - dSdt) ** 2) + torch.mean((E_t - dEdt) ** 2) + torch.mean((Is_t - dIsdt) ** 2) + torch.mean((Ia_t - dIadt) ** 2) + torch.mean((H_t - dHdt) ** 2) + torch.mean((C_t - dCdt) ** 2) + torch.mean((R_t - dRdt) ** 2) + torch.mean((D_t - dDdt) ** 2)
    
    # Initial condition loss
    Is0, H0, C0, D0 = Is_data[0], H_data[0], C_data[0], D_data[0]
    initial_cost = torch.mean((Is_pred[0] - Is0) ** 2) + torch.mean((H_pred[0] - H0) ** 2) + torch.mean((C_pred[0] - C0) ** 2) + torch.mean((D_pred[0] - D0) ** 2)
    
    loss = data_loss + residual_loss + initial_cost
    return loss

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
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

def train_model(model, parameter_net, optimizer, scheduler, time_stamps, data_scaled, num_epochs=100, early_stopping=None):
    """Train the EpiNet model."""
    train_losses = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        parameter_net.train()

        train_loss = 0.0

        t = time_stamps.to(device).float()
        data = data_scaled.to(device).float()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        model_output = model(t)
        parameters = parameter_net(t)
        constants = model.get_constants()

        # Compute loss
        loss = einn_loss(model_output, data, parameters, t, constants)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        train_losses.append(train_loss)
        scheduler.step(train_loss)

        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}")

        # Check early stopping
        if early_stopping:
            early_stopping(train_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return train_losses

# Initialize model, optimizer, and scheduler
model = EpiNet(num_layers=5, hidden_neurons=32, output_size=8).to(device)
parameter_net = ParameterNet(num_layers=3, hidden_neurons=32, output_size=6).to(device)
optimizer = optim.Adam(
    list(model.parameters()) + list(parameter_net.parameters()), lr=1e-4
)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.8)

# Early stopping
early_stopping = EarlyStopping(patience=100, verbose=False)

# Create timestamps tensor
time_stamps = (
    torch.tensor(train_data.index.values, dtype=torch.float32)
    .view(-1, 1)
    .to(device)
    .requires_grad_()
)

# Train the model
train_losses = train_model(
    model,
    parameter_net,
    optimizer,
    scheduler,
    time_stamps,
    data_scaled,
    num_epochs=50000,
    early_stopping=early_stopping,
)

# Plot the loss history
plt.figure()
plt.plot(np.log10(train_losses), label="Training Loss", color="blue")
plt.xlabel("Epoch")
plt.ylabel("Log10(Loss)")
plt.title("Training Loss History")
plt.legend()
plt.show()

# Test the model on the validation set
model.eval()
parameter_net.eval()

with torch.no_grad():
    time_stamps = torch.tensor(data.index.values, dtype=torch.float32).view(-1, 1).to(device).requires_grad_()
    model_output = model(time_stamps)
    parameters = parameter_net(time_stamps)

# Convert model output to DataFrame and inverse transform to original scale
observed_model_output = pd.DataFrame(
    {
        "daily_confirmed": model_output[:, 2].cpu().numpy(),
        "daily_hospitalized": model_output[:, 4].cpu().numpy(),
        "covidOccupiedMVBeds": model_output[:, 5].cpu().numpy(),
        "daily_deceased": model_output[:, 7].cpu().numpy(),
    },
    index=data.index,
)

observed_model_output_scaled = scaler.inverse_transform(observed_model_output)
observed_model_output_scaled = pd.DataFrame(
    observed_model_output_scaled,
    columns=["daily_confirmed", "daily_hospitalized", "covidOccupiedMVBeds", "daily_deceased"],
    index=data.index
)

# Plot Observed Data vs Predicted Data
def plot_observed_vs_predicted(data, observed_model_output_scaled, variable, title, ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(data["date"], data[variable], label="Observed", color="blue")
    plt.plot(data["date"], observed_model_output_scaled[variable], label="Predicted", color="red", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_observed_vs_predicted(data, observed_model_output_scaled, "daily_confirmed", "Daily Confirmed Cases: Observed vs Predicted", "Daily Confirmed")
plot_observed_vs_predicted(data, observed_model_output_scaled, "daily_hospitalized", "Daily Hospitalized Cases: Observed vs Predicted", "Daily Hospitalized")
plot_observed_vs_predicted(data, observed_model_output_scaled, "covidOccupiedMVBeds", "COVID Occupied MV Beds: Observed vs Predicted", "COVID Occupied MV Beds")
plot_observed_vs_predicted(data, observed_model_output_scaled, "daily_deceased", "Daily Deceased Cases: Observed vs Predicted", "Daily Deceased")

# Unobserved Data Visualization
unobserved_model_output = pd.DataFrame(
    {
        "S": model_output[:, 0].cpu().numpy(),
        "E": model_output[:, 1].cpu().numpy(),
        "Ia": model_output[:, 3].cpu().numpy(),
        "R": model_output[:, 6].cpu().numpy(),
    },
    index=data.index,
)

def plot_unobserved_data(unobserved_model_output, variable, title, ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(data["date"], unobserved_model_output[variable], label="Predicted", color="green")
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_unobserved_data(unobserved_model_output, "S", "Susceptible Population Over Time", "Susceptible")
plot_unobserved_data(unobserved_model_output, "E", "Exposed Population Over Time", "Exposed")
plot_unobserved_data(unobserved_model_output, "Ia", "Asymptomatic Infected Population Over Time", "Asymptomatic Infected")
plot_unobserved_data(unobserved_model_output, "R", "Recovered Population Over Time", "Recovered")

# Time-Varying Parameter Estimation Visualization
parameter_estimation = pd.DataFrame(
    {
        "beta": parameters[0].cpu().numpy(),
        "gamma_c": parameters[1].cpu().numpy(),
        "delta_c": parameters[2].cpu().numpy(),
        "eta": parameters[3].cpu().numpy(),
        "mu": parameters[4].cpu().numpy(),
        "omega": parameters[5].cpu().numpy(),
    },
    index=data.index,
)

def plot_time_varying_parameters(parameter_estimation, parameter, title, ylabel):
    plt.figure(figsize=(12, 6))
    plt.plot(data["date"], parameter_estimation[parameter], label="Estimated", color="purple")
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_time_varying_parameters(parameter_estimation, "beta", "Time-Varying Beta", "Beta")
plot_time_varying_parameters(parameter_estimation, "gamma_c", "Time-Varying Gamma_c", "Gamma_c")
plot_time_varying_parameters(parameter_estimation, "delta_c", "Time-Varying Delta_c", "Delta_c")
plot_time_varying_parameters(parameter_estimation, "eta", "Time-Varying Eta", "Eta")
plot_time_varying_parameters(parameter_estimation, "mu", "Time-Varying Mu", "Mu")
plot_time_varying_parameters(parameter_estimation, "omega", "Time-Varying Omega", "Omega")
