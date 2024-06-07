import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm.notebook import tqdm
from scipy.integrate import odeint
from collections import deque
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

# Ensure necessary directories exist
os.makedirs("../../models", exist_ok=True)
os.makedirs("../../reports/figures", exist_ok=True)
os.makedirs("../../reports/results", exist_ok=True)
os.makedirs("../../reports/England", exist_ok=True)

# Device setup for CUDA or CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
plt.rcParams.update(
    {
        "font.size": 14,
        "figure.figsize": [10, 6],
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.titlesize": 18,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 12,
        "legend.frameon": False,
        "legend.loc": "best",
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "grid.color": "grey",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "errorbar.capsize": 4,
        "figure.subplot.wspace": 0.4,
        "figure.subplot.hspace": 0.4,
        "image.cmap": "viridis",
    }
)


def normalized_root_mean_square_error(y_true, y_pred):
    """Calculate the Normalized Root Mean Square Error (NRMSE)."""
    return np.sqrt(mean_squared_error(y_true, y_pred)) / (
        np.max(y_true) - np.min(y_true)
    )


def safe_mean_absolute_scaled_error(y_true, y_pred, y_train, epsilon=1e-10):
    """Calculate the Mean Absolute Scaled Error (MASE) safely."""
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    d = max(d, epsilon)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d


def calculate_errors(y_true, y_pred, y_train, areaname):
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


def calculate_all_metrics(actual, predicted, train_data, label, areaname):
    """Calculate metrics for each state."""
    print(f"\nMetrics for {label}:")
    mape, nrmse, mase, rmse, mae, mse = calculate_errors(
        actual, predicted, train_data, areaname
    )
    return mape, nrmse, mase, rmse, mae, mse


# Define the SEIRD model differential equations
def seird_model(
    y, t, N, beta, alpha, rho, ds, da, omega, dH, mu, gamma_c, delta_c, eta
):
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


def load_preprocess_data(
    filepath, areaname, recovery_period=16, rolling_window=7, end_date=None
):
    """Load and preprocess the COVID-19 data."""
    df = pd.read_csv(filepath)

    # Convert the date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # Select data up to the end_date
    if end_date:
        df = df[df["date"] <= end_date]

    cols_to_smooth = [
        "cumulative_confirmed",
        "cumulative_deceased",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "new_deceased",
        "new_confirmed",
        "newAdmissions",
    ]
    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    return df


data = load_preprocess_data(
    "../../data/processed/england_data.csv",
    "England",
    recovery_period=21,
    rolling_window=7,
    end_date="2021-12-31",
)

plt.plot(data["date"], data["new_deceased"])
plt.title("New Deceased over time")
plt.xlabel("Date")
plt.ylabel("New Deceased")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


def prepare_tensors(data, device):
    """Prepare tensors for training."""
    I = torch.tensor(data["new_confirmed"].values, dtype=torch.float32).view(-1, 1).to(device)
    H = torch.tensor(data["newAdmissions"].values, dtype=torch.float32).view(-1, 1).to(device)
    C = torch.tensor(data["covidOccupiedMVBeds"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = torch.tensor(data["new_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
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
    "new_confirmed",
    "newAdmissions",
    "covidOccupiedMVBeds",
    "new_deceased",
]

# Split and scale the data
data_scaled, scaler = scale_data(data, features, device)


class EpiNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=8):
        super(EpiNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])

        # Output layer
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self.init_xavier()

    def forward(self, t):
        return torch.sigmoid(self.net(t))

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
    def __init__(self, num_layers=2, hidden_neurons=10):
        super(ParameterNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])

        layers.append(nn.Linear(hidden_neurons, 6))
        self.net = nn.Sequential(*layers)

        self.init_xavier()

    def forward(self, t):
        return self.net(t)
    
    def get_parameters(self, t):
        raw_parameters = self.net(t)
        
        # Apply the sigmoid function to ensure the parameters are in the correct range
        beta = torch.sigmoid(raw_parameters[:, 0])
        omega = torch.sigmoid(raw_parameters[:, 1])
        mu = torch.sigmoid(raw_parameters[:, 2])
        gamma_c = torch.sigmoid(raw_parameters[:, 3])
        delta_c = torch.sigmoid(raw_parameters[:, 4])
        eta = torch.sigmoid(raw_parameters[:, 5])

        return beta, omega, mu, gamma_c, delta_c, eta
    
    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Apply the weight initialization to the network
        self.net.apply(init_weights)
        

def einn_loss(model_output, tensor_data, parameters, t):
    """Compute the loss function for the EINN model with L2 regularization."""

    # Split the model output into the different compartments
    S_pred, E_pred, Ia_pred, Is_pred, H_pred, C_pred, D_pred, R_pred = (
        model_output[:, 0],
        model_output[:, 1],
        model_output[:, 2],
        model_output[:, 3],
        model_output[:, 4],
        model_output[:, 5],
        model_output[:, 6],
        model_output[:, 7],
    )

    # Normalize the data
    N = 1

    Is_data, H_data, C_data, D_data = (
        tensor_data[:, 0],
        tensor_data[:, 1],
        tensor_data[:, 2],
        tensor_data[:, 3],
    )

    # Constants based on the table provided
    rho = 0.80  # Proportion of symptomatic infections
    alpha = 1 / 5  # Incubation period (5 days)
    ds = 1 / 4  # Infectious period for symptomatic (4 days)
    da = 1 / 7  # Infectious period for asymptomatic (7 days)
    dH = 1 / 13.4  # Hospitalization days (13.4 days)

    # Learned parameters
    beta_pred, omega_pred, mu_pred, gamma_c_pred, delta_c_pred, eta_pred = parameters

    # Compute the differential equations
    S_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True, retain_graph=True)[0]
    E_t = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True, retain_graph=True)[0]
    Ia_t = grad(Ia_pred, t, grad_outputs=torch.ones_like(Ia_pred), create_graph=True, retain_graph=True)[0]
    Is_t = grad(Is_pred, t, grad_outputs=torch.ones_like(Is_pred), create_graph=True, retain_graph=True)[0]
    H_t = grad(H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True, retain_graph=True)[0]
    C_t = grad(C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True, retain_graph=True)[0]
    R_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True, retain_graph=True)[0]
    D_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True, retain_graph=True)[0]
    
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

    # Compute the loss function
    data_loss = (
        torch.mean((Is_pred - Is_data) ** 2)
        + torch.mean((H_pred - H_data) ** 2)
        + torch.mean((C_pred - C_data) ** 2)
        + torch.mean((D_pred - D_data) ** 2)
    )

    residual_loss = (
        torch.mean((S_t - dSdt) ** 2)
        + torch.mean((E_t - dEdt) ** 2)
        + torch.mean((Ia_t - dIadt) ** 2)
        + torch.mean((Is_t - dIsdt) ** 2)
        + torch.mean((H_t - dHdt) ** 2)
        + torch.mean((C_t - dCdt) ** 2)
        + torch.mean((R_t - dRdt) ** 2)
        + torch.mean((D_t - dDdt) ** 2)
    )

    loss = data_loss + residual_loss
    return loss


# Training loop
def train_model(model, parameter_net, optimizer, scheduler, train_loader, num_epochs=100):
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        parameter_net.train()
        
        train_loss = 0.0
        for t, data in train_loader:
            t = t.to(device).float()
            data = data.to(device).float()

            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            model_output = model(t)
            parameters = parameter_net.get_parameters(t)
            
            # Compute loss
            loss = einn_loss(model_output, data, parameters, t)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * t.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        scheduler.step(train_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}")

    return train_losses


# Prepare data for training
data = load_preprocess_data(
    "../../data/processed/england_data.csv",
    "England",
    recovery_period=21,
    rolling_window=7,
    end_date="2021-12-31",
)

scaled_data, scaler = scale_data(data, features, device)

# Create TensorDataset and DataLoader
time_stamps = torch.tensor(data.index.values, dtype=torch.float32, requires_grad=True).view(-1, 1)
tensor_dataset = torch.utils.data.TensorDataset(time_stamps, scaled_data)
train_size = len(tensor_dataset)
train_dataset = tensor_dataset

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and scheduler
model = EpiNet().to(device)
parameter_net = ParameterNet().to(device)
optimizer = optim.Adam(list(model.parameters()) + list(parameter_net.parameters()), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Train the model
train_losses = train_model(model, parameter_net, optimizer, scheduler, train_loader, num_epochs=100)

# Plot training loss
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('../../reports/figures/training_loss.pdf')
plt.show()

# Save the trained model
torch.save(model.state_dict(), "../../models/epinet_model.pth")
torch.save(parameter_net.state_dict(), "../../models/parameter_net.pth")


# Plot model outputs and parameters
def plot_outputs(model, parameter_net, data, device):
    model.eval()
    parameter_net.eval()

    with torch.no_grad():
        time_stamps = torch.tensor(data.index.values, dtype=torch.float32).view(-1, 1).to(device)
        model_output = model(time_stamps).cpu().numpy()
        parameters = parameter_net.get_parameters(time_stamps).cpu().numpy()

    dates = data["date"]

    # Plot observed vs. predicted outputs
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    axs[0].plot(dates, data["new_confirmed"], label="Observed Infections")
    axs[0].plot(dates, model_output[:, 3], label="Predicted Infections")
    axs[0].set_ylabel("New Confirmed Cases")
    axs[0].legend()

    axs[1].plot(dates, data["newAdmissions"], label="Observed Hospitalizations")
    axs[1].plot(dates, model_output[:, 4], label="Predicted Hospitalizations")
    axs[1].set_ylabel("New Admissions")
    axs[1].legend()

    axs[2].plot(dates, data["covidOccupiedMVBeds"], label="Observed Critical")
    axs[2].plot(dates, model_output[:, 5], label="Predicted Critical")
    axs[2].set_ylabel("Critical Cases")
    axs[2].legend()

    axs[3].plot(dates, data["new_deceased"], label="Observed Deaths")
    axs[3].plot(dates, model_output[:, 6], label="Predicted Deaths")
    axs[3].set_ylabel("New Deaths")
    axs[3].legend()

    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../../reports/figures/observed_vs_predicted.pdf')
    plt.show()

    # Plot unobserved outputs
    fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

    axs[0].plot(dates, model_output[:, 0], label="Susceptible")
    axs[0].set_ylabel("Susceptible")
    axs[0].legend()

    axs[1].plot(dates, model_output[:, 1], label="Exposed")
    axs[1].set_ylabel("Exposed")
    axs[1].legend()

    axs[2].plot(dates, model_output[:, 2], label="Asymptomatic Infected")
    axs[2].set_ylabel("Asymptomatic Infected")
    axs[2].legend()

    axs[3].plot(dates, model_output[:, 7], label="Recovered")
    axs[3].set_ylabel("Recovered")
    axs[3].legend()

    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../../reports/figures/unobserved_outputs.pdf')
    plt.show()

    # Plot time-varying parameters
    fig, axs = plt.subplots(3, 2, figsize=(12, 16), sharex=True)

    axs[0, 0].plot(dates, parameters[:, 0], label="Beta (Infection rate)")
    axs[0, 0].set_ylabel("Beta")
    axs[0, 0].legend()

    axs[0, 1].plot(dates, parameters[:, 1], label="Omega (Symptomatic proportion)")
    axs[0, 1].set_ylabel("Omega")
    axs[0, 1].legend()

    axs[1, 0].plot(dates, parameters[:, 2], label="Mu (Mortality rate)")
    axs[1, 0].set_ylabel("Mu")
    axs[1, 0].legend()

    axs[1, 1].plot(dates, parameters[:, 3], label="Gamma_c (Critical recovery rate)")
    axs[1, 1].set_ylabel("Gamma_c")
    axs[1, 1].legend()

    axs[2, 0].plot(dates, parameters[:, 4], label="Delta_c (Critical mortality rate)")
    axs[2, 0].set_ylabel("Delta_c")
    axs[2, 0].legend()

    axs[2, 1].plot(dates, parameters[:, 5], label="Eta (Reinfection rate)")
    axs[2, 1].set_ylabel("Eta")
    axs[2, 1].legend()

    plt.xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('../../reports/figures/time_varying_parameters.pdf')
    plt.show()

# Plot the outputs and parameters
plot_outputs(model, parameter_net, data, device)
