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
from torch import tensor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

# Ensure the folders exist
os.makedirs("../../models", exist_ok=True)
os.makedirs("../../reports/figures", exist_ok=True)
os.makedirs("../../reports/results", exist_ok=True)
os.makedirs("../../reports/England", exist_ok=True)

# Set CUDA_LAUNCH_BLOCKING for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
plt.rcParams.update(
    {
        "font.size": 14,
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
    mape, nrmse, mase, rmse, mae, mse = calculate_errors(
        actual, predicted, train_data, train_size, areaname
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


areaname = "England"


def load_preprocess_data(
    filepath, areaname, recovery_period=16, rolling_window=7, end_date=None
):
    """Load and preprocess the COVID-19 data."""
    df = pd.read_csv(filepath)

    # Convert the date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # select data up to the end_date
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
    areaname,
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
# plt.savefig("../../reports/figures/S(t)_over_time.pdf")
plt.show()


def prepare_tensors(data, device):
    """Prepare tensors for training."""
    I = tensor(data["new_confirmed"].values, dtype=torch.float32).view(-1, 1).to(device)
    H = tensor(data["newAdmissions"].values, dtype=torch.float32).view(-1, 1).to(device)
    C = (
        tensor(data["covidOccupiedMVBeds"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    D = tensor(data["new_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
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
        torch.manual_seed(self.retain_seed)  # Ensure reproducibility

        layers = [nn.Linear(1, hidden_neurons), nn.ReLU()]  # Input layer

        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.ReLU()])  # Hidden layers

        layers.append(nn.Linear(hidden_neurons, 1))  # Output layer
        self.net = nn.Sequential(*layers)

        self.init_xavier()  # Initialize the weights using Xavier initialization

    def forward(self, t):
        return torch.sigmoid(self.net(t))  # Use sigmoid if needed for specific constraints

    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("relu")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Apply the weight initialization to the network
        self.net.apply(init_weights)

# class ParameterNet(nn.Module):
#     def __init__(self, num_layers=2, hidden_neurons=10):
#         super(ParameterNet, self).__init__()
#         self.retain_seed = 100
#         torch.manual_seed(self.retain_seed)  # Ensure reproducibility

#         layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]  # Input layer

#         for _ in range(num_layers - 1):
#             layers.extend(
#                 [nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]
#             )  # Hidden layers

#         layers.append(nn.Linear(hidden_neurons, 1))  # Output layer
#         self.net = nn.Sequential(*layers)

#         self.init_xavier()  # Initialize the weights using Xavier initialization

#     def forward(self, t):
#         return torch.sigmoid(self.net(t))

#     def init_xavier(self):
#         def init_weights(layer):
#             if isinstance(layer, nn.Linear):
#                 g = nn.init.calculate_gain("tanh")
#                 nn.init.xavier_normal_(layer.weight, gain=g)
#                 if layer.bias is not None:
#                     layer.bias.data.fill_(0)

#         # Apply the weight initialization to the network
#         self.net.apply(init_weights)


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

    S_data = N - torch.sum(tensor_data, dim=1)

    # Constants based on the table provided
    rho = 0.80  # Proportion of symptomatic infections
    alpha = 1 / 5  # Incubation period (5 days)
    ds = 1 / 4  # Infectious period for symptomatic (4 days)
    da = 1 / 7  # Infectious period for asymptomatic (7 days)
    dH = 1 / 13.4  # Hospitalization days (13.4 days)

    # learned parameters
    beta, omega, mu, gamma_c, delta_c, eta = parameters

    # Compute the loss function
    S_t = grad(S_pred, t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    E_t = grad(E_pred, t, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]
    Ia_t = grad(Ia_pred, t, grad_outputs=torch.ones_like(Ia_pred), create_graph=True)[0]
    Is_t = grad(Is_pred, t, grad_outputs=torch.ones_like(Is_pred), create_graph=True)[0]
    H_t = grad(H_pred, t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    C_t = grad(C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
    R_t = grad(R_pred, t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    D_t = grad(D_pred, t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]

    # Compute the differential equations
    dSdt = -beta * (Is_pred + Ia_pred) / N * S_pred + eta * R_pred
    dEdt = beta * (Is_pred + Ia_pred) / N * S_pred - alpha * E_pred
    dIsdt = alpha * rho * E_pred - ds * Is_pred
    dIadt = alpha * (1 - rho) * E_pred - da * Ia_pred
    dHdt = ds * omega * Is_pred - dH * H_pred - mu * H_pred
    dCdt = dH * (1 - omega) * H_pred - gamma_c * C_pred - delta_c * C_pred
    dRdt = (
        ds * (1 - omega) * Is_pred
        + da * Ia_pred
        + dH * (1 - mu) * H_pred
        + gamma_c * C_pred
        - eta * R_pred
    )
    dDdt = mu * H_pred + delta_c * C_pred

    # Compute the loss function
    # data loss
    data_loss = (
        torch.mean((S_pred - S_data) ** 2)
        + torch.mean((Is_pred - Is_data) ** 2)
        + torch.mean((H_pred - H_data) ** 2)
        + torch.mean((C_pred - C_data) ** 2)
        + torch.mean((D_pred - D_data) ** 2)
    )

    # residual loss
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

    # total loss
    loss = data_loss + residual_loss
    return loss, beta, omega, mu, gamma_c, delta_c, eta


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
        
# Initialize the model and optimizer
model = EpiNet(num_layers=5, hidden_neurons=20, output_size=8).to(device)

# Separate networks for each parameter
beta_net = ParameterNet(num_layers=1, hidden_neurons=5).to(device)
omega_net = ParameterNet(num_layers=1, hidden_neurons=5).to(device)
mu_net = ParameterNet(num_layers=1, hidden_neurons=5).to(device)
gamma_c_net = ParameterNet(num_layers=1, hidden_neurons=5).to(device)
delta_c_net = ParameterNet(num_layers=1, hidden_neurons=5).to(device)
eta_net = ParameterNet(num_layers=1, hidden_neurons=5).to(device)

# population

# Define the optimizer and scheduler
optimizer = optim.Adam(
    list(model.parameters())
    + list(beta_net.parameters())
    + list(omega_net.parameters())
    + list(mu_net.parameters())
    + list(gamma_c_net.parameters())
    + list(delta_c_net.parameters())
    + list(eta_net.parameters()),
    lr=3e-4,
)

scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)

# Initialize the early stopping object
earlystopping = EarlyStopping(patience=100, verbose=False)

# Define the number of epochs
num_epochs = 100000

t = (
    torch.linspace(0, len(data_scaled) - 1, len(data_scaled))
    .view(-1, 1)
    .to(device)
    .requires_grad_(True)
)

# Initialize the loss history
loss_history = []


# Train the model
def train_model(
    model,
    param_nets,
    optimizer,
    t,
    data_scaled,
    earlystopping,
    num_epochs,
    loss_history,
    scheduler,
):
    """
    Trains the model using the given parameters.

    Args:
        model (torch.nn.Module): The model to be trained.
        param_nets (list): List of parameter networks.
        optimizer (torch.optim.Optimizer): The optimizer for the model.
        t (torch.Tensor): The input tensor.
        data_scaled (torch.Tensor): The scaled data tensor.
        earlystopping (EarlyStopping): The early stopping object.
        num_epochs (int): The number of epochs to train for.
        loss_history (list): The list to store the loss values.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler for the optimizer.
        lambda_reg (float, optional): The regularization parameter. Defaults to 1e-4.

    Returns:
        tuple: A tuple containing the trained model, parameter networks, and the loss history.
    """
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for net in param_nets:
            net.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        model_output = model(t)
        beta = param_nets[0](t)
        omega = param_nets[1](t)
        mu = param_nets[2](t)
        gamma_c = param_nets[3](t)
        delta_c = param_nets[4](t)
        eta = param_nets[5](t)

        parameters = [beta, omega, mu, gamma_c, delta_c, eta]

        # Calculate the loss
        loss = einn_loss(model_output, data_scaled, parameters, t)[0]

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Save the loss
        loss_history.append(loss.item())

        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")

        # Update the learning rate
        scheduler.step()

        # Check for early stopping
        earlystopping(loss, model)
        if earlystopping.early_stop:
            print("Early stopping")
            break

    return model, param_nets, loss_history, parameters


# Function to plot the loss history
def plot_loss(losses, title="Training Loss", filename=None):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(losses) + 1), losses, label="Loss", color="black")
    plt.yscale("log")
    plt.title(f"{title}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, format="pdf", dpi=600)
    plt.show()


# Train the model
model, param_nets, loss_history, parameters = train_model(
    model,
    [beta_net, omega_net, mu_net, gamma_c_net, delta_c_net, eta_net],
    optimizer,
    t,
    data_scaled,
    earlystopping,
    num_epochs,
    loss_history,
    scheduler,
)

# Plot the loss history
plot_loss(
    loss_history,
    title="Training Loss",
    filename="../../reports/England/loss_history.pdf",
)


# Plot time-varying parameters
def plot_parameters(param_nets, t, filename=None):
    params = ["beta", "omega", "mu", "gamma_c", "delta_c", "eta"]
    plt.figure(figsize=(10, 12))
    for i, param in enumerate(params):
        plt.subplot(3, 2, i + 1)
        param_values = param_nets[i](t).detach().cpu().numpy()
        plt.plot(t.cpu().detach().numpy(), param_values, label=param)
        plt.title(f"{param} over time")
        plt.xlabel("Time")
        plt.ylabel(param)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, format="pdf", dpi=600)
    plt.show()


plot_parameters(
    [beta_net, omega_net, mu_net, gamma_c_net, delta_c_net, eta_net],
    t,
    filename="../../reports/England/parameters_over_time.pdf",
)


def plot_fitting(data, model_output, scaler, filename=None):
    """
    Plots the fitting of the model's predictions to the actual data.
    
    Args:
        data (pd.DataFrame): The original data containing dates and actual values.
        model_output (torch.Tensor): The model's predictions.
        scaler (MinMaxScaler): The scaler used for transforming the data.
        filename (str, optional): The filename to save the plot. Defaults to None.
    """
    dates = data["date"]
    actuals = data[features].values
    predictions = model_output.detach().cpu().numpy()
    
    # Select the relevant columns for the inverse transformation
    predictions_to_scale = predictions[:, :len(features)]
    
    # Inverse transform the predictions
    predictions_scaled = scaler.inverse_transform(predictions_to_scale)
    
    labels = ["new_confirmed", "newAdmissions", "covidOccupiedMVBeds", "new_deceased"]

    plt.figure(figsize=(20, 6))
    for i, label in enumerate(labels):
        plt.subplot(1, 4, i + 1)
        plt.plot(dates, actuals[:, i], label="Actual", color='blue')
        plt.plot(dates, predictions_scaled[:, i], label="Predicted", color='orange')
        plt.title(f"Fitting of {label} over time")
        plt.xlabel("Date")
        plt.ylabel(label)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, format="pdf", dpi=600)
    plt.show()



# Get model predictions

model_output = model(t)

# Plot the fitting of the available data
plot_fitting(
    data,
    model_output,
    scaler,
    filename="../../reports/England/fitting_of_available_data.pdf"
)

# Plot the inference of unobserved dynamics
def plot_inference_unobserved_dynamics(model_output, filename=None):
    states = ["S", "E", "Ia", "R"]
    model_output_np = model_output.detach().cpu().numpy()

    plt.figure(figsize=(30, 6))
    for i, state in enumerate(states):
        plt.subplot(1, 8, i + 1)
        plt.plot(t.cpu().detach().numpy(), model_output_np[:, i], label=state)
        plt.title(f"Inference of {state} over time")
        plt.xlabel("Time")
        plt.ylabel(state)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, format="pdf", dpi=600)
    plt.show()


# Plot the inference of unobserved dynamics
plot_inference_unobserved_dynamics(
    model_output, filename="../../reports/England/inference_unobserved_dynamics.pdf"
)
