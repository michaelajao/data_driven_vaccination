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

    # Select the columns of interest
    # df = df[df["nhs_region"] == areaname].reset_index(drop=True)

    # # reset the index
    # df = df[::-1].reset_index(drop=True)  # Reverse dataset if needed

    # Convert the date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    # calculate the recovered column from data
    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df[
        "cumulative_deceased"
    ].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)

    # select data up to the end_date
    if end_date:
        df = df[df["date"] <= end_date]

    # calculate the susceptible column from data
    df["susceptible"] = (
        df["population"]
        - df["cumulative_confirmed"]
        - df["cumulative_deceased"]
        - df["recovered"]
    )

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
    ]
    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    # normalize the data with the population
    # df["cumulative_confirmed"] = df["cumulative_confirmed"] / df["population"]
    # df["cumulative_deceased"] = df["cumulative_deceased"] / df["population"]
    # df["hospitalCases"] = df["hospitalCases"] / df["population"]
    # df["covidOccupiedMVBeds"] = df["covidOccupiedMVBeds"] / df["population"]
    # df["new_deceased"] = df["new_deceased"] / df["population"]
    # df["new_confirmed"] = df["new_confirmed"] / df["population"]
    # df["recovered"] = df["recovered"] / df["population"]
    # df["population"] = df["population"] / df["population"]
    # df["susceptible"] = df["susceptible"] / df["population"]
    # df["newAdmissions"] = df["newAdmissions"] / df["population"]
    # df["cumAdmissions"] = df["cumAdmissions"] / df["population"]

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
    R = tensor(data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
    return I, H, C, D, R


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
    "recovered",
]

# Split and scale the data
data_scaled, scaler = scale_data(data, features, device)


class EpiNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=6):
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


class BetaNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10):
        super(BetaNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)  # Ensure reproducibility

        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]  # Input layer

        for _ in range(num_layers - 1):
            layers.extend(
                [nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]
            )  # Hidden layers

        layers.append(nn.Linear(hidden_neurons, 6))  # Output layer
        self.net = nn.Sequential(*layers)

        self.init_xavier()  # Initialize the weights using Xavier initialization

    def forward(self, t):
        return self.net(t)  # Forward pass

    # time varying parameters estimation
    def get_params(self, t):
        beta, omega, mu, gamma_c, delta_c, eta = torch.split(self.net(t), 1, dim=1)
        beta = torch.sigmoid(beta)
        omega = torch.sigmoid(omega)
        mu = torch.sigmoid(mu)
        gamma_c = torch.sigmoid(gamma_c)
        delta_c = torch.sigmoid(delta_c)
        eta = torch.sigmoid(eta)

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


def einn_loss(model_output, tensor_data, parameters, t, model, lambda_reg=1e-4):
    """Compute the loss function for the EINN model with L2 regularization."""

    # Split the model output into the different compartments
    S_pred, E_pred, Ia_pred, Is_pred, H_pred, C_pred, D_pred, R_pred = torch.split(
        model_output, 1, dim=1
    )

    # Normalize the data
    N = 1

    Is_data, H_data, C_data, D_data, R_data = (
        tensor_data[:, 0].view(-1, 1),
        tensor_data[:, 1].view(-1, 1),
        tensor_data[:, 2].view(-1, 1),
        tensor_data[:, 3].view(-1, 1),
        tensor_data[:, 4].view(-1, 1),
    )

    # initial data for E_data and Ia_data should be zero
    E_data = torch.zeros_like(Is_data)
    Ia_data = torch.zeros_like(Is_data)
    S_data = N - Ia_data - Is_data - H_data - C_data - R_data - D_data

    # Constants based on the table provided
    rho = 0.80  # Proportion of symptomatic infections
    alpha = 1 / 5  # Incubation period (5 days)
    d_s = 1 / 4  # Infectious period for symptomatic (4 days)
    d_a = 1 / 7  # Infectious period for asymptomatic (7 days)
    d_h = 1 / 13.4  # Hospitalization days (13.4 days)

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
    dSdt = -beta * (Is_data + Ia_data) / N * S_data + eta * R_data
    dEdt = beta * (Is_data + Ia_data) / N * S_data - alpha * E_data
    dIsdt = alpha * rho * E_data - d_s * Is_data
    dIadt = alpha * (1 - rho) * E_data - d_a * Ia_data
    dHdt = d_s * omega * Is_data - d_h * H_data - mu * H_data
    dCdt = d_h * (1 - omega) * H_data - gamma_c * C_data - delta_c * C_data
    dRdt = (
        d_s * (1 - omega) * Is_data
        + d_a * Ia_data
        + d_h * (1 - mu) * H_data
        + gamma_c * C_data
        - eta * R_data
    )
    dDdt = mu * H_data + delta_c * C_data

    # Compute the loss function
    # data loss
    data_loss = (
        torch.mean((S_pred - S_data) ** 2)
        + torch.mean((E_pred - E_data) ** 2)
        + torch.mean((Ia_pred - Ia_data) ** 2)
        + torch.mean((Is_pred - Is_data) ** 2)
        + torch.mean((H_pred - H_data) ** 2)
        + torch.mean((C_pred - C_data) ** 2)
        + torch.mean((R_pred - R_data) ** 2)
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
    
    # parameter loss
    beta_loss = torch.mean(beta ** 2)
    omega_loss = torch.mean(omega ** 2)
    mu_loss = torch.mean(mu ** 2)
    gamma_c_loss = torch.mean(gamma_c ** 2)
    delta_c_loss = torch.mean(delta_c ** 2)
    eta_loss = torch.mean(eta ** 2)
    
    param_loss = beta_loss + omega_loss + mu_loss + gamma_c_loss + delta_c_loss + eta_loss

    # # Initial condition loss
    # S0, E0, Ia0, Is0, H0, C0, R0, D0 = S_data[0], E_data[0], Ia_data[0], Is_data[0], H_data[0], C_data[0], R_data[0], D_data[0]
    # initial_loss = (
    #     (S_pred[0] - S0) ** 2
    #     + (E_pred[0] - E0) ** 2
    #     + (Ia_pred[0] - Ia0) ** 2
    #     + (Is_pred[0] - Is0) ** 2
    #     + (H_pred[0] - H0) ** 2
    #     + (C_pred[0] - C0) ** 2
    #     + (R_pred[0] - R0) ** 2
    #     + (D_pred[0] - D0) ** 2
    # )

    # # L2 regularization term
    # l2_reg = torch.tensor(0.).to(device)
    # for param in model.parameters():
    #     l2_reg += torch.norm(param)

    # total loss
    loss = data_loss + residual_loss + param_loss

    return loss


# define the early stopping class
# class EarlyStopping:
#     def __init__(self, patience=7, verbose=False, delta=0, path="checkpoint.pt"):
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#         self.path = path

#     def __call__(self, val_loss, model):

#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0

#     def save_checkpoint(self, val_loss, model):
#         if self.verbose:
#             print(
#                 f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
#             )
#         torch.save(model.state_dict(), self.path)
#         self.val_loss_min = val_loss
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

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

# Initialize the model and optimizer
model = EpiNet(num_layers=5, hidden_neurons=20, output_size=8).to(device)
beta_net = BetaNet(num_layers=1, hidden_neurons=5).to(device)

# population
N = data["population"].iloc[0]

# Define the optimizer and scheduler
# optimizer = optim.Adam(list(model.parameters()) + list(beta_net.parameters()), lr=1e-4)
model_optimizer = optim.Adam(model.parameters(), lr=1e-4)
params_optimizer = optim.Adam(model.parameters(), lr=1e-4)

# scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)
model_scheduler = StepLR(model_optimizer, step_size=5000, gamma=0.9)
params_scheduler = StepLR(params_optimizer, step_size=5000, gamma=0.9)


earlystopping = EarlyStopping(patience=100, verbose=False)
num_epochs = 100000

t = (
    torch.tensor(np.arange(1, len(data) + 1), dtype=torch.float32)
    .view(-1, 1)
    .to(device)
    .requires_grad_(True)
)

# Initialize the loss history
loss_history = []

# # train the model
# def train_model(model, beta_net, optimizer, t, data_scaled, N, earlystopping, num_epochs, loss_history, scheduler, lambda_reg=1e-4):
#     for epoch in tqdm(range(num_epochs)):
#         model.train()
#         beta_net.train()

#         # Zero the gradients
#         optimizer.zero_grad()

#         # Forward pass
#         model_output = model(t)
#         parameters = beta_net.get_params(t)

#         # Calculate the loss
#         loss = einn_loss(model_output, data_scaled, parameters, t, model, lambda_reg)
#         # Backward pass
#         loss.backward()

#         # Update the weights
#         optimizer.step()

#         # Update the learning rate
#         scheduler.step()

#         # Save the loss
#         loss_history.append(loss.item())

#         # Early stopping
#         earlystopping(loss.item())
#         if earlystopping.early_stop:
#             print("Early stopping")
#             break

#         if (epoch + 1) % 500 == 0 or epoch == 0:
#             print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
#             beta, omega, mu, gamma_c, delta_c, eta = parameters
#             print(f"beta (first 5 values): {beta[:5].cpu().detach().numpy().flatten()}")
#             print(f"omega (first 5 values): {omega[:5].cpu().detach().numpy().flatten()}")
#             print(f"mu (first 5 values): {mu[:5].cpu().detach().numpy().flatten()}")
#             print(f"gamma_c (first 5 values): {gamma_c[:5].cpu().detach().numpy().flatten()}")
#             print(f"delta_c (first 5 values): {delta_c[:5].cpu().detach().numpy().flatten()}")
#             print(f"eta (first 5 values): {eta[:5].cpu().detach().numpy().flatten()}")

#     return model, beta_net, loss_history


# Train the model
def train_model(
    model,
    beta_net,
    model_optimizer,
    params_optimizer,
    t,
    data_scaled,
    earlystopping,
    num_epochs,
    loss_history,
    model_scheduler,
    params_scheduler,
    lambda_reg=1e-4,
):
    """
    Trains the model using the given parameters.

    Args:
        model (torch.nn.Module): The model to be trained.
        beta_net (torch.nn.Module): The beta network.
        model_optimizer (torch.optim.Optimizer): The optimizer for the model.
        params_optimizer (torch.optim.Optimizer): The optimizer for the parameters.
        t (torch.Tensor): The input tensor.
        data_scaled (torch.Tensor): The scaled data tensor.
        earlystopping (EarlyStopping): The early stopping object.
        num_epochs (int): The number of epochs to train for.
        loss_history (list): The list to store the loss values.
        model_scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler for the model optimizer.
        params_scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler for the params optimizer.
        lambda_reg (float, optional): The regularization parameter. Defaults to 1e-4.

    Returns:
        tuple: A tuple containing the trained model, beta network, and the loss history.
    """
    for epoch in tqdm(range(num_epochs)):
        model.train()
        beta_net.train()

        # Zero the gradients
        model_optimizer.zero_grad()
        params_optimizer.zero_grad()

        # Forward pass
        model_output = model(t)
        parameters = beta_net.get_params(t)

        # Calculate the loss
        loss = einn_loss(model_output, data_scaled, parameters, t, model, lambda_reg)
        # Backward pass
        loss.backward()

        # Update the weights
        model_optimizer.step()
        params_optimizer.step()

        # Update the learning rate
        model_scheduler.step()
        params_scheduler.step()

        # Save the loss
        loss_history.append(loss.item())

        # Early stopping
        earlystopping(loss.item())
        if earlystopping.early_stop:
            print("Early stopping")
            break

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
            # beta, omega, mu, gamma_c, delta_c, eta = parameters
            # print(f"beta (first 5 values): {beta[:5].cpu().detach().numpy().flatten()}")
            # print(f"omega (first 5 values): {omega[:5].cpu().detach().numpy().flatten()}")
            # print(f"mu (first 5 values): {mu[:5].cpu().detach().numpy().flatten()}")
            # print(f"gamma_c (first 5 values): {gamma_c[:5].cpu().detach().numpy().flatten()}")
            # print(f"delta_c (first 5 values): {delta_c[:5].cpu().detach().numpy().flatten()}")
            # print(f"eta (first 5 values): {eta[:5].cpu().detach().numpy().flatten()}")

    return model, beta_net, loss_history


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


# # Train the model
# model, beta_net, loss_history = train_model(model, beta_net, optimizer, t, data_scaled, N, earlystopping, num_epochs, loss_history, scheduler)

# Train the model
model, beta_net, loss_history = train_model(
    model,
    beta_net,
    model_optimizer,
    params_optimizer,
    t,
    data_scaled,
    earlystopping,
    num_epochs,
    loss_history,
    model_scheduler,
    params_scheduler,
)



# Plot the loss history
plot_loss(
    loss_history,
    title="Training Loss",
    filename="../../reports/England/loss_history.pdf",
)


# plot the predicted values
def plot_predictions(t, model, N, data, scaler, areaname, filename):
    with torch.no_grad():
        predictions = model(t).cpu().numpy()

    t_np = t.cpu().detach().numpy().flatten()

    S_pred, E_pred, Ia_pred, Is_pred, H_pred, C_pred, D_pred, R_pred = (
        predictions[:, 0],
        predictions[:, 1],
        predictions[:, 2],
        predictions[:, 3],
        predictions[:, 4],
        predictions[:, 5],
        predictions[:, 6],
        predictions[:, 7],
    )

    fig, axs = plt.subplots(8, 1, figsize=(10, 30), sharex=True)

    # Plotting S (Susceptible)
    axs[0].plot(t_np, S_pred, "r-", label="$S_{PINN}$")
    axs[0].set_title("Susceptible")
    axs[0].legend()

    # Plotting E (Exposed)
    axs[1].plot(t_np, E_pred, "r-", label="$E_{PINN}$")
    axs[1].set_title("Exposed")
    axs[1].legend()

    # Plotting Ia (Asymptomatic)
    axs[2].plot(t_np, Ia_pred, "r-", label="$Ia_{PINN}$")
    axs[2].set_title("Asymptomatic")
    axs[2].legend()

    # Plotting Is (Symptomatic)
    axs[3].plot(t_np, Is_pred, "r-", label="$Is_{PINN}$")
    axs[3].set_title("Symptomatic")
    axs[3].legend()

    # Plotting H (Hospitalized)
    axs[4].plot(t_np, H_pred, "r-", label="$H_{PINN}$")
    axs[4].set_title("Hospitalized")
    axs[4].legend()

    # Plotting C (Critical)
    axs[5].plot(t_np, C_pred, "r-", label="$C_{PINN}$")
    axs[5].set_title("Critical")
    axs[5].legend()

    # Plotting R (Recovered)
    axs[6].plot(t_np, R_pred, "r-", label="$R_{PINN}$")
    axs[6].set_title("Recovered")
    axs[6].legend()

    # Plotting D (Deceased)
    axs[7].plot(t_np, D_pred, "r-", label="$D_{PINN}$")
    axs[7].set_title("Deceased")
    axs[7].legend()

    plt.tight_layout()
    # plt.savefig(filename, format='pdf', dpi=600)
    plt.show()


# Plot the predicted values
plot_predictions(
    t,
    model,
    N,
    data_scaled,
    scaler,
    areaname,
    filename="../../reports/England/predictions.pdf",
)

# plot the time varying parameters


# Debugging by printing initial and final parameters
def plot_parameters(t, beta_net, filename):
    with torch.no_grad():
        beta, omega, mu, gamma_c, delta_c, eta = beta_net.get_params(t)
        
    beta = beta.cpu().detach().numpy().flatten()
    omega = omega.cpu().detach().numpy().flatten()
    mu = mu.cpu().detach().numpy().flatten()
    gamma_c = gamma_c.cpu().detach().numpy().flatten()
    delta_c = delta_c.cpu().detach().numpy().flatten()
    eta = eta.cpu().detach().numpy().flatten()
    

    # plot the parameters

    fig, axs = plt.subplots(6, 1, figsize=(10, 20), sharex=True)

    # Plotting beta
    axs[0].plot(t_np, beta, "r-", label="$\\beta_{\mathrm{PINN}}$")
    axs[0].set_title("Transmission Rate")
    axs[0].legend()

    # Plotting omega
    axs[1].plot(t_np, omega, "r-", label="$\\omega_{\mathrm{PINN}}$")
    axs[1].set_title("Hospitalization Rate")
    axs[1].legend()

    # Plotting mu
    axs[2].plot(t_np, mu, "r-", label="$\\mu_{\mathrm{PINN}}$")
    axs[2].set_title("Mortality Rate")
    axs[2].legend()

    # Plotting gamma_c
    axs[3].plot(t_np, gamma_c, "r-", label="$\\gamma_{c, \mathrm{PINN}}$")
    axs[3].set_title("Recovery Rate")
    axs[3].legend()

    # Plotting delta_c
    axs[4].plot(t_np, delta_c, "r-", label="$\\delta_{c, \mathrm{PINN}}$")
    axs[4].set_title("Critical Mortality Rate")
    axs[4].legend()

    # Plotting eta
    axs[5].plot(t_np, eta, "r-", label="$\\eta_{\mathrm{PINN}}$")
    axs[5].set_title("Recovered Rate")
    axs[5].legend()

    plt.tight_layout()
    plt.savefig(filename, format="pdf", dpi=600)
    plt.show()
    print("Initial Parameters:")
    print("beta:", beta[:5])
    print("omega:", omega[:5])
    print("mu:", mu[:5])
    print("gamma_c:", gamma_c[:5])
    print("delta_c:", delta_c[:5])
    print("eta:", eta[:5])

    print("Final Parameters:")
    print("beta:", beta[-5:])
    print("omega:", omega[-5:])
    print("mu:", mu[-5:])
    print("gamma_c:", gamma_c[-5:])
    print("delta_c:", delta_c[-5:])
    print("eta:", eta[-5:])


# Adjusted parameter scaling and plot functions
# Plot the time varying parameters
plot_parameters(t, beta_net, filename="../../reports/England/parameters.pdf")


# # plot the R0 and Rt values
# def plot_R0_Rt(t, beta_net, filename):
#     beta, omega, mu, gamma_c, delta_c, eta = beta_net.get_params(t)

#     beta = beta.cpu().detach().numpy().flatten()
#     omega = omega.cpu().detach().numpy().flatten()
#     mu = mu.cpu().detach().numpy().flatten()
#     gamma_c = gamma_c.cpu().detach().numpy().flatten()
#     delta_c = delta_c.cpu().detach().numpy().flatten()
#     eta = eta.cpu().detach().numpy().flatten()

#     t_np = t.cpu().detach().numpy().flatten()

#     # Calculate R0 and Rt
#     R0 = beta / (mu + gamma_c + delta_c)
#     Rt = beta / (mu + gamma_c + delta_c + eta)

#     # Plot the R0 and Rt values
#     fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

#     # Plotting R0
#     axs[0].plot(t_np, R0, 'r-', label='$R_0$')
#     axs[0].set_title('Basic Reproduction Number ($R_0$)')
#     axs[0].legend()

#     # Plotting Rt
#     axs[1].plot(t_np, Rt, 'r-', label='$R_t$')
#     axs[1].set_title('Effective Reproduction Number ($R_t$)')
#     axs[1].legend()

#     # show the line at R0 = 1
#     axs[0].axhline(y=1, color='black', linestyle='--', label='$R_0 = 1$')
#     axs[1].axhline(y=1, color='black', linestyle='--', label='$R_t = 1$')

#     plt.tight_layout()
#     # plt.savefig(filename, format='pdf', dpi=600)
#     plt.show()

#     print("Initial R0 and Rt values:")
#     print("R0:", R0[:5])
#     print("Rt:", Rt[:5])

#     print("Final R0 and Rt values:")
#     print("R0:", R0[-5:])
#     print("Rt:", Rt[-5:])

# # Plot the R0 and Rt values
# plot_R0_Rt(t, beta_net, filename="../../reports/England/R0_Rt.pdf")


# # plot the result
# def plot_results(t, model, N, data, scaler, areaname, filename):
#     model.eval()
#     with torch.no_grad():
#         predictions = model(t).cpu().numpy()

#     t_np = t.cpu().detach().numpy().flatten()

#     I_data, H_data, C_data, D_data, R_data = data[:, 0].view(-1, 1), data[:, 1].view(-1, 1), data[:, 2].view(-1, 1), data[:, 3].view(-1, 1), data[:, 4].view(-1, 1)

#     I_pred, H_pred, C_pred, D_pred, R_pred = predictions[:, 3], predictions[:, 4], predictions[:, 5], predictions[:, 6], predictions[:, 7]

#     fig, axs = plt.subplots(5, 1, figsize=(10, 20), sharex=True)

#     # Plotting I (Infected)
#     axs[0].scatter(t_np, I_data.cpu().detach().numpy().flatten(), color='black', label='$I_{Data}$', s=10)
#     axs[0].plot(t_np, I_pred, 'r-', label='$I_{PINN}$')
#     axs[0].set_title('Infected')
#     axs[0].legend()

#     # Plotting H (Hospitalized)
#     axs[1].scatter(t_np, H_data.cpu().detach().numpy().flatten(), color='black', label='$H_{Data}$', s=10)
#     axs[1].plot(t_np, H_pred, 'r-', label='$H_{PINN}$')
#     axs[1].set_title('Hospitalized')
#     axs[1].legend()

#     # Plotting C (Critical)
#     axs[2].scatter(t_np, C_data.cpu().detach().numpy().flatten(), color='black', label='$C_{Data}$', s=10)
#     axs[2].plot(t_np, C_pred, 'r-', label='$C_{PINN}$')
#     axs[2].set_title('Critical')
#     axs[2].legend()

#     # Plotting R (Recovered)
#     axs[3].scatter(t_np, R_data.cpu().detach().numpy().flatten(), color='black', label='$R_{Data}$', s=10)
#     axs[3].plot(t_np, R_pred, 'r-', label='$R_{PINN}$')
#     axs[3].set_title('Recovered')
#     axs[3].legend()

#     # Plotting D (Deceased)
#     axs[4].scatter(t_np, D_data.cpu().detach().numpy().flatten(), color='black', label='$D_{Data}$', s=10)
#     axs[4].plot(t_np, D_pred, 'r-', label='$D_{PINN}$')
#     axs[4].set_title('Deceased')
#     axs[4].legend()

#     plt.tight_layout()
#     plt.savefig(filename, format='pdf', dpi=600)
#     plt.show()

# # Plot the results
# plot_results(t, model, N, data_scaled, scaler, areaname, filename="../../reports/England/predictions.pdf")


# # Plot results
# # def plot_results(t, I_pred,  N):
# #     model.eval()
# #     with torch.no_grad():
# #         predictions = model(t).cpu().numpy()

# #     t_np = t.cpu().detach().numpy().flatten()


# #     fig, axs = plt.subplots(6, 1, figsize=(10, 20))

# #     # Plotting S (Susceptible)
# #     S_pred = N - I_pred - H_pred - C_pred - R_pred - D_pred
# #     axs[0].plot(t_np, S_pred, 'r-', label='$S_{PINN}$')
# #     axs[0].set_title('S')
# #     axs[0].set_xlabel('Time t (days)')
# #     axs[0].legend()

# #     # Plotting I (Infected)
# #     axs[1].scatter(t_np, I_data.cpu().detach().numpy().flatten(), color='black', label='$I_{Data}$', s=10)
# #     axs[1].plot(t_np, I_pred, 'r-', label='$I_{PINN}$')
# #     axs[1].set_title('I')
# #     axs[1].set_xlabel('Time t (days)')
# #     axs[1].legend()

# #     # Plotting H (Hospitalized)
# #     axs[2].plot(t_np, H_pred, 'r-', label='$H_{PINN}$')
# #     axs[2].set_title('H')
# #     axs[2].set_xlabel('Time t (days)')
# #     axs[2].legend()

# #     # Plotting C (Critical)
# #     axs[3].plot(t_np, C_pred, 'r-', label='$C_{PINN}$')
# #     axs[3].set_title('C')
# #     axs[3].set_xlabel('Time t (days)')
# #     axs[3].legend()

# #     # Plotting R (Recovered)
# #     axs[4].scatter(t_np, R_data.cpu().detach().numpy().flatten(), color='black', label='$R_{Data}$', s=10)
# #     axs[4].plot(t_np, R_pred, 'r-', label='$R_{PINN}$')
# #     axs[4].set_title('R')
# #     axs[4].set_xlabel('Time t (days)')
# #     axs[4].legend()

# #     # Plotting D (Deceased)
# #     axs[5].scatter(t_np, D_data.cpu().detach().numpy().flatten(), color='black', label='$D_{Data}$', s=10)
# #     axs[5].plot(t_np, D_pred, 'r-', label='$D_{PINN}$')
# #     axs[5].set_title('D')
# #     axs[5].set_xlabel('Time t (days)')
# #     axs[5].legend()

# #     plt.tight_layout()
# #     # plt.savefig(f"../../reports/figures/{title}.pdf")
# #     plt.show()


# # Plot the actual vs predicted values``
