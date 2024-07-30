import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator
import scienceplots
from tqdm.notebook import tqdm
from scipy.integrate import odeint
from collections import deque
import torch
import torch.nn as nn
from torch.autograd import grad
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torchsummary import summary
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


# Device setup for CUDA or CPU
def get_device():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            if torch.cuda.get_device_name(i):
                return torch.device(f"cuda:{i}")
    return torch.device("cpu")


device = get_device()
print(f"Using device: {device}")

# Set a specific style for research paper quality plots
plt.style.use(['science', 'ieee', 'no-latex'])
# Customizing color cycle with monochrome settings for clarity in black-and-white printing
mark_every = 0.1
monochrome = (
    cycler("color", ["k"])
    * cycler("markevery", [mark_every])
    * cycler("marker", ["", "o", "^", "s", "v"])
    * cycler("linestyle", ["-", "--", ":", (0, (5, 2, 5, 5, 1, 4))])
)

plt.rc("axes", prop_cycle=monochrome)

# Update matplotlib rcParams for consistent plot settings
plt.rcParams.update(
    {
        "font.size": 16,
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
        "axes.labelsize": 14,
        "axes.titlesize": 20,
        "axes.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.formatter.limits": (0, 5),
        "axes.formatter.use_mathtext": True,
        "axes.formatter.useoffset": False,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        "legend.fontsize": 16,
        "legend.frameon": True,
        "legend.loc": "best",
        "lines.linewidth": 2.5,
        "lines.markersize": 10,
        "xtick.labelsize": 14,
        "xtick.direction": "in",
        "xtick.top": True,
        "ytick.labelsize": 14,
        "ytick.direction": "in",
        "ytick.right": True,
        "grid.color": "grey",
        "grid.linestyle": "--",
        "grid.linewidth": 0.75,
        "errorbar.capsize": 4,
        "figure.subplot.wspace": 0.4,
        "figure.subplot.hspace": 0.4,
        "image.cmap": "viridis",
        "lines.antialiased": True,
        "patch.antialiased": True,
        "text.antialiased": True,
        "axes.labelpad": 10,
        "axes.titlepad": 15,
        "xtick.major.pad": 5,
        "ytick.major.pad": 5,
        "figure.subplot.left": 0.1,
        "figure.subplot.right": 0.95,
        "figure.subplot.bottom": 0.1,
        "figure.subplot.top": 0.9,
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


def calculate_errors(y_true, y_pred, y_train):
    """Calculate and return various error metrics."""
    nrmse = normalized_root_mean_square_error(y_true, y_pred)
    mase = safe_mean_absolute_scaled_error(y_true, y_pred, y_train)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return nrmse, mase, mae, mape


def save_metrics(metrics, areaname):
    """Save metrics to a CSV file."""
    metrics_df = pd.DataFrame(metrics, columns=["Metric", "Value"])
    metrics_df.to_csv(f"../../reports/results/{areaname}_metrics.csv", index=False)
    print(f"Metrics saved to ../../reports/results/{areaname}_metrics.csv")


# Define the SEIRD model differential equations
def seird_model(
    y, t, N, beta, alpha, rho, ds, da, omega, dH, mu, gamma_c, delta_c, eta
):
    S, E, Is, Ia, H, C, R, D = y

    dSdt = -beta * (Is + Ia) / N * S + (eta * R)
    dEdt = beta * (Is + Ia) / N * S - (alpha * E)
    dIsdt = alpha * rho * E - ds * Is
    dIadt = alpha * (1 - rho) * E - (da * Ia)
    dHdt = ds * omega * Is - (dH * H) - (mu * H)
    dCdt = dH * (1 - omega) * (H - gamma_c * C) - delta_c * C
    dRdt = (
        ds * (1 - omega) * Is + (da * Ia) + dH * (1 - mu) * H + (gamma_c * C) - eta * R
    )
    dDdt = mu * H + (delta_c * C)

    return [dSdt, dEdt, dIsdt, dIadt, dHdt, dCdt, dRdt, dDdt]


def load_and_preprocess_data(
    filepath, areaname, rolling_window=7, start_date="2020-04-01", end_date="2021-08-31"
):
    df = pd.read_csv(filepath)
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


data = load_and_preprocess_data(
    "../../data/processed/england_data.csv",
    "England",
    rolling_window=7,
    start_date="2020-05-01",
    end_date="2021-12-31",
)

# split data into training and validation sets, the validation should be the last 7 days
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

        self._rho = nn.Parameter(
            torch.tensor([torch.rand(1)], device=device), requires_grad=True
        )
        self._alpha = nn.Parameter(
            torch.tensor([torch.rand(1)], device=device), requires_grad=True
        )
        self._ds = nn.Parameter(
            torch.tensor([torch.rand(1)], device=device), requires_grad=True
        )
        self._da = nn.Parameter(
            torch.tensor([torch.rand(1)], device=device), requires_grad=True
        )
        self._dH = nn.Parameter(
            torch.tensor([torch.rand(1)], device=device), requires_grad=True
        )

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

        # for _ in range(num_layers - 1):
        #     layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        # Hidden layers with residual connections
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_neurons))

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

    data_loss = (
        torch.mean((Is_pred - Is_data) ** 2)
        + torch.mean((H_pred - H_data) ** 2)
        + torch.mean((C_pred - C_data) ** 2)
        + torch.mean((D_pred - D_data) ** 2)
    )

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

    loss = data_loss + residual_loss + initial_cost
    return loss


class EarlyStopping:
    def __init__(self, patience=200, verbose=False, delta=0):
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


def train_model(
    model,
    parameter_net,
    optimizer,
    scheduler,
    time_stamps,
    data_scaled,
    num_epochs=100,
    early_stopping=None,
):
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
model = EpiNet(num_layers=6, hidden_neurons=32, output_size=8).to(device)
parameter_net = ParameterNet(num_layers=6, hidden_neurons=32, output_size=6).to(device)
optimizer = optim.Adam(
    list(model.parameters()) + list(parameter_net.parameters()), lr=1e-4
)
scheduler = StepLR(optimizer, step_size=5000, gamma=0.9)

# Early stopping
early_stopping = EarlyStopping(patience=200, verbose=False)

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

# Plot training loss
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.yscale("log")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.savefig("../../reports/figures/training_loss.pdf")
plt.savefig("../../reports/figures/training_loss.png")  # Save as PNG
plt.show()

# Save the trained model
torch.save(model.state_dict(), "../../models/epinet_model3.pth")
torch.save(parameter_net.state_dict(), "../../models/parameter_net3.pth")

# Load the trained model
model.load_state_dict(torch.load("../../models/epinet_model3.pth"))
parameter_net.load_state_dict(torch.load("../../models/parameter_net3.pth"))


def plot_observed_vs_predicted(dates, data, observed_model_output_scaled):
    """Plot observed vs predicted data."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axs[0, 0].plot(dates, data["daily_confirmed"], color="blue")
    axs[0, 0].plot(
        dates,
        observed_model_output_scaled[:, 0],
        linestyle="--",
        color="red",
    )
    axs[0, 0].set_ylabel("New Confirmed Cases")
    axs[0, 0].set_xlabel("Date")

    axs[0, 1].plot(dates, data["daily_hospitalized"], color="blue")
    axs[0, 1].plot(
        dates,
        observed_model_output_scaled[:, 1],
        linestyle="--",
        color="red",
    )
    axs[0, 1].set_ylabel("New Admissions")
    axs[0, 1].set_xlabel("Date")

    axs[1, 0].plot(dates, data["covidOccupiedMVBeds"], color="blue")
    axs[1, 0].plot(
        dates,
        observed_model_output_scaled[:, 2],
        linestyle="--",
        color="red",
    )
    axs[1, 0].set_ylabel("Critical Cases")
    axs[1, 0].set_xlabel("Date")

    axs[1, 1].plot(dates, data["daily_deceased"], color="blue")
    axs[1, 1].plot(
        dates,
        observed_model_output_scaled[:, 3],
        linestyle="--",
        color="red",
    )
    axs[1, 1].set_ylabel("New Deaths")
    axs[1, 1].set_xlabel("Date")

    for ax in axs.flat:
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y")
        ax.legend(["Observed", "Predicted"])

    plt.tight_layout()
    plt.savefig("../../reports/figures/observed_vs_predicted.pdf")
    plt.savefig("../../reports/figures/observed_vs_predicted.png")
    plt.show()


def plot_unobserved_states(dates, model_output):
    """Plot unobserved states."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    axs[0, 0].plot(
        dates,
        model_output[:, 0].cpu(),
        label=r"$S$",
        color="green",
    )
    axs[0, 0].set_ylabel(r"$S$")
    axs[0, 0].set_xlabel("Date")

    axs[0, 1].plot(
        dates, model_output[:, 1].cpu(), label=r"$E$", color="green"
    )
    axs[0, 1].set_ylabel(r"$E$")
    axs[0, 1].set_xlabel("Date")

    axs[1, 0].plot(
        dates,
        model_output[:, 3].cpu(),
        label=r"$I_a$",
        color="green",
    )
    axs[1, 0].set_ylabel(r"$I_a$")
    axs[1, 0].set_xlabel("Date")

    axs[1, 1].plot(
        dates, model_output[:, 6].cpu(), label=r"$R$", color="green"
    )
    axs[1, 1].set_ylabel(r"$R$")
    axs[1, 1].set_xlabel("Date")

    for ax in axs.flat:
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y")

    plt.tight_layout()
    plt.savefig("../../reports/figures/unobserved_outputs.pdf")
    plt.savefig("../../reports/figures/unobserved_outputs.png")
    plt.show()


def plot_time_varying_parameters(dates, parameters):
    """Plot time-varying parameters."""
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    parameters_np = [p.cpu().numpy() for p in parameters]

    latex_labels = [
        r"$\beta$",
        r"$\gamma_c$",
        r"$\delta_c$",
        r"$\eta$",
        r"$\mu$",
        r"$\omega$",
    ]
    colors = [
        "purple",
        "red",
        "orange",
        "blue",
        "green",
        "violet",
    ]

    for i, (ax, param, label, color) in enumerate(
        zip(axs.flat, parameters_np, latex_labels, colors)
    ):
        ax.plot(dates, param, label=label, color=color)
        ax.set_ylabel(label)
        ax.set_xlabel("Date")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()

    plt.tight_layout(pad=2.0)
    plt.savefig("../../reports/figures/time_varying_parameters.pdf", bbox_inches="tight")
    plt.savefig("../../reports/figures/time_varying_parameters.png", bbox_inches="tight")
    plt.show()


def plot_model_outputs(model, time_stamps, parameter_net, data, scaler):
    """Plot the observed vs predicted data and time-varying parameters."""
    model.eval()
    parameter_net.eval()

    with torch.no_grad():
        model_output = model(time_stamps)
        parameters = parameter_net(time_stamps)

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
    dates = data["date"]

    plot_observed_vs_predicted(dates, data, observed_model_output_scaled)
    plot_unobserved_states(dates, model_output)
    plot_time_varying_parameters(dates, parameters)

    return parameters, observed_model_output_scaled


parameters, observed_model_output_scaled = plot_model_outputs(
    model, time_stamps, parameter_net, train_data, scaler
)

I, H, C, D = prepare_tensors(train_data, device)
I_pred, H_pred, C_pred, D_pred = (
    observed_model_output_scaled[:, 0],
    observed_model_output_scaled[:, 1],
    observed_model_output_scaled[:, 2],
    observed_model_output_scaled[:, 3],
)

metrics = []

metric_names = ["NRMSE", "MASE", "MAE", "MAPE"]
areas = ["Infections", "Hospitalizations", "Critical", "Deaths"]
observed_data = [I.cpu().numpy(), H.cpu().numpy(), C.cpu().numpy(), D.cpu().numpy()]
predicted_data = [I_pred, H_pred, C_pred, D_pred]
train_data_values = [
    train_data["daily_confirmed"].values,
    train_data["daily_hospitalized"].values,
    train_data["covidOccupiedMVBeds"].values,
    train_data["daily_deceased"].values,
]

for obs, pred, train, area in zip(
    observed_data, predicted_data, train_data_values, areas
):
    nrmse, mase, mae, mape = calculate_errors(obs, pred, train)
    print(f"Metrics for {area}:")
    print(f"Normalized Root Mean Square Error (NRMSE): {nrmse:.4f}")
    print(f"Mean Absolute Scaled Error (MASE): {mase:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")

    metrics.append([f"{area}_NRMSE", nrmse])
    metrics.append([f"{area}_MASE", mase])
    metrics.append([f"{area}_MAE", mae])
    metrics.append([f"{area}_MAPE", mape])

save_metrics(metrics, "England")


def plot_forecast_and_validation(
    model, parameter_net, train_data, val_data, scaler, device, days, filename
):
    """Plot the forecast and validation data."""
    model.eval()
    parameter_net.eval()

    future_dates = val_data["date"]
    future_time_stamps = (
        torch.tensor(val_data.index.values, dtype=torch.float32).view(-1, 1).to(device)
    )

    with torch.no_grad():
        future_model_output = model(future_time_stamps)

    future_forecast = pd.DataFrame(
        {
            "daily_confirmed": future_model_output[:, 2].cpu().numpy(),
            "daily_hospitalized": future_model_output[:, 4].cpu().numpy(),
            "covidOccupiedMVBeds": future_model_output[:, 5].cpu().numpy(),
            "daily_deceased": future_model_output[:, 7].cpu().numpy(),
        },
        index=future_dates,
    )

    future_forecast_scaled = scaler.inverse_transform(future_forecast)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

    axs[0, 0].plot(
        val_data["date"], val_data["daily_confirmed"], label="Observed", color="blue"
    )
    axs[0, 0].plot(
        val_data["date"],
        future_forecast_scaled[:, 0],
        label="Forecasted",
        linestyle="--",
        color="red",
    )
    axs[0, 0].set_ylabel("New Confirmed Cases")
    axs[0, 0].set_xlabel("Date")

    axs[0, 1].plot(
        val_data["date"], val_data["daily_hospitalized"], label="Observed", color="blue"
    )
    axs[0, 1].plot(
        val_data["date"],
        future_forecast_scaled[:, 1],
        label="Forecasted",
        linestyle="--",
        color="red",
    )
    axs[0, 1].set_ylabel("New Admissions")
    axs[0, 1].set_xlabel("Date")

    axs[1, 0].plot(
        val_data["date"], val_data["covidOccupiedMVBeds"], label="Observed", color="blue"
    )
    axs[1, 0].plot(
        val_data["date"],
        future_forecast_scaled[:, 2],
        label="Forecasted",
        linestyle="--",
        color="red",
    )
    axs[1, 0].set_ylabel("Critical Cases")
    axs[1, 0].set_xlabel("Date")

    axs[1, 1].plot(
        val_data["date"], val_data["daily_deceased"], label="Observed", color="blue"
    )
    axs[1, 1].plot(
        val_data["date"],
        future_forecast_scaled[:, 3],
        label="Forecasted",
        linestyle="--",
        color="red",
    )
    axs[1, 1].set_ylabel("New Deaths")
    axs[1, 1].set_xlabel("Date")

    for ax in axs.flat:
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y")
        ax.legend()

    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{filename}.pdf")
    plt.savefig(f"../../reports/figures/{filename}.png")
    plt.show()

    return future_forecast_scaled


all_forecast_metrics = []

days_forecasts = [7, 14, 21, 28]
for days in days_forecasts:
    val_data = data[-days:]
    future_forecast_scaled = plot_forecast_and_validation(
        model, parameter_net, train_data, val_data, scaler, device, days, f"{days}_days_forecast_validation"
    )

    val_I, val_H, val_C, val_D = prepare_tensors(val_data, device)
    val_I_pred, val_H_pred, val_C_pred, val_D_pred = (
        future_forecast_scaled[:, 0],
        future_forecast_scaled[:, 1],
        future_forecast_scaled[:, 2],
        future_forecast_scaled[:, 3],
    )

    metrics_days = []

    for obs, pred, train, area in zip(
        [
            val_I.cpu().numpy(),
            val_H.cpu().numpy(),
            val_C.cpu().numpy(),
            val_D.cpu().numpy(),
        ],
        [val_I_pred, val_H_pred, val_C_pred, val_D_pred],
        train_data_values,
        areas,
    ):
        nrmse, mase, mae, mape = calculate_errors(obs, pred, train)
        print(f"Metrics for {area} ({days} days):")
        print(f"Normalized Root Mean Square Error (NRMSE): {nrmse:.4f}")
        print(f"Mean Absolute Scaled Error (MASE): {mase:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")

        metrics_days.append([f"{area}_NRMSE", nrmse])
        metrics_days.append([f"{area}_MASE", mase])
        metrics_days.append([f"{area}_MAE", mae])
        metrics_days.append([f"{area}_MAPE", mape])

        all_forecast_metrics.append([f"{days}_days_{area}_NRMSE", nrmse])
        all_forecast_metrics.append([f"{days}_days_{area}_MASE", mase])
        all_forecast_metrics.append([f"{days}_days_{area}_MAE", mae])
        all_forecast_metrics.append([f"{days}_days_{area}_MAPE", mape])

    save_metrics(metrics_days, f"England_{days}_days")

# Save all forecast metrics to a single DataFrame
all_forecast_metrics_df = pd.DataFrame(all_forecast_metrics, columns=["Metric", "Value"])
all_forecast_metrics_df.to_csv(f"../../reports/results/England_all_forecast_metrics.csv", index=False)
print(f"All forecast metrics saved to ../../reports/results/England_all_forecast_metrics.csv")
