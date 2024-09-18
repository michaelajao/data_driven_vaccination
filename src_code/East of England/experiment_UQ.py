# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import scienceplots
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
plt.style.use(["science", "ieee", "no-latex"])
plt.rcParams.update({
    # ... (same as your previous style settings)
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

area_name = "East of England"

def load_preprocess_data(filepath, area_name, recovery_period=16, rolling_window=7, start_date="2020-04-01", end_date=None):
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

    # Apply 7-day rolling average to smooth out data
    for col in required_columns[2:]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1, center=False).mean().fillna(0)

    # Select data from start date to end date
    mask = (df["date"] >= start_date)
    if end_date:
        mask &= (df["date"] <= end_date)
    df = df.loc[mask].reset_index(drop=True)
    
    return df

# Load and preprocess data
data = load_preprocess_data(
    "../../data/processed/merged_nhs_covid_data.csv",
    area_name,
    recovery_period=21,
    rolling_window=7,
    start_date="2020-05-01",
    end_date="2021-12-31",
)

# Plotting new deceased cases over time
plt.plot(data["date"], data["daily_deceased"])
plt.title("New Daily Deceased over time")
plt.xlabel("Date")
plt.ylabel("New Deceased")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Split data into training and validation sets, the validation should be the last 7 days
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

features = [
    "daily_confirmed",
    "daily_hospitalized",
    "covidOccupiedMVBeds",
    "daily_deceased",
]

# Split and scale the data
train_data_scaled, val_data_scaled, scaler = scale_data(train_data, val_data, features, device)

# Define the ResidualBlock with Dropout for UQ
class ResidualBlock(nn.Module):
    def __init__(self, hidden_neurons, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_neurons, hidden_neurons),
            nn.Tanh(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_neurons, hidden_neurons),
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(x + self.layer(x))

# Define the EpiNet model with Dropout for UQ
class EpiNet(nn.Module):
    def __init__(self, num_layers=3, hidden_neurons=10, output_size=8, dropout_rate=0.1):
        super(EpiNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        
        # Hidden layers with residual connections
        for _ in range(num_layers):
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(ResidualBlock(hidden_neurons, dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)
        
        # Initialize parameters with constraints
        self._rho = nn.Parameter(torch.tensor([0.8], device=device), requires_grad=True)
        self._alpha = nn.Parameter(torch.tensor([0.2], device=device), requires_grad=True)
        self._ds = nn.Parameter(torch.tensor([0.1], device=device), requires_grad=True)
        self._da = nn.Parameter(torch.tensor([0.1], device=device), requires_grad=True)
        self._dH = nn.Parameter(torch.tensor([0.05], device=device), requires_grad=True)

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
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)

        # Apply the weight initialization to the network
        self.net.apply(init_weights)

# Define the ParameterNet
class ParameterNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=6, dropout_rate=0.1):
        super(ParameterNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_neurons, hidden_neurons),
                nn.Tanh()
            ])

        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)

        self.init_xavier()

    def forward(self, t):
        raw_parameters = self.net(t)

        # Apply sigmoid and scale to specific ranges
        beta = 0.1 + 0.9 * torch.sigmoid(raw_parameters[:, 0])    # Range [0.1, 1.0]
        gamma_c = 0.0 + 0.5 * torch.sigmoid(raw_parameters[:, 1]) # Range [0.0, 0.5]
        delta_c = 0.0 + 0.5 * torch.sigmoid(raw_parameters[:, 2]) # Range [0.0, 0.5]
        eta = 0.0 + 0.2 * torch.sigmoid(raw_parameters[:, 3])     # Range [0.0, 0.2]
        mu = 0.0 + 0.1 * torch.sigmoid(raw_parameters[:, 4])      # Range [0.0, 0.1]
        omega = 0.0 + 0.5 * torch.sigmoid(raw_parameters[:, 5])   # Range [0.0, 0.5]

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

    Is_data, H_data, C_data, D_data = tensor_data[:, 0], tensor_data[:, 1], tensor_data[:, 2], tensor_data[:, 3]

    N = 1  # Since data is normalized

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
    
    # total loss
    loss = data_loss + residual_loss + initial_cost
    return loss

class EarlyStopping:
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

def train_model(model, parameter_net, optimizer, scheduler, time_stamps, data_scaled, num_epochs=5000, early_stopping=None):
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

        if (epoch + 1) % 500 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}")

        # Check early stopping
        if early_stopping:
            early_stopping(train_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return train_losses

# Initialize model, optimizer, and scheduler
model = EpiNet(num_layers=6, hidden_neurons=10, output_size=8, dropout_rate=0.1).to(device)
parameter_net = ParameterNet(num_layers=5, hidden_neurons=10, output_size=6, dropout_rate=0.1).to(device)
optimizer = optim.Adam(
    list(model.parameters()) + list(parameter_net.parameters()), lr=1e-4
)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, verbose=True)

# Early stopping
early_stopping = EarlyStopping(patience=50, verbose=True)

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
    train_data_scaled,
    num_epochs=10000,
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

# Function to perform predictions with uncertainty using Monte Carlo Dropout
def predict_with_uncertainty(model, parameter_net, t, n_samples=100):
    model.train()  # Keep dropout layers active
    parameter_net.train()

    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            model_output = model(t)
            predictions.append(model_output.cpu().numpy())

    predictions = np.array(predictions)  # Shape: (n_samples, batch_size, output_size)
    return predictions

# Test the model on the full data set with uncertainty quantification
n_samples = 100  # Number of Monte Carlo samples
time_stamps_full = (
    torch.tensor(data.index.values, dtype=torch.float32)
    .view(-1, 1)
    .to(device)
    .requires_grad_()
)

# Obtain predictions with uncertainty
predictions = predict_with_uncertainty(model, parameter_net, time_stamps_full, n_samples)

# Calculate mean and standard deviation
mean_predictions = predictions.mean(axis=0)
std_predictions = predictions.std(axis=0)

# Convert model output to DataFrame and inverse transform to original scale
observed_model_output = pd.DataFrame(
    {
        "daily_confirmed": mean_predictions[:, 2],
        "daily_hospitalized": mean_predictions[:, 4],
        "covidOccupiedMVBeds": mean_predictions[:, 5],
        "daily_deceased": mean_predictions[:, 7],
    },
    index=data.index,
)

# Inverse transform the scaled data
observed_model_output_scaled = scaler.inverse_transform(observed_model_output)
observed_model_output_scaled = pd.DataFrame(
    observed_model_output_scaled,
    columns=["daily_confirmed", "daily_hospitalized", "covidOccupiedMVBeds", "daily_deceased"],
    index=data.index,
)

# Standard deviations for the observed variables
observed_std = pd.DataFrame(
    {
        "daily_confirmed_std": std_predictions[:, 2],
        "daily_hospitalized_std": std_predictions[:, 4],
        "covidOccupiedMVBeds_std": std_predictions[:, 5],
        "daily_deceased_std": std_predictions[:, 7],
    },
    index=data.index,
)

# Define unobserved model output
unobserved_model_output = pd.DataFrame(
    {
        "S": mean_predictions[:, 0],
        "E": mean_predictions[:, 1],
        "Ia": mean_predictions[:, 3],
        "R": mean_predictions[:, 6],
    },
    index=data.index,
)

# Plot Observed Data vs Predicted Data with Uncertainty
variables = ["daily_confirmed", "daily_hospitalized", "covidOccupiedMVBeds", "daily_deceased"]
variable_labels = [r"$I_s$", r"$H$", r"$C$", r"$D$"]
std_variables = ["daily_confirmed_std", "daily_hospitalized_std", "covidOccupiedMVBeds_std", "daily_deceased_std"]

fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

for i, (var, label, std_var) in enumerate(zip(variables, variable_labels, std_variables)):
    ax = axs[i]
    ax.plot(data["date"], data[var], label="Observed", color="blue")
    ax.plot(data["date"], observed_model_output_scaled[var], label="Predicted", color="red", linestyle="--")
    # Plot uncertainty intervals
    lower_bound = observed_model_output_scaled[var] - 2 * observed_std[std_var]
    upper_bound = observed_model_output_scaled[var] + 2 * observed_std[std_var]
    ax.fill_between(data["date"], lower_bound, upper_bound, color='red', alpha=0.3, label="Confidence Interval (±2σ)")
    ax.set_ylabel(label)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

fig.suptitle("Observed vs Predicted Data with Uncertainty", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Unobserved Data Visualization in 4x1 layout
fig, axs = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

def plot_unobserved_data(ax, unobserved_model_output, variable, ylabel):
    ax.plot(data["date"], unobserved_model_output[variable], label="Predicted", color="green")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.tick_params(axis='x', rotation=45)

plot_unobserved_data(axs[0], unobserved_model_output, "S", r"$S$")
plot_unobserved_data(axs[1], unobserved_model_output, "E", r"$E$")
plot_unobserved_data(axs[2], unobserved_model_output, "Ia", r"$I_a$")
plot_unobserved_data(axs[3], unobserved_model_output, "R", r"$R$")

fig.suptitle("Unobserved Data Predictions", fontsize=14, fontweight="bold")
plt.xlabel("Date")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Calculate R_t with uncertainty
beta_samples = []
mu_samples = []
ds = model.ds.detach()
for _ in range(n_samples):
    model.train()  # Keep dropout active
    parameter_net.train()
    with torch.no_grad():
        parameters_sample = parameter_net(time_stamps_full)
        beta_pred_sample = parameters_sample[0].cpu().numpy()
        mu_pred_sample = parameters_sample[4].cpu().numpy()
        beta_samples.append(beta_pred_sample)
        mu_samples.append(mu_pred_sample)

beta_samples = np.array(beta_samples)
mu_samples = np.array(mu_samples)

# Calculate R_t samples
Rt_samples = beta_samples / (ds.cpu().numpy() + mu_samples)

# Calculate mean and standard deviation
Rt_mean = Rt_samples.mean(axis=0)
Rt_std = Rt_samples.std(axis=0)

# Plot R_t with uncertainty
plt.figure(figsize=(10, 6))
plt.plot(data["date"], Rt_mean, label="$R_t$", color="orange")
plt.fill_between(
    data["date"],
    Rt_mean - 2 * Rt_std,
    Rt_mean + 2 * Rt_std,
    alpha=0.3,
    label="Confidence Interval (±2σ)",
    color="orange",
)
plt.ylabel("$R_t$")
plt.xlabel("Date")
plt.title("Effective Reproduction Number Over Time with Uncertainty")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Compute error metrics on validation data
print("\nValidation Metrics:")
for col in features:
    print(f"\nMetrics for {col}:")
    calculate_errors(
        val_data[col].values,
        observed_model_output_scaled.loc[val_data.index, col].values,
        train_data[col].values,
        train_size=len(train_data),
        area_name=area_name
    )
