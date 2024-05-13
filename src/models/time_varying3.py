
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.integrate import solve_ivp

from sklearn.preprocessing import MinMaxScaler
from tqdm.notebook import tqdm

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set the random seed for reproducibility
plt.style.use("seaborn-v0_8-poster")
plt.rcParams.update(
    {
        # Font settings for clarity and compatibility with academic publications
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 14,  # Base font size for better readability
        "text.usetex": False,  # Enable LaTeX for text rendering for a professional look
        # Figure aesthetics & size for detailed visuals and fit on publication pages
        "figure.figsize": [8, 4],  # Default figure size
        "figure.facecolor": "white",  # White figure background for clean print
        "figure.autolayout": True,  # Automatic layout adjustments
        "figure.dpi": 400,  # High resolution figures for publication quality
        "savefig.dpi": 400,  # High resolution saving settings
        "savefig.format": "pdf",  # Save figures in PDF format for publications
        "savefig.bbox": "tight",  # Tight bounding box around figures
        # title, xlabel and ylabel bold
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        # Axes aesthetics for clarity and precision
        "axes.labelsize": 12,
        "axes.titlesize": 18,  # Prominent titles for immediate recognition
        "axes.facecolor": "white",  # White axes background
        "axes.grid": True,  # Enable grid for better readability
        "axes.spines.top": False,  # Disable top spine for aesthetic purposes
        "axes.spines.right": False,  # Disable right spine for aesthetic purposes
        "axes.formatter.limits": (0, 5),  # Threshold for scientific notation
        "axes.formatter.use_mathtext": True,  # Use mathtext for scientific notations
        "axes.formatter.useoffset": False,  # Disable offset on axes
        "axes.xmargin": 0,  # No margin around x-axis
        "axes.ymargin": 0,  # No margin around y-axis
        # Legend aesthetics
        "legend.fontsize": 12,
        "legend.frameon": False,  # No frame around legend for cleaner look
        "legend.loc": "best",  # Optimal legend positioning
        # Line aesthetics
        "lines.linewidth": 2,  # Thicker lines for visibility
        "lines.markersize": 8,  # Slightly smaller markers for balance
        # Tick aesthetics
        "xtick.labelsize": 12,
        "xtick.direction": "in",  # Ticks inside the plot
        "xtick.top": False,  # Disable top ticks for aesthetic purposes
        "ytick.labelsize": 12,
        "ytick.direction": "in",  # Ticks inside the plot
        "ytick.right": False,  # Disable right ticks for aesthetic purposes
        # Grid settings
        "grid.color": "grey",  # Grid color
        "grid.linestyle": "--",  # Dashed grid lines
        "grid.linewidth": 0.5,  # Thin grid lines
        # Error bar aesthetics
        "errorbar.capsize": 4,  # Error bar cap length
        # Layout settings
        "figure.subplot.wspace": 0.4,  # Adjust horizontal spacing between subplots
        "figure.subplot.hspace": 0.4,  # Adjust vertical spacing between subplots
        # Latex and color map settings
        "image.cmap": "viridis",  # Preferred color map for images
        "text.latex.preamble": r"\usepackage{amsmath}",  # Latex preamble for math expressions
    }
)

# Device setup for CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_pytorch():
    # Print PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Print CUDA version
        print(f"CUDA version: {torch.version.cuda}")
        
        # List available GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA not available. PyTorch will run on CPU.")
        
check_pytorch()

# N is the total population
def SIHCRD_model(t, y, beta, gamma, delta, rho, eta, kappa, mu, xi, N):
    """
    Define the SIHCRD model as a system of differential equations.
    
    Parameters:
    - t: Time step (days).
    - y: State variables [S, I, H, C, R, D].
    - beta: Infection rate.
    - gamma: Recovery rate.
    - delta: Mortality rate.
    - rho: Hospitalization rate.
    - eta: Critical rate.
    - kappa: Recovery rate from hospitalization.
    - mu: Recovery rate from critical condition.
    - xi: Mortality rate from critical condition.
    - N: Total population.
    
    Returns:
    - A list of the rates of change for each state variable.
    
    
    """
    S, I, H, C, R, D = y
    dSdt = -(beta * I / N) * S
    dIdt = (beta * S / N) * I - (gamma + rho + delta) * I
    dHdt = rho * I - (eta + kappa) * H
    dCdt = eta * H - (mu + xi) * C
    dRdt = gamma * I + kappa * H + mu * C
    dDdt = delta * I + xi * C
    return [dSdt, dIdt, dHdt, dCdt, dRdt, dDdt]


# Define the initial conditions
def initial_conditions(N, I0, H0, C0, R0, D0):
    """
    Define the initial conditions for the SIHCRD model.

    Parameters:
    - N: Total population.
    - I0: Initial number of infected individuals.
    - H0: Initial number of hospitalized individuals.
    - C0: Initial number of critical individuals.
    - R0: Initial number of recovered individuals.
    - D0: Initial number of deceased individuals.

    Returns:
    - A list of the initial conditions for each state variable.
    """
    S0 = N - I0 - H0 - C0 - R0 - D0
    return [S0, I0, H0, C0, R0, D0]


# Define the time span
t0 = 0
tf = 360

# Define the parameters
N = 1e6
I0 = 100
H0 = 10
C0 = 5
R0 = 1
D0 = 1
beta = 0.3
gamma = 0.1
delta = 0.01
rho = 0.05
eta = 0.05
kappa = 0.05
mu = 0.05
xi = 0.01



# Define the initial conditions
y0 = initial_conditions(N, I0, H0, C0, R0, D0)

# Solve the system of differential equations
sol = solve_ivp(
    SIHCRD_model,
    [t0, tf],
    y0,
    args=(beta, gamma, delta, rho, eta, kappa, mu, xi, N),
    dense_output=True,
)

# Extract the solution
t = np.linspace(t0, tf, 1000)
y = sol.sol(t)

# Plot the solution
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, y[1], label="Infected", color="tab:blue")
ax.plot(t, y[2], label="Hospitalized", color="tab:orange")
ax.plot(t, y[3], label="Critical", color="tab:red")
ax.plot(t, y[4], label="Recovered", color="tab:green")
ax.plot(t, y[5], label="Deceased", color="tab:purple")
ax.plot(t, y[0], label="Susceptible", color="tab:cyan")  # Add susceptible population
ax.set_xlabel("Time (days)")
ax.set_ylabel("Number of individuals")
ax.set_title("SIHCRD Model")
ax.legend()
plt.show()

# Load the data
df = pd.read_csv("../../data/processed/england_data.csv").drop(
    columns=["Unnamed: 0"], axis=1
)

df.head()

# Data preprocessing
def load_and_preprocess_data(
    filepath, recovery_period=16, rolling_window=7, start_date="2020-04-01"
):
    """
    Load and preprocess the COVID-19 dataset for the SIHCRD model.

    Parameters:
    - filepath: Path to the CSV file containing the data.
    - recovery_period: Assumed number of days for recovery. Defaults to 16 days.
    - rolling_window: Window size for rolling average calculation. Defaults to 7 days.
    - start_date: The start date for filtering the data. Format 'YYYY-MM-DD'.

    Returns:
    - A preprocessed pandas DataFrame suitable for SIHCRD model integration.
    """
    df = pd.read_csv(filepath)

    # Ensure the dataset has the required columns
    required_columns = [
        "date",
        "cumulative_confirmed",
        "cumulative_deceased",
        "population",
        "covidOccupiedMVBeds",
        "hospitalCases",
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert date to datetime and calculate days since start
    df["date"] = pd.to_datetime(df["date"])
    df["days_since_start"] = (df["date"] - pd.to_datetime(start_date)).dt.days

    # Calculate recovered cases assuming a fixed recovery period
    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df[
        "cumulative_deceased"
    ].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)

    # Calculate active cases
    df["active_cases"] = (
        df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]
    )
    
    # Apply rolling average
    for col in [
        "cumulative_confirmed",
        "cumulative_deceased",
        "hospitalCases",
        "covidOccupiedMVBeds",
        "recovered",
        "active_cases",
        "new_deceased",
    ]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)

    # Filter data based on the start date
    df = df[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)

    # Ensure no negative values
    df[["recovered", "active_cases"]] = df[["recovered", "active_cases"]].clip(lower=0)

    return df

# Load and preprocess the data
data = load_and_preprocess_data(
    "../../data/processed/england_data.csv",
    recovery_period=16,
    start_date="2020-04-01",
).drop(columns=["Unnamed: 0"], axis=1)


# Split the data into training and validation sets
train_data_start = "2020-05-01"
train_data_end = "2020-12-31"

val_data_start = "2021-01-01"
val_data_end = "2021-04-30"

t_mask = (data["date"] >= train_data_start) & (data["date"] <= train_data_end)
train_data = data.loc[t_mask]

v_mask = (data["date"] >= val_data_start) & (data["date"] <= val_data_end)
val_data = data.loc[v_mask]

# Display the training and validation data
def prepare_tensors(data, device):
    # t should be the length of the data starting from 1
    t = (
        torch.tensor(range(1, len(data) + 1), dtype=torch.float32)
        .view(-1, 1)
        .to(device)
        .requires_grad_(True)
    )
    # S = torch.tensor(data["S(t)"].values, dtype=torch.float32).view(-1, 1).to(device)
    I = (
        torch.tensor(data["active_cases"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    R = (
        torch.tensor(data["recovered"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    D = (
        torch.tensor(data["new_deceased"].values, dtype=torch.float32)
        .view(-1, 1)
        .to(device)
    )
    return t, I, R, D


features = [
    "active_cases",
    "recovered",
    "new_deceased",
]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data[features])

scaled_train_data = pd.DataFrame(
    scaler.transform(train_data[features]), columns=features
)
scaled_val_data = pd.DataFrame(scaler.transform(val_data[features]), columns=features)


t_train, I_train, R_train, D_train = prepare_tensors(
    scaled_train_data, device
)

t_val, I_val, R_val, D_val = prepare_tensors(scaled_val_data, device)

train_tensor_data = torch.cat([I_train, R_train, D_train], dim=1)
val_tensor_data = torch.cat([I_val, R_val, D_val], dim=1)


# plot I_train for the period available
plt.plot(I_train.cpu().detach().numpy(), label="Active Cases")
plt.title("Active Cases in England")
plt.xlabel("Days since start")
plt.ylabel("Active Cases")
plt.legend()
plt.show()


# plot the training data and validation data and show the split as a labeled straight dotted line
plt.plot(I_train.cpu().detach().numpy(), label="Training Data")
plt.plot(
    range(len(I_train), len(I_train) + len(I_val)),
    I_val.cpu().detach().numpy(),
    label="Validation Data",
)
plt.axvline(x=len(I_train), color="black", linestyle="--", label="Split")
plt.title("Training and Validation Data Split")
plt.xlabel("Days since start")
plt.ylabel("Active Cases")
plt.legend()
plt.show()


# define the neural network for epi-net
class EpiNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10, output_size=5):
        super(EpiNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Initialize layers array starting with input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        # Append hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])

        # Append output layer
        layers.append(nn.Linear(hidden_neurons, output_size))  # Epidemiological outputs

        # Convert list of layers to nn.Sequential
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

        self.net.apply(init_weights)

# %%
# define the neural network for beta parameter estimation
class BetaNet(nn.Module):
    def __init__(self, num_layers=2, hidden_neurons=10):
        super(BetaNet, self).__init__()
        self.retain_seed = 100
        torch.manual_seed(self.retain_seed)

        # Initialize layers array starting with the input layer
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]

        # Append hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])

        # Append output layer
        layers.append(
            nn.Linear(hidden_neurons, 8)
        )  # Output layer for estimating infection rate β

        # Convert list of layers to nn.Sequential
        self.net = nn.Sequential(*layers)

        # Initialize weights
        self.init_xavier()


    def init_xavier(self):
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_normal_(layer.weight, gain=g)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)
                    
        self.net.apply(init_weights)
        
        # time varying parameters estimation using uniform distribution ranges and sigmoid function
    def forward(self, t):
        
        params = self.net(t)
        # beta (β) is a positive value between 0 and 1 using the sigmoid function
        beta = torch.sigmoid(params[:, 0]) * 0.9 + 0.1
        
        # gamma (γ) is a positive value between 0 and 0.1 using the sigmoid function
        gamma = torch.sigmoid(params[:, 1]) * 0.1
        
        # delta (δ) is a positive value between 0 and 0.01 using the sigmoid function
        delta = torch.sigmoid(params[:, 2]) * 0.01
        
        # rho (ρ) is a positive value between 0 and 0.05 using the sigmoid function
        rho = torch.sigmoid(params[:, 3]) * 0.05
        
        # eta (η) is a positive value between 0 and 0.05 using the sigmoid function
        eta = torch.sigmoid(params[:, 4]) * 0.05
        
        # kappa (κ) is a positive value between 0 and 0.05 using the sigmoid function
        kappa = torch.sigmoid(params[:, 5]) * 0.05
        
        # mu (μ) is a positive value between 0 and 0.05 using the sigmoid function
        mu = torch.sigmoid(params[:, 6]) * 0.05
        
        # xi (ξ) is a positive value between 0 and 0.01 using the sigmoid function
        xi = torch.sigmoid(params[:, 7]) * 0.01
        
        return params
    
# %%
# define the neural network for time varying parameters estimation using relu activation
class TimeVaryingNet(nn.Module):
    def __init__(self, num_layers=1, hidden_neurons=10, output_size=8):
        super(TimeVaryingNet, self).__init__()

        # Initialize layers array starting with the input layer
        layers = [nn.Linear(1, hidden_neurons), nn.LeakyReLU()]

        # Append hidden layers
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.LeakyReLU()])

        # Append output layer
        layers.append(nn.Linear(hidden_neurons, output_size))

        # Convert list of layers to nn.Sequential
        self.net = nn.Sequential(*layers)
        
        # Initialize weights and perform time varying parameters estimation
        # self.init_glorot()
        
    def forward(self, t):
        output = self.net(t)
        return output.get_params(output)
    
    # def init_glorot(self):
    #     def init_weights(layer):
    #         if isinstance(layer, nn.Linear):
    #             nn.init.xavier_normal_(layer.weight)
    #             if layer.bias is not None:
    #                 layer.bias.data.fill_(0)
                    
    #     self.net.apply(init_weights)
    
    # extract the time varying parameters
    def get_params(self, output):
        
        # beta (β) is a positive value between 0 and 1 using the sigmoid function
        beta = torch.sigmoid(output[:, 0]) * 0.9 + 0.1
        
        # gamma (γ) is a positive value between 0 and 0.1 using the sigmoid function
        gamma = torch.sigmoid(output[:, 1]) * 0.1
        
        # delta (δ) is a positive value between 0 and 0.01 using the sigmoid function
        delta = torch.sigmoid(output[:, 2]) * 0.01
        
        # rho (ρ) is a positive value between 0 and 0.05 using the sigmoid function
        rho = torch.sigmoid(output[:, 3]) * 0.05
        
        # eta (η) is a positive value between 0 and 0.05 using the sigmoid function
        eta = torch.sigmoid(output[:, 4]) * 0.05
        
        # kappa (κ) is a positive value between 0 and 0.05 using the sigmoid function
        kappa = torch.sigmoid(output[:, 5]) * 0.05
        
        # mu (μ) is a positive value between 0 and 0.05 using the sigmoid function
        mu = torch.sigmoid(output[:, 6]) * 0.05
        
        # xi (ξ) is a positive value between 0 and 0.01 using the sigmoid function
        xi = torch.sigmoid(output[:, 7]) * 0.01
        
        return beta, gamma, delta, rho, eta, kappa, mu, xi

# %%

def pinn_loss(tensor_data, parameters, model_output, t, N, device):
    I, R, D = tensor_data.unbind(1)

    # Calculate the predicted derivatives
    # S = N - I.sum() - R.sum() - D.sum()
    # H = N - S - I - R - D
    # C = N - S - I - H - R - D
    # S_pred = -beta_pred * S * I / N
    # H_pred = alpha * I - (gamma + delta) * H
    # C_pred = gamma * H - delta * C

    # Using grad outputs need to be of size [batch_size], not [batch_size, 1]
    # S = S.view(-1)
    # H = H.view(-1)
    # C = C.view(-1)
    
    beta_pred = parameters[:, 0].squeeze()
    gamma_pred = parameters[:, 1].squeeze()
    delta_pred = parameters[:, 2].squeeze()
    rho_pred = parameters[:, 3].squeeze()
    eta_pred = parameters[:, 4].squeeze()
    kappa_pred = parameters[:, 5].squeeze()
    mu_pred = parameters[:, 6].squeeze()
    xi_pred = parameters[:, 7].squeeze()
    
    
    
    S_pred, I_pred, H_pred, C_pred, R_pred, D_pred = model_output.unbind(1)
    # S_pred, I_pred, H_pred, C_pred, R_pred, D_pred = (
    #     S_pred.view(-1),
    #     I_pred.view(-1),
    #     H_pred.view(-1),
    #     C_pred.view(-1),
    #     R_pred.view(-1),
    #     D_pred.view(-1),
    # )
    # Compute gradients
    s_t = torch.autograd.grad(outputs=S_pred, inputs=t, grad_outputs=torch.ones_like(S_pred), create_graph=True)[0]
    i_t = torch.autograd.grad(outputs=I_pred, inputs=t, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    h_t = torch.autograd.grad(outputs=H_pred, inputs=t, grad_outputs=torch.ones_like(H_pred), create_graph=True)[0]
    c_t = torch.autograd.grad(outputs=C_pred, inputs=t, grad_outputs=torch.ones_like(C_pred), create_graph=True)[0]
    r_t = torch.autograd.grad(outputs=R_pred, inputs=t, grad_outputs=torch.ones_like(R_pred), create_graph=True)[0]
    d_t = torch.autograd.grad(outputs=D_pred, inputs=t, grad_outputs=torch.ones_like(D_pred), create_graph=True)[0]
    

    
    # Compute the predicted derivatives based on the model
    dSdt = -beta_pred * S_pred * I_pred / N
    dIdt = beta_pred * S_pred * I_pred / N - (gamma_pred + rho_pred + delta_pred) * I_pred
    dHdt = rho_pred * I_pred - (eta_pred + kappa_pred) * H_pred
    dCdt = eta_pred * H_pred - (mu_pred + xi_pred) * C_pred
    dRdt = gamma_pred * I_pred + kappa_pred * H_pred + mu_pred * C_pred
    dDdt = delta_pred * I_pred + xi_pred * C_pred
    
    # Loss components
    # Data loss
    data_loss = torch.mean((I - I_pred) ** 2 + (R - R_pred) ** 2 + (D - D_pred) ** 2)
    
    # physics loss
    physics_loss = torch.mean((s_t - dSdt) ** 2 + (i_t - dIdt) ** 2 + (h_t - dHdt) ** 2 + (c_t - dCdt) ** 2 + (r_t - dRdt) ** 2 + (d_t - dDdt) ** 2)
    
    # initial condition loss
    initial_condition_loss = torch.mean((I[0] - I_pred[0]) ** 2 + (R[0] - R_pred[0]) ** 2 + (D[0] - D_pred[0]) ** 2)
    
    # Total loss
    loss = data_loss + physics_loss + initial_condition_loss
    return loss

# %%
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
            
# Define the model and beta net
model = EpiNet(num_layers=10, hidden_neurons=32, output_size=6).to(device)
# The above code is creating an instance of a neural network model called `BetaNet` with 2 hidden
# layers and 32 neurons in each hidden layer. The model is then moved to the specified device (e.g.,
# GPU or CPU) for computation.
beta_net = BetaNet(num_layers=10, hidden_neurons=32).to(device)

# beta_net = TimeVaryingNet(num_layers=5, hidden_neurons=32, output_size=8).to(device)

# %%
def train_model(
    model,
    beta_net,
    tensor_data,
    t_train,
    N,
    lr,
    num_epochs=1000,
    device=device,
    print_every=100,
    verbose=True,
):
    # Define the optimizers
    model_optimizer = optim.Adam(model.parameters(), lr=lr)
    params_optimizer = optim.Adam(beta_net.parameters(), lr=lr)
    
    # Define the learning rate scheduler
    model_scheduler = StepLR(model_optimizer, step_size=5000, gamma=0.6)
    params_scheduler = StepLR(params_optimizer, step_size=5000, gamma=0.6)
    
    # Initialize the early stopping object
    early_stopping = EarlyStopping(patience=100, verbose=verbose)
    
    # Initialize the loss history
    loss_history = []
    
    # Train the model
    for epoch in tqdm(range(num_epochs)):
        model_optimizer.zero_grad()
        params_optimizer.zero_grad()
        
        # Forward pass to compute the predicted values on 30 time points of the training data
        model_output = model(t_train)
        
        # Forward pass to compute the predicted parameters
        parameters = beta_net(t_train)
        
        # Compute the loss
        loss = pinn_loss(tensor_data, parameters, model_output, t_train, N, device)
    
        
        # Backward pass
        loss.backward()
        
        # Update the parameters
        model_optimizer.step()
        params_optimizer.step()
        
        
        # Append the loss to the loss history
        loss_history.append(loss.item())
        
        # Print the loss every `print_every` epochs without using the `verbose` flag
        if epoch % print_every == 0:
            print(f"Epoch {epoch} - Loss: {loss.item()}")
            
        # Check for early stopping
        early_stopping(loss.item())
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        # Update the learning rate
        model_scheduler.step(loss)
        params_scheduler.step(loss)
        
        
    return model_output, parameters, loss_history

N = data["population"].values[0]

# Train the model with 100 data points
model_output, parameters, loss_history = train_model(
    model,
    beta_net,
    train_tensor_data,
    t_train,
    N,
    lr=1e-5,
    num_epochs=50000,
    device=device,
    print_every=500,
    verbose=False,
)

# plot the loss history in base 10
plt.plot(np.log10(loss_history))
plt.title("Log Loss History")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.show()

# plot the loss history
plt.plot(loss_history)
plt.title("Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# plot the actual and predicted of the training data for active cases 
model_output = model(t_train)
plt.plot(I_train.cpu().detach().numpy(), label="Actual Active Cases")
plt.plot(model_output[:, 1].cpu().detach().numpy(), label="Predicted Active Cases")
plt.title("Actual vs Predicted Active Cases")
plt.xlabel("Days since start")
plt.ylabel("Active Cases")
plt.legend()
plt.show()

# plot the actual and predicted of the training data for active cases using the dates
def plot_training_results(t_train, actual_data, model_output):
    """
    Plot the actual and predicted infected data from the training set.

    Parameters:
    - t_train: Tensor of time points (typically the training dataset's time indices).
    - actual_data: Actual infected data from the training set.
    - model_output: Output from the trained model, expected to contain predicted values.
    """
    # Convert tensors to numpy arrays for plotting
    t_train_np = t_train.cpu().detach().numpy().flatten()
    actual_data_np = actual_data.cpu().detach().numpy().flatten()
    predicted_data_np = model_output.cpu().detach().numpy().flatten()  # Assuming the first column is the infected prediction

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(t_train_np, actual_data_np, label='Actual Infected', color='blue')
    plt.plot(t_train_np, predicted_data_np, label='Predicted Infected', color='red', linestyle='--')
    plt.title('Comparison of Actual and Predicted Infected Cases')
    plt.xlabel('Time (days since start)')
    plt.ylabel('Number of Infected Individuals')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_training_results(t_train, I_train, model_output[:, 1])

# %%
# extract the time varying beta values
beta_values = beta_net(t_train).cpu().detach().numpy()

# plot the beta values
plt.plot(beta_values)
plt.title("Time Varying Beta Values")
plt.xlabel("Days since start")
plt.ylabel("Beta")
plt.show()



# %%
# plot the actual and predicted of the training data for hospital cases
plt.plot(H_train.cpu().detach().numpy(), label="Actual Hospital Cases")
plt.plot(H_pred.cpu().detach().numpy(), label="Predicted Hospital Cases")
plt.title("Actual vs Predicted Hospital Cases")
plt.xlabel("Days since start")
plt.ylabel("Hospital Cases")
plt.legend()
plt.show()


# %%
# plot the actual and predicted of the training data for critical cases and the validation data actual and predicted for critical cases
plt.plot(C_train.cpu().detach().numpy(), label="Training Data")
plt.plot(C_pred.cpu().detach().numpy(), label="Predicted Training Data")
plt.plot(range(len(C_train), len(C_train) + len(C_val)), C_val.cpu().detach().numpy(), label="Validation Data")
plt.title("Training and Validation Data for Critical Cases")
plt.xlabel("Days since start")
plt.ylabel("Critical Cases")
plt.legend()
plt.show()

# predict with 50 time points from the validation data
model.eval()
with torch.no_grad():
    t = t_val[:50]
    I_pred, H_pred, C_pred, R_pred, D_pred = model(t).unbind(1)
    
    # plot the actual and predicted of the validation data
    plt.plot(C_val.cpu().detach().numpy()[:50], label="Actual Critical Cases")
    plt.plot(C_pred.cpu().detach().numpy(), label="Predicted Critical Cases")
    plt.title("Actual vs Predicted Critical Cases")
    plt.xlabel("Days since start")
    plt.ylabel("Critical Cases")
    plt.legend()
    plt.show()

# %%
# using the trained model, predict and test on the validation data using the ODE model
model.eval()

# Define the initial conditions as the last point in the trained output
I0 = I_pred[-1].item()
H0 = H_pred[-1].item()
C0 = C_pred[-1].item()
R0 = R_pred[-1].item()
D0 = D_pred[-1].item()

# Define the initial conditions
y0 = initial_conditions(N, I0, H0, C0, R0, D0)

# extract the last value of beta
beta = beta_values[-1]

# Solve the system of differential equations to predict for the t_val
sol = solve_ivp(
    SIHCRD_model,
    [0, len(t_val)],
    y0,
    args=(beta, 0.1, 0.01, 0.05, N),
    dense_output=True,
)

# Extract the solution
t = np.linspace(0, len(t_val), 1000)
y = sol.sol(t)

# Plot the solution
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(t, y[1], label="Infected", color="tab:blue")
ax.plot(t, y[2], label="Hospitalized", color="tab:orange")
ax.plot(t, y[3], label="Critical", color="tab:red")
ax.plot(t, y[4], label="Recovered", color="tab:green")
ax.plot(t, y[5], label="Deceased", color="tab:purple")
ax.plot(t, y[0], label="Susceptible", color="tab:cyan")  # Add susceptible population
ax.set_xlabel("Time (days)")
ax.set_ylabel("Number of individuals")
ax.set_title("SIHCRD Model")
ax.legend()
plt.show()


# %%
# predicting on the validation data
model.eval()
with torch.no_grad():
    t = t_val
    I_pred, H_pred, C_pred, R_pred, D_pred = model(t).unbind(1)
    
    # plot the actual and predicted of the validation data
    plt.plot(I_val.cpu().detach().numpy(), label="Actual Active Cases", color="tab:orange")
    plt.plot(I_pred.cpu().detach().numpy(), label="Predicted Active Cases", color="tab:blue")
    
    plt.title("Actual vs Predicted Active Cases")
    plt.xlabel("Days since start")
    plt.ylabel("Active Cases")
    plt.legend()
    plt.show()    

# %%
beta_value = beta_net(t).cpu().detach().numpy() 


S0 = N - I_train[0] - H_train[0] - C_train[0] - R_train[0] - D_train[0]  
y0 = [S0.item(), I_train[0].item(), H_train[0].item(), C_train[0].item(), R_train[0].item(), D_train[0].item()]

# %%
# Ensure t is handled correctly if it's a tensor
start_time = t_val[0].cpu().item()  # Get the start time as a float
end_time = t_val[-1].cpu().item()  # Get the end time as a float

# Simulate forward in time using solve_ivp
sol = solve_ivp(
    lambda t, y: SIHCRD_model(t, y, beta_value, gamma, delta, alpha, N),
    `
)

# %%
# training loop using 30 time point of the training data and simulating forward in time
model.eval()
with torch.no_grad():
    t = t_train[:30]
    I_pred, H_pred, C_pred, R_pred, D_pred = model(t).unbind(1)
    
    # define the initial conditions
    S0 = N - I0 - H0 - C0 - R0 - D0
    y0 = [S0, I0, H0, C0, R0, D0]
    
    # simulate forward in time
    sol = solve_ivp(
        SIHCRD_model,
        [t[0].item(), t[-1].item()],
        y0,
        args=(beta_net(t), 0.1, 0.01, 0.05, N),
        dense_output=True,
    )
    
    # extract the solution
    t = np.linspace(t[0].item(), t[-1].item(), 1000)
    y = sol.sol(t)
    
    # plot the actual and predicted of the training data
    plt.plot(I_train.cpu().detach().numpy()[:30], label="Actual Active Cases")
    plt.plot(I_pred.cpu().detach().numpy(), label="Predicted Active Cases")
    plt.plot(t, y[1], label="Simulated Active Cases")
    plt.title("Actual vs Predicted Active Cases")
    plt.xlabel("Days since start")
    plt.ylabel("Active Cases")
    plt.legend()
    plt.show()

# %%


# %%
# # Filter train and validation datasets
# train_data = data[(data['date'] >= train_data_start) & (data['date'] <= train_data_end)]
# val_data = data[(data['date'] >= val_data_start) & (data['date'] <= val_data_end)]

# # Define features for scaling
# features = ["active_cases", "hospitalCases", "covidOccupiedMVBeds", "recovered", "new_deceased"]

# # Apply MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(train_data[features])
# scaled_train_data = pd.DataFrame(scaler.transform(train_data[features]), columns=features)
# scaled_val_data = pd.DataFrame(scaler.transform(val_data[features]), columns=features)

# %%
# def prepare_tensors(data, device):
#     # Ensure the time tensor `t` requires gradients for autograd operations
#     t = torch.arange(1, len(data) + 1, dtype=torch.float32, requires_grad=True).view(-1, 1).to(device)
#     tensors = [torch.tensor(data[feature].values, dtype=torch.float32).view(-1, 1).to(device) for feature in features]
#     return [t] + tensors


# %%
# # Prepare tensors for training and validation data
# train_tensors = prepare_tensors(scaled_train_data, device)
# val_tensors = prepare_tensors(scaled_val_data, device)

# # Extract the tensors
# t_train, I_train, R_train, D_train, H_train, C_train = train_tensors
# t_val, I_val, R_val, D_val, H_val, C_val = val_tensors

# train_tensor = torch.cat([I_train, H_train, C_train, R_train, D_train], dim=1)
# val_tensor = torch.cat([I_val, H_val, C_val, R_val, D_val], dim=1)

# %%


# %%



