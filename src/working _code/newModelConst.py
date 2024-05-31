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
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Ensure the folders exist
os.makedirs("../../models", exist_ok=True)
os.makedirs("../../reports/figures", exist_ok=True)
os.makedirs("../../reports/results", exist_ok=True)

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

# Define the SEIRD model differential equations
def seird_model(y, t, N, beta, alpha, rho, ds, da, omega, dH, mu, gamma_c, delta_c, eta):
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


def load_preprocess_data(filepath, areaname, rolling_window=7, end_date=None):
    """Load and preprocess the COVID-19 data."""
    df = pd.read_csv(filepath)
    
    # Select the columns of interest
    df = df[df["nhs_region"] == areaname].reset_index(drop=True)
    
    # reset the index
    df = df[::-1].reset_index(drop=True)  # Reverse dataset if needed
    
    # Convert the date column to datetime
    df["date"] = pd.to_datetime(df["date"])
    
    # select data up to the end_date
    if end_date:
        df = df[df["date"] <= end_date]
        
    # calculate the susceptible column from data
    df["susceptible"] = df["population"] - df["cumulative_confirmed"] - df["cumulative_deceased"]
    
    cols_to_smooth = ["susceptible", "cumulative_confirmed", "cumulative_deceased", "hospitalCases", "covidOccupiedMVBeds", "new_deceased", "new_confirmed", "newAdmissions", "cumAdmissions"]
    for col in cols_to_smooth:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)
        
    # normalize the data with the population
    df["cumulative_confirmed"] = df["cumulative_confirmed"] / df["population"]
    df["cumulative_deceased"] = df["cumulative_deceased"] / df["population"]
    df["hospitalCases"] = df["hospitalCases"] / df["population"]
    df["covidOccupiedMVBeds"] = df["covidOccupiedMVBeds"] / df["population"]
    df["new_deceased"] = df["new_deceased"] / df["population"]
    df["new_confirmed"] = df["new_confirmed"] / df["population"]
    df["population"] = df["population"] / df["population"]
    df["susceptible"] = df["susceptible"] / df["population"]
    df["newAdmissions"] = df["newAdmissions"] / df["population"]
    df["cumAdmissions"] = df["cumAdmissions"] / df["population"]
    
    return df

data = load_preprocess_data("../../data/processed/merged_data.csv", "London", rolling_window=7, end_date="2020-08-31")

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
    t = tensor(np.arange(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
    I = tensor(data["new_confirmed"].values, dtype=torch.float32).view(-1, 1).to(device)
    H = tensor(data["newAdmissions"].values, dtype=torch.float32).view(-1, 1).to(device)
    C = tensor(data["covidOccupiedMVBeds"].values, dtype=torch.float32).view(-1, 1).to(device)
    D = tensor(data["new_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
    return t, I, H, C, D

def split_and_scale_data(data, train_size, features, device):
    """Split and scale data into training and validation sets."""
    scaler = MinMaxScaler()
    scaler.fit(data[features])

    train_data = data.iloc[:train_size]
    val_data = data.iloc[train_size:]

    scaled_train_data = pd.DataFrame(scaler.transform(train_data[features]), columns=features)
    scaled_val_data = pd.DataFrame(scaler.transform(val_data[features]), columns=features)

    t_train, I_train, H_train, C_train, D_train = prepare_tensors(scaled_train_data, device)
    t_val, I_val, H_val, C_val, D_val = prepare_tensors(scaled_val_data, device)

    tensor_data = {
        "train": (t_train, I_train, H_train, C_train, D_train),
        "val": (t_val, I_val, H_val, C_val, D_val),
    }
    
    return tensor_data, scaler

features = ["new_confirmed", "newAdmissions", "covidOccupiedMVBeds", "new_deceased"]

# set the train size in days
train_size = 60

tensor_data, scaler = split_and_scale_data(data, train_size, features, device)

# Define the neural network model
class StateNN(nn.Module):
    def __init__(self, inverse=False, init_beta=None, init_omega=None, init_mu=None, init_gamma=None, init_delta=None, init_eta=None, retrain_seed=42, num_layers=4, hidden_neurons=20):
        super(StateNN, self).__init__()
        self.retrain_seed = retrain_seed
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 8))
        self.net = nn.Sequential(*layers)

        if inverse:
            self._beta = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32).to(device), requires_grad=True)
            self._omega = nn.Parameter(torch.tensor(init_omega, dtype=torch.float32).to(device), requires_grad=True)
            self._mu = nn.Parameter(torch.tensor(init_mu, dtype=torch.float32).to(device), requires_grad=True)
            self._gamma_c = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32).to(device), requires_grad=True)
            self._delta_c = nn.Parameter(torch.tensor(init_delta, dtype=torch.float32).to(device), requires_grad=True)
            self._eta = nn.Parameter(torch.tensor(init_eta, dtype=torch.float32).to(device), requires_grad=True)
        else:
            self._beta = None
            self._omega = None
            self._mu = None
            self._gamma_c = None
            self._delta_c = None
            self._eta = None

        self.init_xavier()

    def forward(self, t):
        return torch.sigmoid(self.net(t))

    @property
    def beta(self):
        return torch.sigmoid(self._beta)

    @property
    def omega(self):
        return torch.sigmoid(self._omega)

    @property
    def delta(self):
        return torch.sigmoid(self._mu)
    
    @property
    def gamma_c(self):
        return torch.sigmoid(self._gamma_c)
    
    @property
    def delta_c(self):
        return torch.sigmoid(self._delta_c)
    
    @property
    def eta(self):
        return torch.sigmoid(self._eta)
    

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        self.apply(init_weights)

# Update the loss function to include eta
def einn_loss(model_output, tensor_data, parameters, t, train_size):
    """Compute the loss function for the EINN model."""
    
    # Split the model output into the different compartments
    S_pred, E_pred, Ia_pred, Is_pred, H_pred, C_pred, R_pred, D_pred = torch.split(model_output, 1, dim=1)
    
    # Extract training and validation data from tensor_data
    Is_train, H_train, C_train, D_train = tensor_data["train"]
    Is_val, H_val, C_val, D_val = tensor_data["val"]
    
    # Normalize the data
    N = 1.0
    
    # Combine training and validation data for total data
    Is_total = torch.cat([Is_train, Is_val], dim=0)
    H_total = torch.cat([H_train, H_val], dim=0)
    C_total = torch.cat([C_train, C_val], dim=0)
    D_total = torch.cat([D_train, D_val], dim=0)
    
    E_total = Is_total[0]
    Ia_total = 0.0
    R_total = 0.0
    S_total = N - E_total - Ia_total - Is_total - H_total - C_total - R_total - D_total
    
    # Constants based on the table provided
    rho = 0.75  # Proportion of symptomatic infections
    alpha = 1 / 5.2  # Incubation period (3.4 days)
    d_s = 1 / 2.9  # Infectious period for symptomatic (2.9 days)
    d_a = 1 / 6  # Infectious period for asymptomatic (6 days)
    d_h = 1 / 7  # Hospitalization days (7 days)
    
    # learned parameters
    beta = parameters.beta
    omega = parameters.omega
    mu = parameters.delta
    gamma_c = parameters.gamma_c
    delta_c = parameters.delta_c
    eta = parameters.eta
    
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
    dSdt, dEdt, dIadt, dIsdt, dHdt, dCdt, dRdt, dDdt = seird_model(
        [S_pred, E_pred, Is_pred, Ia_pred, H_pred, C_pred, R_pred, D_pred],
        t, N, beta, alpha, rho, d_s, d_a, omega, d_h, mu, gamma_c, delta_c, eta
    )
    
    if train_size is not None:
        index = torch.randperm(train_size)
    else:
        index = torch.arange(len(t))
        
    # Ensure indices are within bounds
    index = index[index < len(S_total)]
    
    # Compute the loss function
    # data loss
    data_loss = (
        torch.mean((S_pred[index] - S_total[index]) ** 2)
        + torch.mean((E_pred[index] - E_total[index]) ** 2)
        + torch.mean((Is_pred[index] - Is_total[index]) ** 2)
        + torch.mean((Ia_pred[index] - Ia_total[index]) ** 2)
        + torch.mean((H_pred[index] - H_total[index]) ** 2)
        + torch.mean((C_pred[index] - C_total[index]) ** 2)
        + torch.mean((D_pred[index] - D_total[index]) ** 2)
        + torch.mean((R_pred[index] - R_total[index]) ** 2)
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
    
    # Initial condition loss
    S0, E0, Ia0, Is0, H0, C0, R0, D0 = S_total[0], E_total[0], Ia_total, Is_total[0], H_total[0], C_total[0], R_total, D_total[0]
    initial_loss = (
        (S_pred[0] - S0) ** 2
        + (E_pred[0] - E0) ** 2
        + (Ia_pred[0] - Ia0) ** 2
        + (Is_pred[0] - Is0) ** 2
        + (H_pred[0] - H0) ** 2
        + (C_pred[0] - C0) ** 2
        + (R_pred[0] - R0) ** 2
        + (D_pred[0] - D0) ** 2
    )
    
    # total loss
    loss = data_loss + residual_loss + initial_loss
    
    return loss


# early stopping
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0
        self.loss_history = deque(maxlen=patience + 1)
        self.best_model = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.best_model = model.state_dict()

# Network prediction         
def network_prediction(t, model, device, scaler, N):
    """Generate predictions from the SEIRDNet model."""
    t_tensor = torch.from_numpy(t).float().view(-1, 1).to(device).requires_grad_(True)

    with torch.no_grad():
        predictions = model(t_tensor).cpu().numpy()
        predictions = scaler.inverse_transform(predictions)

    return predictions

model = StateNN(
    inverse=True,
    init_beta=0.1,
    init_omega=0.01,
    init_mu=0.01,
    init_gamma=0.01,
    init_delta=0.01,
    init_eta=0.01,
    retrain_seed=seed,
    num_layers=5,
    hidden_neurons=32
).to(device)
            
N = data["population"].iloc[0]
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=10000, gamma=0.9)
earlystopping = EarlyStopping(patience=100, verbose=False)
num_epochs = 100000

t = torch.tensor(np.arange(1, len(data) + 1), dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)

index = torch.randperm(len(tensor_data["train"][0]))

loss_history = []

# Train the model
def train_model(model, optimizer, scheduler, earlystopping, num_epochs, t, tensor_data, index, loss_history):
    """Train the model."""
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        model_output = model(t)
        
        # Compute the loss function
        loss = einn_loss(model_output, tensor_data, model, t, len(index))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Record the loss
        loss_history.append(loss.item())
        
        # Early stopping
        earlystopping(loss.item(), model)
        if earlystopping.early_stop:
            print("Early stopping")
            break
        
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
            
    return loss_history, model


loss_history, model = train_model(model, optimizer, scheduler, earlystopping, num_epochs, t, tensor_data, index, loss_history)

# plot the loss history
def plot_loss(losses, title):
    plt.plot(np.arange(1, len(losses) + 1), losses, label='Loss', color='black')
    plt.yscale('log')
    plt.title(f"{title} loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.legend()
    plt.show()
    
plot_loss(loss_history, "EINN")

