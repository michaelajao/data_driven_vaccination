# import os
# import torch
# from torch import nn, optim
# import lightning as pl
# from pytorch_lightning.callbacks import ModelCheckpoint
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from tqdm import tqdm


# # Set random seed for reproducibility
# pl.seed_everything(42)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# # select all avbailable GPUs
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# # Set the default style
# plt.style.use("seaborn-v0_8-white")
# plt.rcParams.update(
#     {
#         "lines.linewidth": 2,
#         "font.family": "serif",
#         "axes.titlesize": 20,
#         "axes.labelsize": 14,
#         "figure.figsize": [15, 8],
#         "figure.autolayout": True,
#         "axes.spines.top": False,
#         "axes.spines.right": False,
#         "axes.grid": True,
#         "grid.color": "0.75",
#         "legend.fontsize": "medium",
#     }
# )

# # Load the data
# path = "../../data/raw/pickle/covid19_data.pkl"
# data = pd.read_pickle(path)

# # select one of the region
# region = "North East England"
# data = data[data["region"] == region]

# # Convert the date to datetime
# data["date"] = pd.to_datetime(data["date"])

# min_date = data["date"].min()
# max_date = data["date"].max()

# data_range = max_date - min_date
# train_end = min_date + pd.Timedelta(days=data_range.days * 0.70)
# val_end = train_end + pd.Timedelta(days=data_range.days * 0.15)

# # Split the data into train, validation and test
# train = data[data['date'] < train_end]
# val = data[(data['date'] >= train_end) & (data['date'] < val_end)]
# test = data[data['date'] >= val_end]

# total_sample = len(data)
# train_percent = len(train) / total_sample * 100
# val_percent = len(val) / total_sample * 100
# test_percent = len(test) / total_sample * 100

# print(f"Train: {len(train)} samples ({train_percent:.2f}%)")
# print(f"Validation: {len(val)} samples ({val_percent:.2f}%)")
# print(f"Test: {len(test)} samples ({test_percent:.2f}%)")



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# runge-kutta method
from scipy.integrate import odeint, solve_ivp
import numpy as np
import matplotlib.pyplot as plt

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

# Set up matplotlib
plt.rcParams.update({
    "font.family": "serif",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.titlesize": 20,
    "axes.labelsize": 12,
    "figure.figsize": [20, 10],
    "figure.autolayout": True,
    "legend.fontsize": "medium",
    "legend.frameon": False,
    "legend.loc": "best",
    "lines.linewidth": 2.5,
    "lines.markersize": 10,
    "font.size": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    
})

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device setup for CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SIR Model Differential Equations
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# SIR Neural Network Model
class SIRNet(nn.Module):
    def __init__(self, inverse=False, init_beta=None, init_gamma=None, retrain_seed=42):
        super(SIRNet, self).__init__()
        self.retrain_seed = retrain_seed
        self.net = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 3)
        )

        # Adjustments for inverse model with customizable initial values
        if inverse:
            self._beta = nn.Parameter(torch.tensor([init_beta if init_beta is not None else torch.rand(1)], device=device))
            self._gamma = nn.Parameter(torch.tensor([init_gamma if init_gamma is not None else torch.rand(1)], device=device))
        else:
            self._beta = None
            self._gamma = None

        # Initialize the network weights
        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    @property
    def beta(self):
        return torch.sigmoid(self._beta) if self._beta is not None else None

    @property
    def gamma(self):
        return torch.sigmoid(self._gamma) if self._gamma is not None else None

    # Initialize the neural network with Xavier Initialization
    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(init_weights)


# Common loss function for both forward and inverse problems
def sir_loss(model, model_output, SIR_tensor, t_tensor, N, beta=None, gamma=None):
    S_pred, I_pred, R_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2]
    S_t = torch.autograd.grad(S_pred, t_tensor, torch.ones_like(S_pred), create_graph=True)[0]
    I_t = torch.autograd.grad(I_pred, t_tensor, torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t_tensor, torch.ones_like(R_pred), create_graph=True)[0]

    if beta is None:  # Use model's parameters for inverse problem
        beta, gamma = model.beta, model.gamma

    dSdt = -(beta * S_pred * I_pred) / N
    dIdt = (beta * S_pred * I_pred) / N - gamma * I_pred
    dRdt = gamma * I_pred

    loss = torch.mean((S_t - dSdt) ** 2) + torch.mean((I_t - dIdt) ** 2) + torch.mean((R_t - dRdt) ** 2)
    loss += torch.mean((model_output - SIR_tensor) ** 2)  # Data fitting loss
    return loss

# Early stopping class
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
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
            
# Training function
def train(model, t_tensor, SIR_tensor, epochs=1000, lr=0.001, N=None, beta=None, gamma=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5,)
    early_stopping = EarlyStopping(patience=5, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        model_output = model(t_tensor)
        loss = sir_loss(model, model_output, SIR_tensor, t_tensor, N, beta, gamma)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Check early stopping
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    print("Training finished")


# Data Preparation
N = 1000  # Total population
beta = 0.5  # Infection rate
gamma = 0.2  # Recovery rate
I0, R0 = 1, 0  # Initial number of infected and recovered individuals
S0 = N - I0  # Initial number of susceptible individuals

# Time points for observation in days
t = np.linspace(0, 50, 1000)  # Time grid

# Solve the SIR model using runge-kutta method
ret = odeint(deriv, [S0, I0, R0], t, args=(N, beta, gamma))

S, I, R = ret.T / N  # Normalizing the data

# Conversion to Tensors
t_tensor = torch.tensor(t, dtype=torch.float32, device=device).view(-1, 1)
S_tensor = torch.tensor(S, dtype=torch.float32, device=device).view(-1, 1)
I_tensor = torch.tensor(I, dtype=torch.float32, device=device).view(-1, 1)
R_tensor = torch.tensor(R, dtype=torch.float32, device=device).view(-1, 1)
SIR_tensor = torch.cat([S_tensor, I_tensor, R_tensor], dim=1)
t_tensor.requires_grad = True

# Train the forward problem
model_forward = SIRNet().to(device)
print("Training Forward Model")
train(model_forward, t_tensor, SIR_tensor, epochs=50000, lr=0.0001, N=N, beta=beta, gamma=gamma)

# Train the inverse problem
model_inverse = SIRNet(inverse=True, init_beta=0.1, init_gamma=0.1, retrain_seed=100).to(device)
print("\nTraining Inverse Model")
train(model_inverse, t_tensor, SIR_tensor, epochs=50000, lr=0.0001, N=N)

def plot_results(t, S, I, R, model, title):
    model.eval()
    with torch.no_grad():
        predictions = model(t_tensor).cpu().numpy()

    plt.subplot(1, 3, 1)
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, predictions[:, 0], label='Susceptible (predicted)', linestyle='dashed')
    plt.title('Susceptible')
    plt.xlabel('Time')
    plt.ylabel('Proportion of Population')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(t, I, label='Infected')
    plt.plot(t, predictions[:, 1], label='Infected (predicted)', linestyle='dashed')
    plt.title('Infected')
    plt.xlabel('Time')
    plt.ylabel('Proportion of Population')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(t, R, label='Recovered')
    plt.plot(t, predictions[:, 2], label='Recovered (predicted)', linestyle='dashed')
    plt.title('Recovered')
    plt.xlabel('Time')
    plt.ylabel('Proportion of Population')
    plt.legend()

 # Adjust the layout
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{title}.pdf")
    plt.show()

# Plot results for the forward model
plot_results(t, S, I, R, model_forward, "Forward Model Predictions")

# Plot results for the inverse model
plot_results(t, S, I, R, model_inverse, "Inverse Model Predictions")
