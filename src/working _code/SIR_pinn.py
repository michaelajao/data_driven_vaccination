import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tqdm.autonotebook import tqdm

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the default style
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {   "font.family": "serif",
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

# def load_and_preprocess_data(filepath, recovery_period=21, rolling_window=7, start_date="2020-04-01"):
#     df = pd.read_csv(filepath)
#     required_columns = ["date", "cumulative_confirmed", "cumulative_deceased", "population", "new_confirmed", "new_deceased"]
#     for col in required_columns:
#         if col not in df.columns:
#             raise ValueError(f"Missing required column: {col}")

#     df["date"] = pd.to_datetime(df["date"])
#     df["days_since_start"] = (df["date"] - pd.to_datetime(start_date)).dt.days
#     for col in ["new_confirmed", "new_deceased", "cumulative_confirmed", "cumulative_deceased"]:
#         df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)
    
#     df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
#     df["recovered"] = df["recovered"].fillna(0).clip(lower=0)
#     df["active_cases"] = df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]
#     df["S(t)"] = df["population"] - df["active_cases"] - df["recovered"] - df["cumulative_deceased"]
#     df = df[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
#     df[["recovered", "active_cases", "S(t)", "new_confirmed", "new_deceased", "cumulative_confirmed", "cumulative_deceased"]] = df[["recovered", "active_cases", "S(t)", "new_confirmed", "new_deceased", "cumulative_confirmed", "cumulative_deceased"]].clip(lower=0)
    
#     return df

df = pd.read_csv("../../data/processed/england_data.csv")

def load_and_preprocess_data(filepath, recovery_period=21, rolling_window=7, start_date="2020-04-01", end_date="2021-08-31"):
    
    df = pd.read_csv(filepath)
    
    df["date"] = pd.to_datetime(df["date"])
    # Compute daily new values from cumulative values
    df['daily_confirmed'] = df["cumulative_confirmed"].diff().fillna(0)
    df['daily_deceased'] = df["cumulative_deceased"].diff().fillna(0)
    df['daily_hospitalized'] = df["cumAdmissions"].diff().fillna(0)
    
    # ensure no negative values
    df['daily_confirmed'] = df['daily_confirmed'].clip(lower=0)
    df['daily_deceased'] = df['daily_deceased'].clip(lower=0)
    df['daily_hospitalized'] = df['daily_hospitalized'].clip(lower=0)
    
    required_columns = ["date","population", "cumulative_confirmed", "cumulative_deceased", "new_confirmed", "new_deceased", "cumAdmissions", "daily_confirmed", "daily_deceased", "daily_hospitalized", "hospitalCases", "cumAdmissions", "covidOccupiedMVBeds", "newAdmissions"]
    
    # select required columns
    df = df[required_columns]
    
    # 7 day rolling average to smooth out data except for date and population
    for col in required_columns[2:]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)
        
    # select data from start date to end date
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df.loc[mask]
    
    return df


def get_region_name_from_filepath(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]

df = load_and_preprocess_data("../../data/processed/england_data.csv")

# plot recovered cases
plt.plot(df["date"], df["daily_confirmed"])
plt.title("Daily Confirmed Cases")
plt.xlabel("Date")


# start_date = "2021-01-01"
# end_date = "2021-08-31"
# mask = (df["date"] >= start_date) & (df["date"] <= end_date)
# training_data = df.loc[mask]

transformer = MinMaxScaler()
columns_to_scale = ["daily_confirmed", "daily_deceased"]
transformer.fit(df[columns_to_scale])
df[columns_to_scale] = transformer.transform(df[columns_to_scale])

N = df["population"].values[0]

# Convert columns to tensors
t_data = torch.tensor(range(len(df)), dtype=torch.float32).view(-1, 1).requires_grad_(True).to(device)
I_data = torch.tensor(df["daily_confirmed"].values, dtype=torch.float32).view(-1, 1).to(device)
D_data = torch.tensor(df["daily_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
SIR_tensor = torch.cat([I_data, D_data], dim=1).to(device)

class ParamNet(nn.Module):
    def __init__(self, output_size=2, num_layers=3, hidden_neurons=20):
        super(ParamNet, self).__init__()
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)

    def get_params(self, t):
        raw_params = self.forward(t)
        beta = torch.sigmoid(raw_params[:, 0]) * 0.5  # Adjusted scaling
        delta = torch.sigmoid(raw_params[:, 1]) * 0.1  # Adjusted scaling
        return beta, delta

    def init_xavier(self):
        torch.manual_seed(42)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        self.apply(init_weights)

class SIRNet(nn.Module):
    def __init__(self, num_layers=4, hidden_neurons=20):  # Increased layers
        super(SIRNet, self).__init__()
        self.retrain_seed = 42
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 5))
        self.net = nn.Sequential(*layers)
        self.init_xavier()

    def forward(self, t):
        return torch.sigmoid(self.net(t))

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)
        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        self.apply(init_weights)

def compute_seird_derivatives(S, E, I, R, D, beta, delta, N):
    gamma , alpha = 1/5.1, 0.02
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * D
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * D
    return dSdt, dEdt, dIdt, dRdt, dDdt

def enhanced_sir_loss(SIR_tensor, model_output, beta_pred, delta_pred, t_tensor, N):
    S_pred, E_pred, I_pred, R_pred, D_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2], model_output[:, 3], model_output[:, 4]
    
    # initial conditions from data
    N = N / N
    
    E_0 = 0.0
    I_0 = SIR_tensor[:, 0][0]
    D_0 = SIR_tensor[:, 1][0]
    R_0 = 0.0
    
    S_0 = N - E_0 - I_0 - D_0 - R_0
        

    S_t = torch.autograd.grad(S_pred, t_tensor, torch.ones_like(S_pred), create_graph=True)[0]
    E_t = torch.autograd.grad(E_pred, t_tensor, torch.ones_like(E_pred), create_graph=True)[0]
    I_t = torch.autograd.grad(I_pred, t_tensor, torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t_tensor, torch.ones_like(R_pred), create_graph=True)[0]
    D_t = torch.autograd.grad(D_pred, t_tensor, torch.ones_like(D_pred), create_graph=True)[0]

    dSdt_pred, dEdt_pred, dIdt_pred, dRdt_pred, dDdt_pred = compute_seird_derivatives(S_pred, E_pred, I_pred, R_pred, D_pred, beta_pred, delta_pred, N)

    fitting_loss = torch.mean((SIR_tensor[:, 0] - I_pred) ** 2) + torch.mean((SIR_tensor[:, 1] - D_pred) ** 2) + torch.mean((E_0 - E_pred[0]) ** 2) + torch.mean((I_0 - I_pred[0]) ** 2) + torch.mean((R_0 - R_pred[0]) ** 2) + torch.mean((D_0 - D_pred[0]) ** 2) + torch.mean((S_0 - S_pred[0]) ** 2)
    derivative_loss = torch.mean((S_t - dSdt_pred) ** 2) + torch.mean((E_t - dEdt_pred) ** 2) + torch.mean((I_t - dIdt_pred) ** 2) + torch.mean((R_t - dRdt_pred) ** 2) + torch.mean((D_t - dDdt_pred) ** 2)
    
    return fitting_loss + derivative_loss 

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

def train_models(param_model, sir_model, t_data, SIR_tensor, epochs, lr, N):
    optimiser = optim.Adam(list(param_model.parameters()) + list(sir_model.parameters()), lr=lr)
    early_stopping = EarlyStopping(patience=50, verbose=False)
    losses = []

    for epoch in tqdm(range(epochs)):
        param_model.train()
        sir_model.train()

        beta_pred, gamma_pred = param_model.get_params(t_data)
        sir_output = sir_model(t_data)
        loss = enhanced_sir_loss(SIR_tensor, sir_output, beta_pred, gamma_pred, t_data, N)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.6f}")
        
        if early_stopping(loss):
            print("Early stopping triggered.")
            break

    return losses

def plot_SIR_results_subplots(t_data, SIR_tensor, sir_model, title):
    with torch.no_grad():
        sir_output = sir_model(t_data)
        
    # t_np = t_data.cpu().detach().numpy().flatten()
    dates = df["date"]
    I_pred, D_pred = sir_output[:, 2].cpu().numpy(), sir_output[:, 4].cpu().numpy()
    I_actual, D_actual = SIR_tensor[:, 0].cpu().numpy(), SIR_tensor[:, 1].cpu().numpy()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    
    axs[0].plot(dates, I_pred, label="$Infected$ (predicted)", color="red")
    axs[0].plot(dates, I_actual, label="$Infected$ (Actual)", color="red", linestyle="--")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Proportion of Population")
    axs[0].legend()
    
    axs[1].plot(dates, D_pred, label="$Recovered$ (predicted)", color="green")
    axs[1].plot(dates, D_actual, label="$Recovered$ (Actual)", color="green", linestyle="--")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel("Proportion of Population")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_param_results_subplots(t_data, param_model, title, N, model_output=None):
    with torch.no_grad():
        beta_pred, delta_pred = param_model.get_params(t_data)
        alpha = 0.02
        S_t = model_output[:, 0]
        N = N / N

    # t_np = t_data.cpu().detach().numpy().flatten()
    dates = df["date"]
    beta_pred, delta_pred = beta_pred.cpu().numpy(), delta_pred.cpu().numpy()
    
    # R_t of the SEIRD model, R_t = (beta / (gamma + alpha)) * (S_t / N)
    R_t = (beta_pred / (1/5.1 + alpha)) * (S_t / N)
    

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    axs[0].plot(dates, beta_pred, label="Beta", color="purple")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Rate")
    axs[0].legend()

    axs[1].plot(dates, delta_pred, label="Gamma", color="orange")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel("Rate")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    
    plt.plot(dates, R_t, label="$R_t$", color="blue")
    plt.axhline(y=1, color='r', linestyle='--', label="$R_t=1$")
    plt.xlabel("Days")
    plt.ylabel("$R_t$")
    plt.legend()
    plt.title(f'{title} - Effective Reproduction Number $R_t$')
    plt.tight_layout()
    plt.show()

def plot_loss(losses, title):
    plt.grid(True, which="both", ls=":")
    plt.plot(np.arange(1, len(losses) + 1), np.log10(losses), label="Loss")
    plt.title(f"{title} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Log10 Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_E_pred(t, model, title):
    model.eval()
    with torch.no_grad():
        predictions = model(t).cpu().numpy()

    # t_np = t.cpu().detach().numpy().flatten()
    dates = df["date"]
    S_pred = predictions[:, 0]
    E_pred = predictions[:, 1]
    R_pred = predictions[:, 3]
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    
    axs[0].plot(dates, S_pred, label="S(t)", color="blue")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Proportion of Population")
    axs[0].legend()
    
    axs[1].plot(dates, E_pred, label="E(t)", color="orange")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel("Proportion of Population")
    axs[1].legend()
    
    axs[2].plot(dates, R_pred, label="R(t)", color="green")
    axs[2].set_xlabel("Days")
    axs[2].set_ylabel("Proportion of Population")
    axs[2].legend()
    
    
    plt.tight_layout()
    plt.show()

# Initialize the models
param_model = ParamNet(output_size=2, num_layers=3, hidden_neurons=20).to(device)
sir_model = SIRNet(num_layers=4, hidden_neurons=20).to(device)

# Train the models and collect losses
losses = train_models(param_model, sir_model, t_data, SIR_tensor, epochs=50000, lr=2e-4, N=N)

# Plot the results
plot_SIR_results_subplots(t_data, SIR_tensor, sir_model, "SIR Model Predictions")
plot_param_results_subplots(t_data, param_model, "SIR", N, model_output=sir_model(t_data))
plot_loss(losses, "SIR")
plot_E_pred(t_data, sir_model, "SIR")