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
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "text.usetex": False,
    "figure.figsize": (8, 5),
    "figure.facecolor": "white",
    "figure.autolayout": True,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "axes.labelsize": 14,
    "axes.titlesize": 20,
    "axes.facecolor": "white",
    "legend.fontsize": 12,
    "legend.frameon": False,
    "legend.loc": "best",
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

def load_and_preprocess_data(filepath, recovery_period=21, rolling_window=7, start_date="2020-04-01"):
    df = pd.read_csv(filepath)
    required_columns = ["date", "cumulative_confirmed", "cumulative_deceased", "population", "new_confirmed", "new_deceased"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["date"] = pd.to_datetime(df["date"])
    df["days_since_start"] = (df["date"] - pd.to_datetime(start_date)).dt.days
    for col in ["new_confirmed", "new_deceased", "cumulative_confirmed", "cumulative_deceased"]:
        df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0)
    
    df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
    df["recovered"] = df["recovered"].fillna(0).clip(lower=0)
    df["active_cases"] = df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]
    df["S(t)"] = df["population"] - df["active_cases"] - df["recovered"] - df["cumulative_deceased"]
    df = df[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)
    df[["recovered", "active_cases", "S(t)", "new_confirmed", "new_deceased", "cumulative_confirmed", "cumulative_deceased"]] = df[["recovered", "active_cases", "S(t)", "new_confirmed", "new_deceased", "cumulative_confirmed", "cumulative_deceased"]].clip(lower=0)
    
    return df

def get_region_name_from_filepath(filepath):
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]

df = load_and_preprocess_data("../../data/processed/england_data.csv")

# plot recovered cases
plt.plot(df["date"], df["recovered"])
plt.title("Recovered Cases")
plt.xlabel("Date")
plt.ylabel("Recovered Cases")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

start_date = "2021-01-01"
end_date = "2021-08-31"
mask = (df["date"] >= start_date) & (df["date"] <= end_date)
training_data = df.loc[mask]

transformer = MinMaxScaler()
columns_to_scale = ["S(t)", "active_cases", "recovered"]
transformer.fit(training_data[columns_to_scale])
training_data[columns_to_scale] = transformer.transform(training_data[columns_to_scale])

N = training_data["population"].values[0]

# Convert columns to tensors
S_data = torch.tensor(training_data["S(t)"].values, dtype=torch.float32).view(-1, 1).to(device)
t_data = torch.tensor(range(len(training_data)), dtype=torch.float32).view(-1, 1).requires_grad_(True).to(device)
I_data = torch.tensor(training_data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
R_data = torch.tensor(training_data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
SIR_tensor = torch.cat([S_data, I_data, R_data], dim=1).to(device)
        
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
        beta = torch.sigmoid(raw_params[:, 0]) 
        gamma = torch.sigmoid(raw_params[:, 1])
        return beta, gamma

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
    def __init__(self, num_layers=3, hidden_neurons=20):
        super(SIRNet, self).__init__()
        self.retrain_seed = 42
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()])
        layers.append(nn.Linear(hidden_neurons, 4))
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

def compute_sir_derivatives(S, E, I, R, beta, gamma, alpha=1/5, N=1):
    dSdt = -(beta * S * I) / N
    dEdt = (beta * S * I) / N - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def enhanced_sir_loss(SIR_tensor, model_output, beta_pred, gamma_pred, t_tensor, N):
    S_pred, E_pred, I_pred, R_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2], model_output[:, 3]
    S_actual, I_actual, R_actual = SIR_tensor[:, 0], SIR_tensor[:, 1], SIR_tensor[:, 2]

    S_t = torch.autograd.grad(S_pred, t_tensor, torch.ones_like(S_pred), create_graph=True)[0]
    E_t = torch.autograd.grad(E_pred, t_tensor, torch.ones_like(E_pred), create_graph=True)[0]
    I_t = torch.autograd.grad(I_pred, t_tensor, torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t_tensor, torch.ones_like(R_pred), create_graph=True)[0]

    dSdt_pred, dEdt_pred, dIdt_pred, dRdt_pred = compute_sir_derivatives(S_pred, E_pred, I_pred, R_pred, beta_pred, gamma_pred, N=N)

    fitting_loss = torch.mean((S_pred - S_actual) ** 2) + torch.mean((I_pred - I_actual) ** 2) + torch.mean((R_pred - R_actual) ** 2)
    derivative_loss = torch.mean((S_t - dSdt_pred) ** 2) + torch.mean((E_t - dEdt_pred) ** 2) + torch.mean((I_t - dIdt_pred) ** 2) + torch.mean((R_t - dRdt_pred) ** 2)
    
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
    early_stopping = EarlyStopping(patience=100, verbose=False)
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
        
    t_np = t_data.cpu().detach().numpy().flatten()
    S_pred, I_pred, R_pred = sir_output[:, 0].cpu().numpy(), sir_output[:, 2].cpu().numpy(), sir_output[:, 3].cpu().numpy()
    S_actual, I_actual, R_actual = SIR_tensor[:, 0].cpu().numpy(), SIR_tensor[:, 1].cpu().numpy(), SIR_tensor[:, 2].cpu().numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].plot(t_np, S_pred, label="$Susceptible$ (predicted)", color="blue")
    axs[0].plot(t_np, S_actual, label="$Susceptible$ (Actual)", color="blue", linestyle="--")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Proportion of Population")
    axs[0].legend()
    
    axs[1].plot(t_np, I_pred, label="$Infected$ (predicted)", color="red")
    axs[1].plot(t_np, I_actual, label="$Infected$ (Actual)", color="red", linestyle="--")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel("Proportion of Population")
    axs[1].legend()
    
    axs[2].plot(t_np, R_pred, label="$Recovered$ (predicted)", color="green")
    axs[2].plot(t_np, R_actual, label="$Recovered$ (Actual)", color="green", linestyle="--")
    axs[2].set_xlabel("Days")
    axs[2].set_ylabel("Proportion of Population")
    axs[2].legend()
    
    # plt.suptitle(f'{title} - SIR Model Predictions')
    plt.tight_layout()
    plt.show()

def plot_param_results_subplots(t_data, param_model, title):
    with torch.no_grad():
        beta_pred, gamma_pred = param_model.get_params(t_data)

    t_np = t_data.cpu().detach().numpy().flatten()
    beta_pred, gamma_pred = beta_pred.cpu().numpy(), gamma_pred.cpu().numpy()
    
    # R_t = beta_pred / gamma_pred
    R_t = beta_pred * (1 / gamma_pred)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    axs[0].plot(t_np, beta_pred, label="Beta", color="purple")
    axs[0].set_xlabel("Days")
    axs[0].set_ylabel("Rate")
    axs[0].legend()

    axs[1].plot(t_np, gamma_pred, label="Gamma", color="orange")
    axs[1].set_xlabel("Days")
    axs[1].set_ylabel("Rate")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    
    plt.plot(t_np, R_t, label="$R_t$", color="blue")
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

    t_np = t.cpu().detach().numpy().flatten()
    E_pred = predictions[:, 1]
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    
    axs.plot(t_np, E_pred, 'r-', label='$E_{pred}$')
    axs.set_title('E')
    axs.set_xlabel('Time t (days)')
    axs.legend()
    
    plt.tight_layout()
    plt.show()

    

# Initialize the models
param_model = ParamNet(output_size=2, num_layers=1, hidden_neurons=5).to(device)
sir_model = SIRNet(num_layers=4, hidden_neurons=20).to(device)

# Train the models and collect losses
losses = train_models(param_model, sir_model, t_data, SIR_tensor, epochs=50000, lr=2e-4, N=N)

# Plot the results
plot_SIR_results_subplots(t_data, SIR_tensor, sir_model, "SIR Model Predictions")
plot_param_results_subplots(t_data, param_model, "Parameter Dynamics")
plot_loss(losses, "SIR")
plot_E_pred(t_data, sir_model, "SIR")

