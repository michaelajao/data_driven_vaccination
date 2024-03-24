import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp, odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set up matplotlib
plt.rcParams.update({
    # Font settings for clarity and compatibility with academic publications
    "font.family": "serif",  # Consistent font family
    # "font.serif": ["Times", "Computer Modern Roman"],  # Preferred serif fonts
    "font.size": 14,  # Base font size for better readability
    "text.usetex": False,  # Enable LaTeX for text rendering for a professional look

    # Figure aesthetics & size for detailed visuals and fit on publication pages
    "figure.figsize": (8, 5),  # Adjusted figure size for a balance between detail and fit
    "figure.facecolor": "white",  # White figure background for clean print
    "figure.autolayout": True,  # Enable automatic layout adjustments
    "figure.dpi": 400,  # High resolution figures
    "savefig.dpi": 400,  # High resolution saving settings
    "savefig.format": "pdf",  # Save figures in PDF format for publications
    "savefig.bbox": "tight",  # Tight bounding box around figures

    # Axes aesthetics for clarity and precision
    "axes.labelsize": 14,  # Clear labeling with larger font size
    "axes.titlesize": 20,  # Prominent titles for immediate recognition
    "axes.facecolor": "white",  # White axes background

    # Legend aesthetics for distinguishing plot elements
    "legend.fontsize": 12,  # Readable legend font size
    "legend.frameon": False,  # No frame around legend for cleaner look
    "legend.loc": "best",  # Optimal legend positioning

    # Line aesthetics for clear visual distinctions
    "lines.linewidth": 2,  # Thicker lines for visibility
    "lines.markersize": 8,  # Slightly smaller markers for balance

    # Tick label sizes for readability
    "xtick.labelsize": 12, 
    "ytick.labelsize": 12,
    "xtick.direction": "in",  # Ticks inside the plot
    "ytick.direction": "in",  # Ticks inside the plot
})

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Device setup for CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_and_preprocess_data(filepath, recovery_period=21, rolling_window=7, start_date="2020-04-01"):
    """
    Load and preprocess COVID-19 dataset.

    Parameters:
    - filepath: Path to the CSV file containing the data.
    - recovery_period: Assumed number of days for recovery. Defaults to 21 days.
    - rolling_window: Window size for rolling average calculation. Defaults to 7 days.
    - start_date: The start date for filtering the data. Format 'YYYY-MM-DD'.

    Returns:
    - A preprocessed pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)

        required_columns = [
            "date", "cumulative_confirmed", "cumulative_deceased", 
            "population", "covidOccupiedMVBeds"
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        df["date"] = pd.to_datetime(df["date"])
        df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

        for col in ["cumulative_confirmed", "cumulative_deceased", "new_confirmed", "new_deceased", 
                    "covidOccupiedMVBeds", "hospitalCases", "cumAdmissions", "newAdmissions"]:
            df[col] = df[col].rolling(window=rolling_window, min_periods=1).mean().fillna(0).astype(int)

        df["recovered"] = df["cumulative_confirmed"].shift(recovery_period) - df["cumulative_deceased"].shift(recovery_period)
        df["recovered"] = df["recovered"].clip(lower=0)  # Ensure recovered cases do not go negative

        df["active_cases"] = df["cumulative_confirmed"] - df["recovered"] - df["cumulative_deceased"]
        df["S(t)"] = df["population"] - df["active_cases"] - df["recovered"] - df["cumulative_deceased"]
        df.fillna(0, inplace=True)

        df = df[df["date"] >= pd.to_datetime(start_date)]
        df = df.reset_index(drop=True)

        # Ensure no negative values in calculated columns
        for col in ["recovered", "active_cases", "S(t)"]:
            df[col] = df[col].clip(lower=0)

        return df
    except FileNotFoundError:
        print("File not found. Please check the filepath and try again.")
    except pd.errors.EmptyDataError:
        print("No data found in the file.")
    except ValueError as e:
        print(e)


def split_time_series_data(df, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Splits the DataFrame into training, validation, and test sets while maintaining the time series order.

    Args:
        df (pd.DataFrame): The input DataFrame with time series data.
        train_size (float): Proportion of the dataset to allocate to training.
        val_size (float): Proportion of the dataset to allocate to validation.
        test_size (float): Proportion of the dataset to allocate to testing.

    Returns:
        tuple: Three DataFrames corresponding to the training, validation, and test sets.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size, val_size, and test_size should sum to 1.")

    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    train_data = df.iloc[:train_end]
    val_data = df.iloc[train_end:val_end]
    test_data = df.iloc[val_end:]

    return train_data, val_data, test_data


data = load_and_preprocess_data("../../data/processed/england_data.csv", recovery_period=21, rolling_window=7, start_date="2020-04-01")

# Standardize the data
data["cumulative_confirmed"] = data["cumulative_confirmed"] / data["population"]
data["cumulative_deceased"] = data["cumulative_deceased"] / data["population"]
data["covidOccupiedMVBeds"] = data["covidOccupiedMVBeds"] / data["population"]
data["active_cases"] = data["active_cases"] / data["population"]
data["hospitalCases"] = data["hospitalCases"] / data["population"]
data["recovered"] = data["recovered"] / data["population"]

# split data
train_data, val_data, test_data = split_time_series_data(data, train_size=0.7, val_size=0.15, test_size=0.15)


train_data = train_data[["days_since_start", "cumulative_confirmed", "cumulative_deceased", "covidOccupiedMVBeds", "hospitalCases", "recovered", "active_cases", "S(t)"]]

plt.figure(figsize=(12, 6))
plt.plot(train_data["days_since_start"], train_data["cumulative_confirmed"], label="Confirmed Cases")

plt.xlabel("Days Since Start")
plt.ylabel("Proportion of Population")
plt.title("Confirmed and Deceased Cases Over Time")
plt.legend()
plt.show()

t_train = torch.tensor(train_data["days_since_start"].values, dtype=torch.float32).view(-1, 1).to(device).requires_grad_(True)
S_train = torch.tensor(train_data["S(t)"].values, dtype=torch.float32).view(-1, 1).to(device)
# E_train = torch.tensor(train_data["cumulative_confirmed"].values, dtype=torch.float32).view(-1, 1).to(device)
I_train = torch.tensor(train_data["active_cases"].values, dtype=torch.float32).view(-1, 1).to(device)
R_train = torch.tensor(train_data["recovered"].values, dtype=torch.float32).view(-1, 1).to(device)
D_train = torch.tensor(train_data["cumulative_deceased"].values, dtype=torch.float32).view(-1, 1).to(device)
H_train = torch.tensor(train_data["hospitalCases"].values, dtype=torch.float32).view(-1, 1).to(device)
SEIRDH_train = torch.cat([S_train, I_train, R_train, D_train, H_train], dim=1).to(device)


# SAIRDH Neural Network Model
class SAIRDHNet(nn.Module):
    def __init__(self, num_layers=4, hidden_neurons=20, retrain_seed=42):
        super(SAIRDHNet, self).__init__()
        self.retrain_seed = retrain_seed
        layers = [nn.Linear(1, hidden_neurons), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_neurons, hidden_neurons), nn.Tanh()]
        # Adjust the output layer to have 6 outputs corresponding to S, A, I, R, D, H
        layers.append(nn.Linear(hidden_neurons, 6))
        self.net = nn.Sequential(*layers)

        # Parameters for the SAIRDH model; initializing within a range using the sigmoid function
        self.params = nn.Parameter(torch.rand(6, device=device), requires_grad=True) 

        # Initialize the network weights
        self.init_xavier()

    def forward(self, t):
        return self.net(t)

    # Properties for the parameters to ensure they are within a realistic range
    @property
    def beta(self):
        # Transmission rate
        return torch.sigmoid(self.params[0]) * 0.9 + 0.1

    @property
    def sigma(self):
        # Rate of transition from exposed to infectious
        return torch.sigmoid(self.params[1]) * 0.09 + 0.01

    @property
    def rho(self):
        # Hospitalization rate
        return torch.sigmoid(self.params[2]) * 0.09 + 0.01

    @property
    def delta(self):
        # Mortality rate outside the hospital
        return torch.sigmoid(self.params[3]) * 0.09 + 0.01

    @property
    def eta(self):
        # Rate of becoming critical from hospitalized
        return torch.sigmoid(self.params[4]) * 0.09 + 0.01

    @property
    def theta(self):
        # Mortality rate in the hospital
        return torch.sigmoid(self.params[5]) * 0.09 + 0.01

    def init_xavier(self):
        torch.manual_seed(self.retrain_seed)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                g = nn.init.calculate_gain("tanh")
                nn.init.xavier_uniform_(m.weight, gain=g)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)  # Small positive bias initialization

        self.apply(init_weights)

                
                
# Define the loss function
def sairdh_loss(model, model_output, SAIRDH_tensor, t_tensor, N, params=None):
    S_pred, A_pred, I_pred, R_pred, D_pred, H_pred = model_output[:, 0], model_output[:, 1], model_output[:, 2], model_output[:, 3], model_output[:, 4], model_output[:, 5]
    
    # Derivatives of the compartments with respect to time
    S_t = torch.autograd.grad(S_pred, t_tensor, torch.ones_like(S_pred), create_graph=True)[0]
    A_t = torch.autograd.grad(A_pred, t_tensor, torch.ones_like(A_pred), create_graph=True)[0]
    I_t = torch.autograd.grad(I_pred, t_tensor, torch.ones_like(I_pred), create_graph=True)[0]
    R_t = torch.autograd.grad(R_pred, t_tensor, torch.ones_like(R_pred), create_graph=True)[0]
    D_t = torch.autograd.grad(D_pred, t_tensor, torch.ones_like(D_pred), create_graph=True)[0]
    H_t = torch.autograd.grad(H_pred, t_tensor, torch.ones_like(H_pred), create_graph=True)[0]

    # Parameters for the model
    if params is None:  # Use model's parameters for inverse problem
        beta, sigma, rho, delta, eta, theta, gamma = model.beta, model.sigma, model.rho, model.delta, model.eta, model.theta, model.gamma
    else:
        beta, sigma, rho, delta, eta, theta, gamma = params
    
    # Differential equations
    dSdt = -(beta * S_pred * I_pred) / N
    dAdt = (beta * S_pred * I_pred) / N - sigma * A_pred
    dIdt = sigma * A_pred - rho * I_pred - delta * I_pred
    dRdt = gamma * (I_pred + H_pred)
    dDdt = delta * I_pred + theta * H_pred
    dHdt = rho * I_pred - eta * H_pred - theta * H_pred

    # Physics-informed loss: the difference between the predicted derivatives and the actual rate of change
    physics_loss = torch.mean((S_t - dSdt) ** 2) + torch.mean((A_t - dAdt) ** 2) + torch.mean((I_t - dIdt) ** 2) + \
                    torch.mean((R_t - dRdt) ** 2) + torch.mean((D_t - dDdt) ** 2) + torch.mean((H_t - dHdt) ** 2)

    # Data fitting loss: the difference between the predicted and actual compartment sizes
    fitting_loss = torch.mean((model_output - SAIRDH_tensor) ** 2)

    # Total loss
    total_loss = physics_loss + fitting_loss
    return total_loss


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0):
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
            
            
def plot_sairdh_results(t, S, A, I, R, D, H, model, title):
    model.eval()
    with torch.no_grad():
        predictions = model(t).cpu().numpy()
    
    t_np = t.cpu().detach().numpy().flatten()
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))  # Adjusted for additional compartments
    
    compartments = ['Susceptible', 'Exposed', 'Infected', 'Hospitalized', 'Critical', 'Recovered', 'Deceased']
    data = [S, A, I, R, D, H]
    pred_labels = ['S (predicted)', 'A (predicted)', 'I (predicted)', 'R (predicted)', 'D (predicted)', 'H (predicted)']
    
    for ax, data, pred, label, pred_label in zip(axs.flat, data, predictions.T, compartments, pred_labels):
        if data is not None:
            ax.plot(t_np, data.cpu().detach().numpy().flatten(), label=label)
        ax.plot(t_np, pred, linestyle='dashed', label=pred_label)
        ax.set_title(label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Number of Individuals')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"../../reports/figures/{title}.pdf")
    plt.show()


def train_sairdh(model, t_tensor, SAIRDH_tensor, epochs=1000, lr=0.001, N=None, params=None):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        model_output = model(t_tensor)
        
        # Loss calculation for SAIRDH model
        loss = sairdh_loss(model, model_output, SAIRDH_tensor, t_tensor, N, params)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # append the loss
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        early_stopping(loss)
        if early_stopping.early_stop:
            print("Early stopping")
            
            # save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
            }, f"../../models/{model.__class__.__name__}_SAIRDH.pt")
            print("Model saved")
            break
        
    print("Training finished")
    
    return losses


# After training model
with torch.no_grad():
    SAIRDH_pred = model_forward(t_data).cpu().detach().numpy()  # Assuming the output matches SAIRDH compartments order
    SAIRDH_true = SAIRDH_train.cpu().detach().numpy()  # Your true SAIRDH data tensor
    
    # Compute errors for each compartment as needed or in aggregate
    mae = mean_absolute_error(SAIRDH_true, SAIRDH_pred, multioutput='raw_values')
    mse = mean_squared_error(SAIRDH_true, SAIRDH_pred, multioutput='raw_values')
    rmse = np.sqrt(mse)
    
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
