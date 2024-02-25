import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms, datasets

class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        
        self.all_layers = torch.nn.Sequential(
            torch.nn.Linear(num_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        logits = self.all_layers(x)
        return logits
    
def get_data_loaders(train_batch_size, val_batch_size):
    # load MNIST data
    train_dataset = datasets.MNIST(
        root="data", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="data", train=False, transform=transforms.ToTensor()
    )

    train_dataset, val_dataset = random_split(train_dataset, [50000, 10000])

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=train_batch_size, num_workers=0, shuffle=True)
    
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=val_batch_size, num_workers=0, shuffle=False)
    
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=val_batch_size, num_workers=0, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_accuracy(model, data_loader, device):
    
    if device is None:
        device = torch.device("cpu")
    model = model.eval()
    
    correct = 0.0
    total = 0
    
    for idx, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        with torch.no_grad():
            logits = model(inputs)
            
        predictions = torch.argmax(logits, dim=1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
    return correct / total
