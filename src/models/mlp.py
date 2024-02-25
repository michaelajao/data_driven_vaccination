import torch
import torch.nn.functional as F
from shared_utilities import get_data_loaders, get_accuracy, PyTorchMLP
from watermark import watermark


def compute_total_loss(model, data_loader, device):
    if device is None:
        device = torch.device("cpu")
    model = model.eval()

    total_loss = 0.0
    total = 0.0

    for idx, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.no_grad():
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels, reduction="sum")

        total_loss += loss.item()
        total += len(labels)

    return total_loss / total


def train(
    model, optimizer, train_loader, val_loader, num_epochs=10, seed=1, device=None
):
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # set the seed
    torch.manual_seed(seed)

    train_loss = []
    val_loss = []

    for epoch in range(num_epochs):
        model = model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        train_loss.append(running_loss / len(train_loader))
        
        val_loss.append(compute_total_loss(model, val_loader, device))
        
        print(f"Epoch: {epoch + 1}, Training Loss: {train_loss[-1]:.4f}, Validation Loss: {val_loss[-1]:.4f}")
        
    return train_loss, val_loss



if __name__ == "__main__":
    print(watermark(packages="torch", python=True))
    print("Torch CUDA available:", torch.cuda.is_available())
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_data_loaders(64, 64)

    model = PyTorchMLP(num_features=784, num_classes=10)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    train_loss, val_loss = train(
        model, optimizer, train_loader, val_loader, num_epochs=10, seed=1, device=device
    )

    train_accuracy = get_accuracy(model, train_loader, device)
    test_accuracy = get_accuracy(model, test_loader, device)
    val_accuracy = get_accuracy(model, val_loader, device)

    print(
        f"Train accuracy: {train_accuracy:.4f}"
        f"Validation accuracy: {val_accuracy:.4f}"
        f"Test accuracy: {test_accuracy:.4f}"
    )
