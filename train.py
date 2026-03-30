import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
 
from model import NeuralNetwork
 
# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "model.pth"
 
# ── Device ────────────────────────────────────────────────────────────────────
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using device: {device}")
 
# ── Data ──────────────────────────────────────────────────────────────────────
train_data = datasets.MNIST(
    root="data", train=True, download=True, transform=ToTensor()
)
test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=ToTensor()
)
 
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
 
 
# ── Training & evaluation helpers ─────────────────────────────────────────────
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
 
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
 
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
 
        # Backward pass
        optimizer.zero_grad()   # clear old gradients BEFORE computing new ones
        loss.backward()
        optimizer.step()
 
        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"  loss: {loss.item():>7.4f}  [{current:>5d}/{size}]")
 
 
def evaluate(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total_loss, correct = 0.0, 0
 
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
 
    avg_loss = total_loss / num_batches
    accuracy = correct / size
    print(f"  Accuracy: {100 * accuracy:.1f}%  |  Avg loss: {avg_loss:.4f}\n")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
 
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")
        print("-" * 30)
        train(train_loader, model, loss_fn, optimizer)
        evaluate(test_loader, model, loss_fn)
 
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")