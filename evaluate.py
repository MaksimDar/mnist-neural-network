import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 
from model import NeuralNetwork
 
# ── Config ────────────────────────────────────────────────────────────────────
MODEL_SAVE_PATH = "model.pth"
CLASSES = list(range(10))   # digits 0-9
 
# ── Device ────────────────────────────────────────────────────────────────────
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
 
# ── Load data & model ─────────────────────────────────────────────────────────
test_data = datasets.MNIST(
    root="data", train=False, download=True, transform=ToTensor()
)
 
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
model.eval()
 
# ── Sample predictions (first 20 images) ─────────────────────────────────────
print("Sample predictions (first 20 test images):")
print("-" * 35)
with torch.no_grad():
    for i in range(20):
        x, y = test_data[i]
        x = x.to(device)
        pred = model(x)
        predicted = CLASSES[pred[0].argmax(0)]
        actual = CLASSES[y]
        match = "✓" if predicted == actual else "✗"
        print(f"  [{match}]  Predicted: {predicted}  |  Actual: {actual}")
 
# ── Full confusion matrix ─────────────────────────────────────────────────────
print("\nBuilding confusion matrix over full test set...")
 
all_labels, all_preds = [], []
 
with torch.no_grad():
    for x, y in test_data:
        x = x.to(device)
        pred = model(x)
        all_preds.append(CLASSES[pred[0].argmax(0)])
        all_labels.append(y)
 
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
disp.plot(cmap="Blues")
plt.title("MNIST Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Confusion matrix saved to confusion_matrix.png")
