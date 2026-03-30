# mnist-neural-network
MNIST Digit Classifier (PyTorch)
A fully-connected neural network trained on the MNIST dataset to classify handwritten digits (0–9).
Built while following the PyTorch Quickstart Tutorial.

Model architecture
Input (28×28 image)
  → Flatten → 784
  → Linear(784 → 512) + ReLU
  → Linear(512 → 512) + ReLU
  → Linear(512 → 10)
  → Output (10 class scores)

Results
~82–83% test accuracy after 5 epochs (SGD, lr=0.001).

After training, a confusion matrix image is saved as confusion_matrix.png.


Project structure
mnist-pytorch-classifier/
├── model.py          # Neural network definition
├── train.py          # Training loop — run this first
├── evaluate.py       # Predictions + confusion matrix
├── requirements.txt  # Dependencies
└── README.md

How to run
1. Install dependencies
bashpip install -r requirements.txt
2. Train the model
bashpython train.py
This downloads the MNIST data automatically, trains for 5 epochs, and saves model.pth.
3. Evaluate
bashpython evaluate.py
Prints sample predictions and saves a confusion matrix to confusion_matrix.png.

What I learned

Building a custom nn.Module in PyTorch
The full training pipeline: data loading → forward pass → loss → backprop → optimizer step
Why activation functions (ReLU) matter between linear layers
Evaluating a classifier with a confusion matrix


Based on
PyTorch Quickstart Tutorial