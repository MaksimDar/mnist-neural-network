import torch.nn as nn
 
 
class NeuralNetwork(nn.Module):
    """
    A simple fully-connected neural network for MNIST digit classification.
    Input:  28x28 grayscale image (flattened to 784)
    Output: 10 class scores (digits 0-9)
    """
 
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),           # <-- activation added (was missing in original notebook)
            nn.Linear(512, 512),
            nn.ReLU(),           # <-- activation added (was missing in original notebook)
            nn.Linear(512, 10),
        )
 
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits