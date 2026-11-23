
import torch
import torch.nn as nn

# Make random weights reproducible
torch.manual_seed(42)

# Define our neural network
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        return self.network(x)


# Create the model
model = SimpleNeuralNetwork()

# Input tensor with 3 features
x = torch.tensor([[1.0, 2.0, 3.0]])

# Run inference
model.eval()

with torch.inference_mode():
    output = model(x)

print("Input:", x)
print("Input shape:", x.shape)

print("\nModel architecture:")
print(model)

print("\nOutput:", output)
print("Output shape:", output.shape)

print("\nCPU device:", output.device)


# Inspect the first linear layer
first_layer = model.network[0]

weights = first_layer.weight
bias = first_layer.bias

print("\n=== First layer parameters ===")
print("Weights:\n", weights)
print("Weight shape:", weights.shape)
print("Bias:", bias)
print("Bias shape:", bias.shape)

# Reproduce the layer's calculation manually
with torch.inference_mode():
    manual_output = torch.matmul(x, weights.T) + bias
    pytorch_output = first_layer(x)

print("\n=== Compare calculations ===")
print("Manual calculation:", manual_output)
print("PyTorch nn.Linear:", pytorch_output)

print(
    "Results match:",
    torch.allclose(manual_output, pytorch_output)
)

# Apply the ReLU activation
relu_output = torch.relu(pytorch_output)

print("\n=== ReLU activation ===")
print("Before ReLU:", pytorch_output)
print("After ReLU:", relu_output)

