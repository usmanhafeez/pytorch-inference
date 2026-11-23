
from pathlib import Path

import torch
from torchvision import datasets, transforms

from model import FashionCNN

# Paths relative to the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "fashion_cnn.pth"
DATA_DIR = PROJECT_DIR / "data"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Load model architecture
model = FashionCNN()

# Load trained weights
state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)
model.load_state_dict(state_dict)

# Switch to inference mode
model.eval()

# Load test images
test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

# Select the first test image
image, actual_label = test_dataset[0]

# Add batch dimension: [1, 28, 28] -> [1, 1, 28, 28]
input_tensor = image.unsqueeze(0)

# Run inference
with torch.inference_mode():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_label = probabilities.argmax(dim=1).item()

print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)

print("Actual class:", CLASS_NAMES[actual_label])
print("Predicted class:", CLASS_NAMES[predicted_label])
print(
    "Prediction confidence:",
    f"{probabilities[0, predicted_label].item() * 100:.2f}%"
)

print("Prediction correct:", predicted_label == actual_label)

