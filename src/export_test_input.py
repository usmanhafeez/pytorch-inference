
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms

PROJECT_DIR = Path(__file__).resolve().parent.parent

dataset = datasets.FashionMNIST(
    root=PROJECT_DIR / "data",
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

image, actual_label = dataset[0]

# Save 28x28 float32 pixel values
input_path = PROJECT_DIR / "models" / "test_input.bin"

image.numpy().astype(np.float32).tofile(input_path)

print("Actual label:", actual_label)
print("Input shape:", image.shape)
print("Saved:", input_path)

