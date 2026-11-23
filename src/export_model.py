
from pathlib import Path

import torch
from model import FashionCNN

PROJECT_DIR = Path(__file__).resolve().parent.parent

WEIGHTS_PATH = PROJECT_DIR / "models" / "fashion_cnn.pth"
EXPORT_PATH = PROJECT_DIR / "models" / "fashion_cnn.pt"

# Recreate the model architecture
model = FashionCNN()

# Load trained weights
model.load_state_dict(
    torch.load(
        WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# Convert Python model into TorchScript
scripted_model = torch.jit.script(model)

# Save TorchScript model for C++ inference
scripted_model.save(str(EXPORT_PATH))

print("TorchScript model exported successfully!")
print("Saved to:", EXPORT_PATH)

# Verify exported model
loaded_model = torch.jit.load(str(EXPORT_PATH))
loaded_model.eval()

test_input = torch.randn(1, 1, 28, 28)

with torch.inference_mode():
    original_output = model(test_input)
    exported_output = loaded_model(test_input)

print(
    "Original and exported outputs match:",
    torch.allclose(
        original_output,
        exported_output,
        atol=1e-6
    )
)

