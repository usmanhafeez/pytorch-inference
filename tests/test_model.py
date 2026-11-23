
from pathlib import Path

import torch
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model import FashionCNN

MODEL_PATH = PROJECT_DIR / "models" / "fashion_cnn.pth"
TORCHSCRIPT_PATH = PROJECT_DIR / "models" / "fashion_cnn.pt"


def test_model_output_shape():
    model = FashionCNN()
    model.eval()

    input_tensor = torch.randn(1, 1, 28, 28)

    with torch.inference_mode():
        output = model(input_tensor)

    assert output.shape == (1, 10)


def test_model_weights_load():
    model = FashionCNN()

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    assert not model.training


def test_torchscript_matches_python_model():
    model = FashionCNN()

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    scripted_model = torch.jit.load(
        str(TORCHSCRIPT_PATH),
        map_location="cpu"
    )
    scripted_model.eval()

    input_tensor = torch.randn(1, 1, 28, 28)

    with torch.inference_mode():
        python_output = model(input_tensor)
        scripted_output = scripted_model(input_tensor)

    assert torch.allclose(
        python_output,
        scripted_output,
        atol=1e-6,
        rtol=1e-5
    )

