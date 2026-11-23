
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "fashion_cnn.pt"
INPUT_PATH = PROJECT_ROOT / "models" / "test_input.bin"
CPP_EXECUTABLE = PROJECT_ROOT / "build" / "torch_cpp_inference"


def test_python_cpp_numerical_equivalence():
    if not CPP_EXECUTABLE.exists():
        pytest.skip("C++ inference executable not built")

    if not MODEL_PATH.exists() or not INPUT_PATH.exists():
        pytest.skip("Exported model or test input missing")

    torch.set_num_threads(1)

    # Read the same binary image used by C++
    pixels = np.fromfile(INPUT_PATH, dtype=np.float32)
    assert pixels.size == 28 * 28

    input_tensor = torch.from_numpy(
        pixels.copy().reshape(1, 1, 28, 28)
    )

    # Python TorchScript inference
    model = torch.jit.load(str(MODEL_PATH), map_location="cpu")
    model.eval()

    with torch.inference_mode():
        python_logits = model(input_tensor).numpy().flatten()

    # C++ LibTorch inference
    result = subprocess.run(
        [str(CPP_EXECUTABLE)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    logits_lines = [
        line for line in result.stdout.splitlines()
        if line.startswith("LOGITS:")
    ]

    assert len(logits_lines) == 1, (
        f"Expected one LOGITS line, got:\n{result.stdout}"
    )

    cpp_logits = np.fromstring(
        logits_lines[0].removeprefix("LOGITS:"),
        sep=" ",
    )

    assert python_logits.shape == (10,)
    assert cpp_logits.shape == (10,)

    np.testing.assert_allclose(
        cpp_logits,
        python_logits,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Python and C++ inference outputs differ",
    )

    print("\nPython and C++ logits match!")
    print("Maximum absolute difference:",
          np.max(np.abs(python_logits - cpp_logits)))

