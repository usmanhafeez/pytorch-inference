
from pathlib import Path
import re
import subprocess

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
CPP_EXECUTABLE = PROJECT_DIR / "build" / "torch_cpp_inference"


def test_cpp_inference():
    if not CPP_EXECUTABLE.is_file():
        pytest.skip("C++ executable not built")

    result = subprocess.run(
        [str(CPP_EXECUTABLE)],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "C++ inference successful!" in result.stdout
    assert "Predicted class: Ankle boot" in result.stdout

    match = re.search(
        r"Confidence:\s*([0-9.]+)%",
        result.stdout
    )

    assert match is not None, "Confidence missing from output"

    confidence = float(match.group(1))
    assert 0 <= confidence <= 100

