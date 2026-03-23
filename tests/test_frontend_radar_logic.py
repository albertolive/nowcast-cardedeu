"""Tests de la lògica radar del dashboard web (docs/radar_logic.js)."""
import subprocess
from pathlib import Path


def test_radar_logic_node_tests_pass():
    repo_root = Path(__file__).resolve().parent.parent
    test_file = repo_root / "docs" / "radar_logic.test.mjs"

    result = subprocess.run(
        ["node", "--test", str(test_file)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    if result.returncode != 0:
        raise AssertionError(
            "Node radar logic tests failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
