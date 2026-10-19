import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = Path(__file__).resolve().parent / "board_refine_dom_check.js"


def test_task_class_membership_and_magnifier_shortcut():
    node = shutil.which("node")
    assert node, "node is required to execute the board page script"
    completed = subprocess.run(
        [node, str(CHECK)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
