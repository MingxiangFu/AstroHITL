import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = Path(__file__).resolve().parent / "annotation_drag_dom_check.js"


def test_annotation_drop_on_overlay_shows_category_and_clears_list():
    node = shutil.which("node")
    assert node, "node is required to execute the annotation page script"
    completed = subprocess.run(
        [node, str(CHECK)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
