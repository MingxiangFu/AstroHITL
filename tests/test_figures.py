from pathlib import Path

import numpy as np

from hitl_features.figures import save_comparison_figures


def test_four_arm_curves_and_contact_sheet_use_shared_files(tmp_path: Path):
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    arms = []
    for name, shift in (
        ("A_recommended_raw", 0.0),
        ("B_recommended_mapped", 0.05),
        ("C_random_raw", 0.1),
        ("D_random_mapped", 0.15),
    ):
        scores = np.linspace(0.1, 0.9, 8) + shift
        arms.append({"name": name, "y": labels, "scores": scores})
    saved = save_comparison_figures(
        tmp_path,
        arms,
        high=[(8, 0.95), (6, 0.8)],
        low=[(1, 0.05), (2, 0.1)],
    )
    for key in ("roc", "pr", "contact_sheet"):
        path = tmp_path / saved[key]
        assert path.is_file()
        assert path.read_bytes().startswith(b"\x89PNG")
