"""Widths for the feature map and the classification or ranking head.

Parameter counts are the exact number of trainable weights. The classifier
cap is the ResNet-18 count from the v3 comparison: 11,180,103.
"""

from __future__ import annotations

CLASSIFIER_CAP = 11_180_103

# The current map uses a hidden width of 256. The other widths are choices.
MAPPER_HIDDENS = (64, 256, 512, 1024, 2048)

# The current head is (64, 32). Larger pairs are listed when they fit the cap.
CLASSIFIER_PRESETS = (
    (64, 32),
    (256, 128),
    (512, 256),
    (1024, 512),
    (2048, 1024),
    (3072, 1536),
)


def linear_parameter_count(in_dim: int, out_dim: int) -> int:
    return int(in_dim) * int(out_dim) + int(out_dim)


def mapper_parameter_count(in_dim: int, out_dim: int, hidden_dim: int) -> int:
    hidden = int(hidden_dim)
    return linear_parameter_count(in_dim, hidden) + linear_parameter_count(hidden, out_dim)


def head_parameter_count(in_dim: int, out_dim: int, hidden: tuple[int, int]) -> int:
    first, second = int(hidden[0]), int(hidden[1])
    return (
        linear_parameter_count(in_dim, first)
        + linear_parameter_count(first, second)
        + linear_parameter_count(second, out_dim)
    )


def normalize_hidden(hidden) -> tuple[int, int]:
    if hidden is None:
        return (64, 32)
    first, second = int(hidden[0]), int(hidden[1])
    if first < 1 or second < 1:
        raise ValueError("hidden widths must be positive")
    return (first, second)


def largest_classifier_hidden(in_dim: int, n_classes: int, cap: int = CLASSIFIER_CAP) -> tuple[int, int]:
    """Largest (h, h/2) on a step of 64 whose classifier stays within ``cap``."""
    best = (64, 32)
    width = 64
    outputs = max(1, int(n_classes))
    while width <= 16384:
        hidden = (width, width // 2)
        if head_parameter_count(in_dim, outputs, hidden) > int(cap):
            break
        best = hidden
        width += 64
    return best


def mapper_options(in_dim: int, out_dim: int) -> list[dict]:
    return [
        {
            "id": str(hidden),
            "hidden_dim": int(hidden),
            "parameters": mapper_parameter_count(in_dim, out_dim, hidden),
        }
        for hidden in MAPPER_HIDDENS
    ]


def classifier_options(in_dim: int, n_classes: int, cap: int = CLASSIFIER_CAP) -> list[dict]:
    """Raw-feature classifier sizes for ``n_classes`` outputs.

    The matching ranker uses the same hidden pair and one output, so it has
    fewer weights. Every option except the current (64, 32) head is omitted
    when it would pass ``cap``.
    """
    outputs = max(1, int(n_classes))
    hidden_pairs = list(CLASSIFIER_PRESETS)
    largest = largest_classifier_hidden(in_dim, outputs, cap)
    if largest not in hidden_pairs:
        hidden_pairs.append(largest)
    options: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for hidden in hidden_pairs:
        pair = (int(hidden[0]), int(hidden[1]))
        if pair in seen:
            continue
        count = head_parameter_count(in_dim, outputs, pair)
        if pair != (64, 32) and count > int(cap):
            continue
        seen.add(pair)
        options.append(
            {
                "id": f"{pair[0]}x{pair[1]}",
                "hidden": [pair[0], pair[1]],
                "parameters": int(count),
            }
        )
    return options
