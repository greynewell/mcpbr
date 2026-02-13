"""P/R/F1 set-based evaluation for Supermodel benchmarks."""


def normalize_path(filepath: str) -> str:
    """Normalize file path for comparison."""
    p = filepath.replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    p = p.lstrip("/")
    return p


def normalize_name(name: str) -> str:
    """Normalize symbol name for comparison."""
    return name.strip()


def build_comparison_set(
    items: list[dict],
    key_fields: tuple[str, str] = ("file", "name"),
) -> set[tuple[str, str]]:
    """Build a set of normalized tuples from prediction/ground truth items.

    Args:
        items: List of dicts with the key fields.
        key_fields: Tuple of (field_a, field_b) to extract.

    Returns:
        Set of normalized (field_a_value, field_b_value) tuples.
    """
    result = set()
    fa, fb = key_fields
    for item in items:
        a = normalize_path(item.get(fa, "")) if fa == "file" else normalize_name(item.get(fa, ""))
        b = normalize_name(item.get(fb, ""))
        if a and b:
            result.add((a, b))
    return result


def compute_prf1(
    predictions: list[dict],
    ground_truth: list[dict],
    key_fields: tuple[str, str] = ("file", "name"),
) -> dict:
    """Compute precision, recall, F1 from predictions vs ground truth.

    Args:
        predictions: List of prediction dicts.
        ground_truth: List of ground truth dicts.
        key_fields: Fields to use for set comparison.

    Returns:
        Dict with precision, recall, f1_score, tp, fp, fn counts, and resolved boolean.
    """
    pred_set = build_comparison_set(predictions, key_fields)
    gt_set = build_comparison_set(ground_truth, key_fields)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "found": len(pred_set),
        "expected": len(gt_set),
    }
