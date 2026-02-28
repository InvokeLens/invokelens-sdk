"""Prompt fingerprinting for drift detection.

Computes lightweight structural fingerprints of prompt text using
pure-Python hashing and statistics — no external ML dependencies.
"""

import hashlib
import re
from typing import Optional

# Regex to find {variable_name} style template placeholders
_TEMPLATE_VAR_RE = re.compile(r"\{([a-zA-Z_]\w*)\}")


def compute_fingerprint(prompt: str) -> dict:
    """Compute a structural fingerprint of a prompt string.

    Args:
        prompt: The raw prompt text.

    Returns:
        Dict with keys: prompt_hash, structure_hash, char_count,
        word_count, line_count, template_vars.
    """
    if not prompt:
        return {
            "prompt_hash": hashlib.sha256(b"").hexdigest(),
            "structure_hash": hashlib.sha256(b"").hexdigest(),
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
            "template_vars": [],
        }

    # Normalize for hashing: lowercase, strip leading/trailing whitespace
    normalized = prompt.strip().lower()
    prompt_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # Template variable extraction
    template_vars = sorted(set(_TEMPLATE_VAR_RE.findall(prompt)))

    # Structure hash: replace all template vars with {VAR} to capture skeleton
    skeleton = _TEMPLATE_VAR_RE.sub("{VAR}", prompt.strip().lower())
    structure_hash = hashlib.sha256(skeleton.encode("utf-8")).hexdigest()

    return {
        "prompt_hash": prompt_hash,
        "structure_hash": structure_hash,
        "char_count": len(prompt),
        "word_count": len(prompt.split()),
        "line_count": prompt.count("\n") + 1,
        "template_vars": template_vars,
    }


def compute_similarity(a: dict, b: dict) -> float:
    """Compare two fingerprints and return a similarity score (0.0–1.0).

    - 1.0 = exact same prompt text
    - 0.9 = same template structure, different variable values
    - Otherwise: weighted metric comparison

    Args:
        a: Fingerprint dict from compute_fingerprint().
        b: Fingerprint dict from compute_fingerprint().

    Returns:
        Float between 0.0 and 1.0.
    """
    if not a or not b:
        return 0.0

    # Exact match
    if a.get("prompt_hash") == b.get("prompt_hash"):
        return 1.0

    # Same template structure
    if a.get("structure_hash") == b.get("structure_hash"):
        return 0.9

    # Metric-based comparison
    metrics = ["char_count", "word_count", "line_count"]
    ratios = []
    for m in metrics:
        va = a.get(m, 0)
        vb = b.get(m, 0)
        max_val = max(va, vb)
        if max_val == 0:
            ratios.append(1.0)  # Both zero → identical for this metric
        else:
            ratios.append(1.0 - abs(va - vb) / max_val)

    return max(0.0, min(1.0, sum(ratios) / len(ratios)))
