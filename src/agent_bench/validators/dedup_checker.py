"""Deduplication checks for eval cases."""

import hashlib
import json
from typing import Any


def _get_prompt(case: dict[str, Any]) -> str:
    """Extract prompt text from case, checking multiple field names."""
    return (case.get("prompt_or_user_goal") or case.get("prompt") or "").strip().lower()


def compute_case_hash(case: dict[str, Any]) -> str:
    """SHA256 of normalized prompt + initial_state."""
    prompt = _get_prompt(case)
    initial_state = case.get("initial_state") or {}
    state_str = json.dumps(initial_state, sort_keys=True, default=str)
    content = f"{prompt}|{state_str}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _tokenize(text: str) -> set[str]:
    """Simple whitespace + lowercase tokenization."""
    return set(text.strip().lower().split())


def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union


def find_duplicates(cases: list[dict[str, Any]], threshold: float = 0.85) -> list[tuple[str, str, float]]:
    """Return pairs of (id1, id2, similarity) for cases above threshold.

    Uses Jaccard similarity on tokenized prompt words, optimized with upper-bound length pruning.
    """
    duplicates: list[tuple[str, str, float]] = []
    tokenized: list[tuple[str, set[str]]] = []

    for case in cases:
        case_id = case.get("id", "unknown")
        tokens = _tokenize(_get_prompt(case))
        tokenized.append((case_id, tokens))

    for i in range(len(tokenized)):
        for j in range(i + 1, len(tokenized)):
            id1, tokens1 = tokenized[i]
            id2, tokens2 = tokenized[j]
            
            len1 = len(tokens1)
            len2 = len(tokens2)
            
            # Handle edge cases with empty sets
            if len1 == 0 or len2 == 0:
                if len1 == len2:  # both empty
                    sim = 1.0
                else:
                    sim = 0.0
                if sim >= threshold:
                    duplicates.append((id1, id2, sim))
                continue
            
            # Prune mathematically impossible pairs:
            # Jaccard(A, B) = |A ∩ B| / |A ∪ B| <= min(|A|, |B|) / max(|A|, |B|)
            if min(len1, len2) / max(len1, len2) < threshold:
                continue

            sim = _jaccard_similarity(tokens1, tokens2)
            if sim >= threshold:
                duplicates.append((id1, id2, sim))

    return duplicates

