"""Summarize observable MM and Qwen evidence without changing annotation decisions."""

from math import isfinite


METHOD = "min_accepted_mm_and_two_order_qwen_support_v1"


def _bounded_score(value):
    if type(value) in (int, float) and isfinite(value) and 0 <= value <= 1:
        return float(value)
    return None


def candidate_confidence(candidate):
    """Return MM similarity and Qwen support for accepting the proposed extent/class."""
    votes = candidate.get("votes")
    support = []
    if isinstance(votes, list) and len(votes) == 2:
        for vote in votes:
            if not isinstance(vote, dict):
                break
            probabilities = vote.get("probabilities")
            conditional = (_bounded_score(probabilities.get("A")) if isinstance(probabilities, dict) else None)
            mass = _bounded_score(vote.get("allowed_token_mass"))
            if conditional is None or mass is None:
                break
            support.append(conditional * mass)
    return {"candidate_id": candidate.get("id"), "label": candidate.get("label"),
            "mm_score": _bounded_score(candidate.get("score")),
            "qwen_score": min(support) if len(support) == 2 else None}


def _weakest(boxes, key):
    values = [box[key] for box in boxes]
    return min(values) if values and all(value is not None for value in values) else None


def frame_confidence(frame):
    """Derive the original model evidence, including for older saved inference.

    ``score`` is the minimum of both signals over every originally accepted box.
    On a pending frame that number describes the retained boxes only. The scan
    or an unmatched target may still require review, regardless of this score.
    A completed result without an accepted box has a score of zero. Unknown or
    absent evidence remains unavailable. Review edits never change these values.
    """
    exceptions = frame.get("exceptions") or ()
    reasons = list(dict.fromkeys(item["code"] for item in exceptions if isinstance(item, dict)
                                 and isinstance(item.get("code"), str) and item["code"]))
    candidates = frame.get("candidates")
    evaluated = isinstance(candidates, list) and frame.get("status") != "awaiting_inference"
    rows = [candidate for candidate in candidates if isinstance(candidate, dict)] if evaluated else []
    accepted = [candidate_confidence(candidate) for candidate in rows if candidate.get("verdict") == "accepted"]
    mm_score = _weakest(accepted, "mm_score")
    qwen_score = _weakest(accepted, "qwen_score")
    score = None
    if not evaluated:
        status = "unavailable"
        reasons.append("inference_not_run" if frame.get("status") == "awaiting_inference" else "missing_candidate_evidence")
    elif not accepted:
        status, score = "no_accepted", 0.0
        if "no_accepted_boxes" not in reasons:
            reasons.append("no_accepted_boxes")
    else:
        status = "review" if exceptions or frame.get("status") in {"review", "pending"} else "ready"
        if mm_score is None:
            reasons.append("missing_mm_evidence")
        if qwen_score is None:
            reasons.append("missing_qwen_evidence")
        if mm_score is not None and qwen_score is not None:
            score = min(mm_score, qwen_score)
    return {"method": METHOD, "score": score, "mm_score": mm_score, "qwen_score": qwen_score,
            "status": status, "accepted_count": len(accepted), "candidate_count": len(rows),
            "reasons": reasons, "boxes": accepted}
