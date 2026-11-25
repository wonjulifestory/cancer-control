"""
Evaluation Metrics for Cancer Control Survival Models
(Harrell's C-index, Time-dependent metrics)
"""
import numpy as np
from typing import Dict, Any


def concordance_index_fast(time: np.ndarray, event: np.ndarray, risk_scores: np.ndarray) -> float:
    """
    Computes Harrell's C-index (Concordance Index)
    Higher risk_score should correspond to shorter survival time.
    """
    event = (event == 1).astype(int)
    n = len(time)
    concordant = 0
    admissible = 0

    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if time[i] < time[j]:
                admissible += 1
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] == risk_scores[j]:
                    concordant += 0.5

    return concordant / admissible if admissible > 0 else 0.5


def evaluate_survival_model(
    time_true: np.ndarray,
    event_true: np.ndarray,
    risk_scores: np.ndarray
) -> Dict[str, float]:
    """Calculate core performance metrics for the RFP report"""
    try:
        from lifelines.utils import concordance_index
        c_idx = concordance_index(time_true, -risk_scores, (event_true == 1))
    except ImportError:
        c_idx = concordance_index_fast(time_true, event_true, risk_scores)

    return {
        "harrell_c_index": round(float(c_idx), 4),
        "sample_size": len(time_true),
        "event_count": int(np.sum(event_true == 1)),
        "censored_count": int(np.sum(event_true == 0)),
        "competing_risk_count": int(np.sum(event_true == 2))
    }
