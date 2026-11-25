"""
Traditional Statistical & Machine Learning Survival Models
(Cox-PH, Random Survival Forest, and Fine-Gray Competing Risk)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any


class BaselineSurvivalModels:
    """
    고전적 생존분석 및 머신러닝 생존분석 모델 래퍼
    """
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit_cox_ph(self, X_train: np.ndarray, time_train: np.ndarray, event_train: np.ndarray):
        """Cox Proportional Hazards model using lifelines"""
        try:
            from lifelines import CoxPHFitter
            # Event: 1 (만성질환 발생) vs 0 (Censored / Competing risk)
            binary_event = (event_train == 1).astype(int)
            df = pd.DataFrame(X_train, columns=self.feature_names)
            df["time"] = time_train
            df["event"] = binary_event

            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(df, duration_col="time", event_col="event")
            return cph
        except Exception as e:
            print(f"[Warning] Lifelines CoxPH fit error: {e}")
            return None

    def fit_random_survival_forest(self, X_train: np.ndarray, time_train: np.ndarray, event_train: np.ndarray):
        """Random Survival Forest using scikit-survival"""
        try:
            from sksurv.ensemble import RandomSurvivalForest
            binary_event = (event_train == 1)
            y_struct = np.array(
                list(zip(binary_event, time_train)),
                dtype=[("Status", "?"), ("Survival_in_days", "<f8")]
            )

            rsf = RandomSurvivalForest(
                n_estimators=100,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            )
            rsf.fit(X_train, y_struct)
            return rsf
        except Exception as e:
            print(f"[Info] scikit-survival not loaded or fit skipped: {e}")
            return None
