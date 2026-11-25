"""
Configuration file for Cancer Control Study
(RFP 1-2. 암 공공데이터 활용을 통한 근거생성 연구)
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class StudyConfig:
    random_seed: int = 42

    # Survival and Target endpoints
    # 0: Censored, 1: Chronic Disease Event, 2: Cancer Recurrence/Death (Competing Risk)
    event_col: str = "EVENT_STATUS"
    time_col: str = "SURVIVAL_MONTHS"

    # 암종 카테고리 정의
    cancer_types: List[str] = field(
        default_factory=lambda: ["Gastric", "Colorectal", "Liver", "Breast", "Lung", "Pancreas", "Other"]
    )

    # 연속형 입력 변수
    continuous_features: List[str] = field(
        default_factory=lambda: [
            "AGE_AT_DIAGNOSIS",
            "BMI",
            "SBP",
            "DBP",
            "FBS",
            "TOTAL_CHOL",
            "HDL",
            "LDL",
            "TRIGLYCERIDE",
            "EGFR",
            "AST",
            "ALT",
            "CCI_SCORE",  # Charlson Comorbidity Index
            "CARDIO_TOXIC_CHEMO_CYCLES",
            "CHEST_RT_FRACTIONS",
        ]
    )

    # 범주형 입력 변수
    categorical_features: List[str] = field(
        default_factory=lambda: [
            "SEX",
            "CANCER_TYPE",
            "SEER_STAGE",       # 1: Localized, 2: Regional, 3: Distant
            "INCOME_DECILE",    # 1~10 (or grouped: Low, Middle, High)
            "SMOKING_STATUS",   # 0: Never, 1: Former, 2: Current
            "ALCOHOL_FREQ",     # 0: None, 1: Moderate, 2: Heavy
            "PHYSICAL_ACT",     # 0: Low, 1: Moderate, 2: High
            "BASELINE_HTN",     # 고혈압 기저질환 (0/1)
            "BASELINE_DM",      # 당뇨 기저질환 (0/1)
            "BASELINE_DYSLIPID", # 이상지질혈증 (0/1)
        ]
    )

    # 딥러닝 학습 하이퍼파라미터
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 40
    early_stopping_patience: int = 7
