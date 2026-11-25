"""
Synthetic Mock Data Generator for K-CURE Cancer Public Library
(암 공공 라이브러리 모의 데이터 생성기)
"""
import numpy as np
import pandas as pd
from typing import Optional
from src.config import StudyConfig


def generate_mock_kcure_data(
    n_samples: int = 2000,
    config: Optional[StudyConfig] = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    K-CURE 공공 라이브러리의 5대 기관 결합 스키마와 동일한 분포의
    모의(Synthetic) 데이터를 생성하여 로컬 개발 환경을 지원합니다.
    """
    if config is None:
        config = StudyConfig()

    np.random.seed(random_seed)

    # 1. 환자 기본 정보 및 암등록 (중앙암등록본부)
    patient_ids = [f"K_CURE_{i:06d}" for i in range(1, n_samples + 1)]
    age_at_diag = np.random.normal(loc=58, scale=11, size=n_samples).clip(30, 85).astype(int)
    sex = np.random.choice([1, 2], size=n_samples, p=[0.52, 0.48])  # 1: 남, 2: 여

    cancer_types = np.random.choice(
        config.cancer_types,
        size=n_samples,
        p=[0.20, 0.22, 0.12, 0.18, 0.15, 0.05, 0.08]
    )
    seer_stage = np.random.choice([1, 2, 3], size=n_samples, p=[0.55, 0.35, 0.10])

    # 2. 건보공단 자격 및 건강검진 (NHIS)
    income_decile = np.random.randint(1, 11, size=n_samples)
    bmi = np.random.normal(loc=24.2, scale=3.5, size=n_samples).clip(16.0, 40.0).round(1)
    sbp = np.random.normal(loc=126, scale=15, size=n_samples).clip(90, 190).astype(int)
    dbp = np.random.normal(loc=79, scale=10, size=n_samples).clip(55, 120).astype(int)
    fbs = np.random.normal(loc=108, scale=28, size=n_samples).clip(65, 320).astype(int)
    total_chol = np.random.normal(loc=192, scale=38, size=n_samples).clip(100, 350).astype(int)
    hdl = np.random.normal(loc=52, scale=12, size=n_samples).clip(25, 95).astype(int)
    ldl = np.random.normal(loc=115, scale=32, size=n_samples).clip(45, 230).astype(int)
    triglyceride = np.random.normal(loc=140, scale=65, size=n_samples).clip(40, 500).astype(int)
    egfr = np.random.normal(loc=82, scale=18, size=n_samples).clip(20, 130).round(1)
    ast = np.random.normal(loc=26, scale=14, size=n_samples).clip(10, 160).astype(int)
    alt = np.random.normal(loc=24, scale=15, size=n_samples).clip(8, 180).astype(int)

    smoking = np.random.choice([0, 1, 2], size=n_samples, p=[0.55, 0.25, 0.20])
    alcohol = np.random.choice([0, 1, 2], size=n_samples, p=[0.45, 0.40, 0.15])
    physical_act = np.random.choice([0, 1, 2], size=n_samples, p=[0.40, 0.40, 0.20])

    # 3. 심평원 청구 기반 기저질환 및 치료 정보 (HIRA)
    base_htn = (sbp >= 140) | (np.random.rand(n_samples) < 0.25)
    base_dm = (fbs >= 126) | (np.random.rand(n_samples) < 0.18)
    base_dyslipid = (total_chol >= 240) | (np.random.rand(n_samples) < 0.22)
    cci_score = np.random.poisson(lam=1.4, size=n_samples).clip(0, 8)

    cardio_toxic_chemo = np.random.poisson(lam=3.5, size=n_samples).clip(0, 16)
    chest_rt_fractions = np.random.choice([0, 15, 25, 30], size=n_samples, p=[0.60, 0.15, 0.15, 0.10])

    # 4. 생존 및 만성질환 사건 생성 (Time-to-Event with Competing Risk)
    # 위험도 점수 계산 (항암제 심독성, 방사선, 당뇨, 혈압, 연령, BMI 등)
    linear_risk = (
        0.04 * (age_at_diag - 50) +
        0.05 * (bmi - 23) +
        0.015 * (sbp - 120) +
        0.008 * (fbs - 100) +
        0.08 * cardio_toxic_chemo +
        0.03 * chest_rt_fractions +
        0.35 * base_htn.astype(int) +
        0.45 * base_dm.astype(int) +
        0.25 * smoking +
        0.15 * cci_score
    )

    base_hazard = 0.006
    individual_hazard = base_hazard * np.exp(linear_risk * 0.4)

    # 추적기간: 진단 5년 후부터 최대 10년(120개월)
    time_to_chronic = np.random.exponential(scale=1.0 / individual_hazard).round(1)
    time_to_competing_death = np.random.exponential(scale=180.0, size=n_samples).round(1)
    censor_time = np.random.uniform(36, 120, size=n_samples).round(1)

    survival_time = np.minimum.reduce([time_to_chronic, time_to_competing_death, censor_time])

    # 0: Censored, 1: 만성질환 발생, 2: 암 재발/경쟁사망
    event_status = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if survival_time[i] == time_to_chronic[i] and time_to_chronic[i] < censor_time[i]:
            event_status[i] = 1
        elif survival_time[i] == time_to_competing_death[i] and time_to_competing_death[i] < censor_time[i]:
            event_status[i] = 2
        else:
            event_status[i] = 0

    df = pd.DataFrame({
        "ID": patient_ids,
        "AGE_AT_DIAGNOSIS": age_at_diag,
        "SEX": sex,
        "CANCER_TYPE": cancer_types,
        "SEER_STAGE": seer_stage,
        "INCOME_DECILE": income_decile,
        "BMI": bmi,
        "SBP": sbp,
        "DBP": dbp,
        "FBS": fbs,
        "TOTAL_CHOL": total_chol,
        "HDL": hdl,
        "LDL": ldl,
        "TRIGLYCERIDE": triglyceride,
        "EGFR": egfr,
        "AST": ast,
        "ALT": alt,
        "SMOKING_STATUS": smoking,
        "ALCOHOL_FREQ": alcohol,
        "PHYSICAL_ACT": physical_act,
        "BASELINE_HTN": base_htn.astype(int),
        "BASELINE_DM": base_dm.astype(int),
        "BASELINE_DYSLIPID": base_dyslipid.astype(int),
        "CCI_SCORE": cci_score,
        "CARDIO_TOXIC_CHEMO_CYCLES": cardio_toxic_chemo,
        "CHEST_RT_FRACTIONS": chest_rt_fractions,
        config.time_col: survival_time,
        config.event_col: event_status,
    })

    return df


if __name__ == "__main__":
    df = generate_mock_kcure_data(500)
    print("Generated mock dataset shape:", df.shape)
    print("Event distribution:\n", df["EVENT_STATUS"].value_counts(normalize=True))
