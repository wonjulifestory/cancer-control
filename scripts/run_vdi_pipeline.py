"""
Run VDI Batch Pipeline (안심활용센터 VDI 내부 실행용 스크립트)
VDI 내부에서 실제 K-CURE 암 공공 라이브러리 데이터를 읽어와서 일괄 학습 및 통계 요약표를 출력합니다.
"""
import os
import sys
import argparse
import json

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.config import StudyConfig
from src.preprocess import prepare_train_test_split
from src.models.deep_models import DeepSurvTrainer
from src.evaluate import evaluate_survival_model
from src.explain import compute_feature_importance_summary


def load_vdi_data(data_path: str) -> pd.DataFrame:
    """VDI 내부의 실제 데이터 포맷(Parquet, CSV, SAS) 로드"""
    if data_path.endswith(".parquet"):
        return pd.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        return pd.read_csv(data_path)
    elif data_path.endswith(".sas7bdat"):
        return pd.read_sas(data_path, format="sas7bdat")
    else:
        raise ValueError(f"Unsupported file format: {data_path}")


def main():
    parser = argparse.ArgumentParser(description="Cancer Control VDI Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to real K-CURE dataset in VDI")
    parser.add_argument("--out_dir", type=str, default="./outputs_vdi", help="Directory for exportable summaries")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = StudyConfig()

    print(f"[*] Loading real dataset from {args.data_path}...")
    df = load_vdi_data(args.data_path)
    print(f"[*] Loaded records: {df.shape[0]}, Features: {df.shape[1]}")

    print("[*] Running preprocessing...")
    split_data = prepare_train_test_split(df, config=config, test_size=0.2)
    X_train, time_train, event_train = split_data["train"]
    X_test, time_test, event_test = split_data["test"]
    feature_names = split_data["feature_names"]

    print("[*] Training survival deep learning model...")
    trainer = DeepSurvTrainer(in_features=X_train.shape[1], config=config)
    trainer.fit(X_train, time_train, event_train)

    print("[*] Evaluating...")
    risk_scores = trainer.predict_risk(X_test)
    metrics = evaluate_survival_model(time_test, event_test, risk_scores)

    print("[*] Computing non-identifiable feature importance for export...")
    df_importance = compute_feature_importance_summary(trainer, X_test, feature_names, top_k=20)

    # 비식별 반출 파일 저장
    metrics_path = os.path.join(args.out_dir, "exportable_metrics.json")
    importance_path = os.path.join(args.out_dir, "exportable_feature_importance.csv")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    df_importance.to_csv(importance_path, index=False)

    print(f"✅ VDI execution completed! Export files created in: {args.out_dir}")


if __name__ == "__main__":
    main()
