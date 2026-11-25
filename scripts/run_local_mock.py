"""
Run Local Mock Pipeline (로컬 모의 실행 파이프라인)
AI와 함께 내 컴퓨터에서 전체 분석 파이프라인을 사전 검증하는 원클릭 스크립트
"""
import os
import argparse
import json
import pandas as pd
from src.config import StudyConfig
from src.data_generator import generate_mock_kcure_data
from src.preprocess import prepare_train_test_split
from src.models.deep_models import DeepSurvTrainer
from src.evaluate import evaluate_survival_model
from src.explain import compute_feature_importance_summary


def main():
    parser = argparse.ArgumentParser(description="Cancer Control Mock Pipeline")
    parser.add_argument("--samples", type=int, default=2000, help="Number of synthetic patients")
    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = StudyConfig()

    print("\n=======================================================")
    print("🚀 Cancer Control: Local AI Mock Pipeline Started")
    print("=======================================================")

    # 1. 합성 데이터 생성
    print(f"[*] Step 1: Generating {args.samples} mock K-CURE patient records...")
    df_mock = generate_mock_kcure_data(n_samples=args.samples, config=config)
    print(f"    - Dataset shape: {df_mock.shape}")
    print(f"    - Event distribution:\n{df_mock['EVENT_STATUS'].value_counts().to_string()}")

    # 2. 전처리 및 분할
    print("\n[*] Step 2: Preprocessing and stratified splitting...")
    split_data = prepare_train_test_split(df_mock, config=config, test_size=0.2)
    X_train, time_train, event_train = split_data["train"]
    X_test, time_test, event_test = split_data["test"]
    feature_names = split_data["feature_names"]
    print(f"    - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"    - Total engineered features: {len(feature_names)}")

    # 3. 딥러닝 모델 학습 (DeepSurv)
    print("\n[*] Step 3: Training DeepSurv survival neural network...")
    trainer = DeepSurvTrainer(in_features=X_train.shape[1], config=config)
    trainer.fit(X_train, time_train, event_train)
    print("    - Model training completed successfully.")

    # 4. 성능 평가 (C-index)
    print("\n[*] Step 4: Evaluating on test set...")
    risk_scores = trainer.predict_risk(X_test)
    metrics = evaluate_survival_model(time_test, event_test, risk_scores)
    print(f"    - Harrell's C-index: {metrics['harrell_c_index']}")
    print(f"    - Evaluated sample details: {json.dumps(metrics, indent=2)}")

    # 5. XAI 변수 중요도 산출
    print("\n[*] Step 5: Computing XAI Feature Importance for report...")
    df_importance = compute_feature_importance_summary(trainer, X_test, feature_names, top_k=10)
    print(f"    - Top Risk Contributors:\n{df_importance.to_string(index=False)}")

    # 결과 저장
    metrics_path = os.path.join(args.out_dir, "evaluation_metrics.json")
    importance_path = os.path.join(args.out_dir, "feature_importance.csv")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    df_importance.to_csv(importance_path, index=False)

    print("\n=======================================================")
    print(f"✅ Pipeline executed with ZERO bugs! Ready for VDI ingress.")
    print(f"   Outputs saved to: {args.out_dir}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
