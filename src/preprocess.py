"""
Data Preprocessing Pipeline for Cancer Control AI Models
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from src.config import StudyConfig


class CancerDataPreprocessor:
    def __init__(self, config: StudyConfig):
        self.config = config
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.feature_names = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit preprocessor on train data and return (X, time, event)."""
        return self._process(df, is_fit=True)

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform validation/test data using fitted parameters."""
        return self._process(df, is_fit=False)

    def _process(self, df: pd.DataFrame, is_fit: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df_copy = df.copy()

        # 연속형 변수 처리
        num_data = df_copy[self.config.continuous_features].values
        if is_fit:
            num_data = self.num_imputer.fit_transform(num_data)
            num_scaled = self.scaler.fit_transform(num_data)
        else:
            num_data = self.num_imputer.transform(num_data)
            num_scaled = self.scaler.transform(num_data)

        # 범주형 변수 처리
        cat_data = df_copy[self.config.categorical_features].astype(str).values
        if is_fit:
            cat_data = self.cat_imputer.fit_transform(cat_data)
            cat_encoded = self.encoder.fit_transform(cat_data)
            cat_names = self.encoder.get_feature_names_out(self.config.categorical_features).tolist()
            self.feature_names = self.config.continuous_features + cat_names
        else:
            cat_data = self.cat_imputer.transform(cat_data)
            cat_encoded = self.encoder.transform(cat_data)

        # X 결합
        X = np.hstack([num_scaled, cat_encoded])

        # 타겟 라벨
        time = df_copy[self.config.time_col].values.astype(float)
        event = df_copy[self.config.event_col].values.astype(int)

        return X, time, event


def prepare_train_test_split(
    df: pd.DataFrame,
    config: StudyConfig,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Train / Validation / Test 세트 분할 및 전처리 적용"""
    # 층화 추출 (Stratify by Event status)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=config.random_seed,
        stratify=df[config.event_col]
    )

    preprocessor = CancerDataPreprocessor(config)
    X_train, time_train, event_train = preprocessor.fit_transform(train_df)
    X_test, time_test, event_test = preprocessor.transform(test_df)

    return {
        "preprocessor": preprocessor,
        "feature_names": preprocessor.feature_names,
        "train": (X_train, time_train, event_train),
        "test": (X_test, time_test, event_test),
        "train_df": train_df,
        "test_df": test_df
    }
