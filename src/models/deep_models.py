"""
Deep Learning Survival Prediction Models (DeepSurv in PyTorch)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
from src.config import StudyConfig


class SurvivalDataset(Dataset):
    def __init__(self, X: np.ndarray, time: np.ndarray, event: np.ndarray):
        # Sort descending by time for efficient Cox partial likelihood calculation
        order = np.argsort(-time)
        self.X = torch.tensor(X[order], dtype=torch.float32)
        self.time = torch.tensor(time[order], dtype=torch.float32)
        # Event: 1 is primary outcome (chronic disease)
        self.event = torch.tensor((event[order] == 1).astype(int), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.time[idx], self.event[idx]


class DeepSurvNetwork(nn.Module):
    def __init__(self, in_features: int, hidden_dims: List[int], dropout_rate: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        # Log-hazard output layer (scalar)
        layers.append(nn.Linear(prev_dim, 1, bias=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def negative_log_partial_likelihood(log_hazard: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    Breslow approximation of Cox negative log partial likelihood
    Expects input ordered descending by survival time.
    """
    hazard = torch.exp(log_hazard)
    # Cumulative sum from end (at risk set)
    risk_set_sum = torch.cumsum(hazard.flip(0), dim=0).flip(0)
    log_risk = torch.log(risk_set_sum + 1e-7)
    uncensored_likelihood = (log_hazard - log_risk) * event
    num_events = torch.sum(event)
    if num_events == 0:
        return torch.tensor(0.0, requires_grad=True)
    return -torch.sum(uncensored_likelihood) / num_events


class DeepSurvTrainer:
    def __init__(self, in_features: int, config: StudyConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.model = DeepSurvNetwork(
            in_features=in_features,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate
        ).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    def fit(self, X_train: np.ndarray, time_train: np.ndarray, event_train: np.ndarray):
        train_dataset = SurvivalDataset(X_train, time_train, event_train)
        # DeepSurv requires full-batch or large batch sorted by time for accurate risk set
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

        self.model.train()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            for batch_x, _, batch_e in train_loader:
                batch_x = batch_x.to(self.device)
                batch_e = batch_e.to(self.device).unsqueeze(1)

                self.optimizer.zero_grad()
                log_hazard = self.model(batch_x)
                loss = negative_log_partial_likelihood(log_hazard, batch_e)
                loss.backward()
                self.optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                break

        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            risk = self.model(x_tensor).cpu().numpy().flatten()
        return risk
