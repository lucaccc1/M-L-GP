"""UI configuration aligned 1:1 with the original src/*.py pipeline paths and defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AppConfig:
    """Configuration state for the Streamlit GUI wrapping src/*.py."""

    # Fixed paths used by the original src/*.py scripts
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "models"

    # Step 2: Preprocessing parameter (supported by src.preprocessing.filter_sessions_year)
    start_year: int = 2016

    # Step 4: Training & evaluation parameters
    train_sessions: list[str] = field(default_factory=lambda: ["RAC", "SPR"])
    train_end_year: int = 2024
    val_year: int = 2025
    test_year: int = 2026
    reg_max_iter: int = 200
    reg_learning_rate: float = 0.05
    reg_max_depth: int = 4
    clf_max_iter: int = 150
    clf_learning_rate: float = 0.05
    clf_max_depth: int = 3
    random_state: int = 42
    dnf_rank_penalty: float = 15.0

    # Step 5: Monte Carlo simulation UI controls
    n_sims: int = 10000
    std_dev_multiplier: float = 1.0

    @property
    def raw_path(self) -> Path:
        return Path(self.raw_dir)

    @property
    def processed_path(self) -> Path:
        return Path(self.processed_dir)

    @property
    def models_path(self) -> Path:
        return Path(self.models_dir)

    @property
    def seasons_csv(self) -> Path:
        return self.raw_path / "seasons.csv"

    @property
    def events_csv(self) -> Path:
        return self.raw_path / "events.csv"

    @property
    def sessions_csv(self) -> Path:
        return self.raw_path / "sessions.csv"

    @property
    def results_csv(self) -> Path:
        return self.raw_path / "results.csv"

    @property
    def cleaned_sessions_csv(self) -> Path:
        return self.processed_path / "cleaned_sessions.csv"

    @property
    def dataset_ml_csv(self) -> Path:
        return self.processed_path / "dataset_ml.csv"

    @property
    def training_metrics_json(self) -> Path:
        return self.models_path / "training_metrics.json"

    def model_bundle_path(self, session_type: str) -> Path:
        return self.models_path / f"model_{session_type.lower()}.pkl"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AppConfig:
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
