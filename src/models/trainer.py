"""
Model training module for flight cancellation prediction.
Implements AutoGluon TabPFNMix training with best practices.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
from autogluon.tabular import TabularPredictor
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "TabPFNMix"
    model_path_regressor: str = "autogluon/tabpfn-mix-1.0-regressor"
    n_ensembles: int = 1
    max_epochs: int = 3
    eval_metric: str = "roc_auc"
    problem_type: str = "binary"
    verbosity: int = 2
    use_bag_holdout: bool = True
    num_bag_folds: int = 0
    num_stack_levels: int = 0
    save_path: str = "models/artifacts"
    custom_threshold: Optional[float] = 0.02
    seed: int = 42


class ModelTrainer:
    """
    Model trainer for flight cancellation prediction.

    This class handles model training, evaluation, and persistence
    using AutoGluon with TabPFNMix.
    """

    def __init__(self, config: Union[ModelConfig, Dict]):
        """
        Initialize ModelTrainer.

        Args:
            config: ModelConfig object or dictionary with configuration
        """
        if isinstance(config, dict):
            self.config = ModelConfig(**config)
        else:
            self.config = config

        self.predictor = None
        self.training_history = {}
        self.model_metadata = {}

        # Set random seed for reproducibility
        np.random.seed(self.config.seed)

        logger.info(f"ModelTrainer initialized with {self.config.model_name}")

    def _prepare_hyperparameters(self) -> Dict:
        """
        Prepare hyperparameters for AutoGluon.

        Returns:
            Dict: Hyperparameters configuration
        """
        if self.config.model_name.upper() == "TABPFNMIX":
            tabpfnmix_cfg = {
                "model_path_regressor": self.config.model_path_regressor,
                "n_ensembles": self.config.n_ensembles,
                "max_epochs": self.config.max_epochs
            }
            hyperparameters = {"TABPFNMIX": [tabpfnmix_cfg]}
        else:
            # Default to AutoGluon's automatic selection
            hyperparameters = "default"

        logger.debug(f"Hyperparameters: {hyperparameters}")
        return hyperparameters

    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        label: str = "cancelled",
        time_limit: Optional[int] = None
    ) -> TabularPredictor:
        """
        Train the model.

        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            label: Target column name
            time_limit: Time limit in seconds

        Returns:
            TabularPredictor: Trained model
        """
        logger.info(f"Starting model training with {len(train_data):,} samples")

        # Record training start time
        start_time = datetime.now()
        self.training_history['start_time'] = start_time.isoformat()

        # Prepare hyperparameters
        hyperparameters = self._prepare_hyperparameters()

        # Create save path
        save_path = Path(self.config.save_path) / f"ag_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_path.mkdir(parents=True, exist_ok=True)

        # Initialize predictor
        self.predictor = TabularPredictor(
            label=label,
            eval_metric=self.config.eval_metric,
            problem_type=self.config.problem_type,
            path=str(save_path),
            verbosity=self.config.verbosity
        )

        # Prepare ag_args_fit
        ag_args_fit = {
            "ag.max_memory_usage_ratio": 5
        }

        # Train model
        self.predictor.fit(
            train_data=train_data,
            tuning_data=validation_data,
            hyperparameters=hyperparameters,
            presets=None,
            use_bag_holdout=self.config.use_bag_holdout,
            num_bag_folds=self.config.num_bag_folds,
            num_stack_levels=self.config.num_stack_levels,
            time_limit=time_limit,
            ag_args_fit=ag_args_fit
        )

        # Record training end time
        end_time = datetime.now()
        self.training_history['end_time'] = end_time.isoformat()
        self.training_history['duration_seconds'] = (end_time - start_time).total_seconds()

        # Store model metadata
        self._store_metadata(train_data, validation_data)

        logger.info(f"Training completed in {self.training_history['duration_seconds']:.1f} seconds")
        logger.info(f"Model saved to: {save_path}")

        return self.predictor

    def _store_metadata(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame]) -> None:
        """Store model metadata."""
        self.model_metadata = {
            'model_name': self.config.model_name,
            'config': asdict(self.config),
            'training_samples': len(train_data),
            'validation_samples': len(validation_data) if validation_data is not None else 0,
            'features': list(train_data.columns),
            'n_features': len(train_data.columns) - 1,  # Exclude target
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat()
        }

        # Save metadata to file
        metadata_path = Path(self.predictor.path) / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

        logger.debug(f"Metadata saved to {metadata_path}")

    def evaluate(
        self,
        test_data: pd.DataFrame,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate the trained model.

        Args:
            test_data: Test data
            metrics: List of metrics to compute

        Returns:
            Dict: Evaluation metrics
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call train() first.")

        logger.info(f"Evaluating model on {len(test_data):,} samples")

        # Default metrics
        if metrics is None:
            metrics = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']

        # Get leaderboard with metrics
        leaderboard = self.predictor.leaderboard(
            test_data,
            silent=True,
            extra_metrics=metrics
        )

        # Extract best model metrics
        best_model_metrics = leaderboard.iloc[0].to_dict()

        # Log metrics
        for metric, value in best_model_metrics.items():
            if metric in metrics:
                logger.info(f"{metric}: {value:.4f}")

        return best_model_metrics

    def predict(
        self,
        data: pd.DataFrame,
        use_custom_threshold: bool = False
    ) -> np.ndarray:
        """
        Make predictions.

        Args:
            data: Input data
            use_custom_threshold: Whether to use custom threshold

        Returns:
            np.ndarray: Predictions
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call train() first.")

        if use_custom_threshold and self.config.custom_threshold:
            # Get probabilities
            proba = self.predictor.predict_proba(data)
            # Apply custom threshold
            predictions = (proba.iloc[:, 1] > self.config.custom_threshold).astype(int)
            logger.info(f"Applied custom threshold: {self.config.custom_threshold}")
        else:
            predictions = self.predictor.predict(data)

        return predictions

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get prediction probabilities.

        Args:
            data: Input data

        Returns:
            pd.DataFrame: Prediction probabilities
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call train() first.")

        return self.predictor.predict_proba(data)

    def get_feature_importance(
        self,
        data: pd.DataFrame,
        num_shuffle_sets: int = 5
    ) -> pd.DataFrame:
        """
        Calculate feature importance.

        Args:
            data: Data for importance calculation
            num_shuffle_sets: Number of permutation sets

        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call train() first.")

        logger.info("Calculating feature importance")

        importance = self.predictor.feature_importance(
            data,
            num_shuffle_sets=num_shuffle_sets,
            silent=False
        )

        # Log top features
        top_features = importance.nlargest(5, 'importance')
        logger.info(f"Top 5 features:\n{top_features}")

        return importance

    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the trained model.

        Args:
            path: Save path (optional)

        Returns:
            str: Path where model was saved
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call train() first.")

        if path is None:
            path = self.predictor.path

        self.predictor.save(path)
        logger.info(f"Model saved to: {path}")

        return path

    def load_model(self, path: str) -> TabularPredictor:
        """
        Load a saved model.

        Args:
            path: Model path

        Returns:
            TabularPredictor: Loaded model
        """
        self.predictor = TabularPredictor.load(path)
        logger.info(f"Model loaded from: {path}")

        # Load metadata if exists
        metadata_path = Path(path) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
            logger.debug("Metadata loaded")

        return self.predictor

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dict: Model information
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet.")

        info = {
            'models': self.predictor.model_names(),
            'best_model': self.predictor.model_best,
            'path': self.predictor.path,
            'problem_type': self.predictor.problem_type,
            'eval_metric': self.predictor.eval_metric,
            'metadata': self.model_metadata
        }

        return info


def train_model_from_config(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    config_path: str
) -> Tuple[TabularPredictor, Dict[str, float]]:
    """
    Convenience function to train model using a configuration file.

    Args:
        train_data: Training data
        validation_data: Validation data
        config_path: Path to configuration file

    Returns:
        Tuple: Trained predictor and evaluation metrics
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract model configuration
    model_config = config.get('model', {})

    # Create trainer
    trainer = ModelTrainer(model_config)

    # Train model
    predictor = trainer.train(train_data, validation_data)

    # Evaluate model
    metrics = trainer.evaluate(validation_data)

    return predictor, metrics