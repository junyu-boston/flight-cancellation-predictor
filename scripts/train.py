#!/Users/jnvy/.venv/bin/python
"""
Main training pipeline script for flight cancellation prediction.
Orchestrates the complete ML pipeline from data loading to model evaluation.
"""

import sys
import logging
from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.models.trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Main training pipeline."""
    logger.info("=" * 50)
    logger.info("Starting Flight Cancellation Prediction Pipeline")
    logger.info("=" * 50)

    # Load configurations
    data_config = load_config('configs/data_config.yaml')
    model_config = load_config('configs/model_config.yaml')
    training_config = load_config('configs/training_config.yaml')

    # Step 1: Load Data
    logger.info("Step 1: Loading data...")
    data_loader_config = {
        'flight_data_paths': [source['path'] for source in data_config['data_sources']['flight_data']],
        'airport_data_path': data_config['data_sources']['airport_data']['path'],
        'lowercase_columns': data_config['processing'].get('lowercase_columns', True),
        'validate_data': data_config['validation'].get('check_data_types', True)
    }

    loader = DataLoader(data_loader_config)
    df, validation_results = loader.load()

    if validation_results.get('warnings'):
        logger.warning(f"Data validation warnings: {validation_results['warnings']}")

    # Step 2: Feature Engineering
    logger.info("Step 2: Engineering features...")
    feature_config = data_config.get('features', {})
    builder = FeatureBuilder(feature_config)
    df = builder.build_features(df)

    # Step 3: Data Splitting
    logger.info("Step 3: Splitting data...")
    split_config = data_config.get('split', {})

    # Sample if configured
    if data_config.get('sampling', {}).get('enabled', False):
        sample_size = data_config['sampling']['sample_size']
        logger.info(f"Sampling {sample_size} records for development")
        df = df.sample(n=sample_size, random_state=split_config.get('random_state', 123))

    train_df, test_df = train_test_split(
        df,
        test_size=split_config.get('test_size', 0.4),
        random_state=split_config.get('random_state', 123),
        stratify=df['cancelled'] if 'cancelled' in df.columns else None
    )

    logger.info(f"Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Check class balance
    if 'cancelled' in train_df.columns:
        cancellation_rate = train_df['cancelled'].value_counts(normalize=True).get(1.0, 0)
        logger.info(f"Cancellation rate in training: {cancellation_rate:.2%}")

    # Step 4: Model Training
    logger.info("Step 4: Training model...")

    # Prepare model configuration
    tabpfnmix_cfg = model_config['model']['tabpfnmix']
    autogluon_cfg = model_config['autogluon']

    trainer_config = {
        'model_name': model_config['model']['name'],
        'model_path_regressor': tabpfnmix_cfg['model_path_regressor'],
        'n_ensembles': tabpfnmix_cfg['n_ensembles'],
        'max_epochs': tabpfnmix_cfg['max_epochs'],
        'eval_metric': autogluon_cfg['eval_metric'],
        'problem_type': autogluon_cfg['problem_type'],
        'verbosity': autogluon_cfg['verbosity'],
        'use_bag_holdout': autogluon_cfg['use_bag_holdout'],
        'num_bag_folds': autogluon_cfg['num_bag_folds'],
        'num_stack_levels': autogluon_cfg['num_stack_levels'],
        'custom_threshold': model_config.get('calibration', {}).get('rare_event_threshold', 0.02),
        'seed': training_config['training']['parameters']['seed']
    }

    trainer = ModelTrainer(trainer_config)

    # Train model
    predictor = trainer.train(
        train_data=train_df,
        validation_data=test_df,
        label='cancelled',
        time_limit=training_config['training']['parameters'].get('max_runtime_seconds')
    )

    # Step 5: Model Evaluation
    logger.info("Step 5: Evaluating model...")

    # Evaluate on test set
    metrics = trainer.evaluate(
        test_df,
        metrics=training_config['monitoring']['metrics']
    )

    logger.info("=" * 50)
    logger.info("Model Performance Summary:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {metric}: {value:.4f}")

    # Step 6: Feature Importance
    logger.info("Step 6: Calculating feature importance...")

    importance = trainer.get_feature_importance(
        test_df.head(1000),
        num_shuffle_sets=model_config['feature_importance']['num_shuffle_sets']
    )

    # Identify noise features
    noise_threshold = model_config['feature_importance']['importance_threshold']
    noise_features = importance[importance['importance'] <= noise_threshold]

    if len(noise_features) > 0:
        logger.warning(f"Found {len(noise_features)} features with importance <= {noise_threshold}")
        logger.warning(f"Consider removing: {list(noise_features.index)[:5]}")

    # Step 7: Apply Custom Threshold
    if model_config.get('calibration', {}).get('use_custom_threshold', False):
        logger.info("Step 7: Evaluating with custom threshold...")

        threshold = model_config['calibration']['rare_event_threshold']
        predictions = trainer.predict(test_df, use_custom_threshold=True)

        from sklearn.metrics import classification_report
        report = classification_report(
            test_df['cancelled'],
            predictions,
            output_dict=True
        )

        logger.info(f"With threshold = {threshold}:")
        logger.info(f"  Precision: {report.get('1', {}).get('precision', 0):.4f}")
        logger.info(f"  Recall: {report.get('1', {}).get('recall', 0):.4f}")
        logger.info(f"  F1: {report.get('1', {}).get('f1-score', 0):.4f}")

    # Step 8: Save Artifacts
    logger.info("Step 8: Saving artifacts...")

    # Save processed data
    output_path = Path(data_config['output']['processed_data'])
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_path / 'train.parquet')
    test_df.to_parquet(output_path / 'test.parquet')

    # Save feature importance
    importance.to_csv(Path(predictor.path) / 'feature_importance.csv')

    logger.info("=" * 50)
    logger.info("Pipeline Complete!")
    logger.info(f"Model saved to: {predictor.path}")
    logger.info("=" * 50)

    return predictor, metrics


if __name__ == '__main__':
    try:
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)

        # Run pipeline
        predictor, metrics = main()

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise