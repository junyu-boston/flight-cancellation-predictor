"""
Command-line interface for flight cancellation prediction.
Provides commands for training, evaluation, and prediction.
"""

import click
import logging
import sys
from pathlib import Path
import pandas as pd
import yaml
import json
from typing import Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.loader import DataLoader
from src.features.builder import FeatureBuilder
from src.models.trainer import ModelTrainer
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """Flight Cancellation Prediction CLI."""
    ctx.ensure_object(dict)

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if config:
        with open(config, 'r') as f:
            ctx.obj['config'] = yaml.safe_load(f)
    else:
        ctx.obj['config'] = {}

    click.echo("🛫 Flight Cancellation Prediction System")
    click.echo("=" * 40)


@cli.command()
@click.option('--data-config', type=click.Path(exists=True), default='configs/data_config.yaml',
              help='Data configuration file')
@click.option('--output', '-o', type=click.Path(), default='data/processed',
              help='Output directory for processed data')
@click.pass_context
def prepare_data(ctx, data_config, output):
    """Prepare and process data for training."""
    click.echo("📊 Preparing data...")

    # Load data configuration
    with open(data_config, 'r') as f:
        config = yaml.safe_load(f)

    # Update paths to use data/raw directory
    data_loader_config = {
        'flight_data_paths': [source['path'] for source in config['data_sources']['flight_data']],
        'airport_data_path': config['data_sources']['airport_data']['path'],
        'lowercase_columns': config['processing'].get('lowercase_columns', True),
        'validate_data': True
    }

    # Load data
    loader = DataLoader(data_loader_config)
    df, validation_results = loader.load()

    click.echo(f"✅ Loaded {len(df):,} records")

    # Show validation warnings
    if validation_results.get('warnings'):
        click.echo("⚠️  Data validation warnings:")
        for warning in validation_results['warnings']:
            click.echo(f"  - {warning}")

    # Feature engineering
    feature_config = config.get('features', {})
    builder = FeatureBuilder(feature_config)
    df = builder.build_features(df)

    click.echo(f"✅ Engineered features. Shape: {df.shape}")

    # Split data
    split_config = config.get('split', {})
    train_df, test_df = train_test_split(
        df,
        test_size=split_config.get('test_size', 0.4),
        random_state=split_config.get('random_state', 123),
        stratify=df[split_config.get('stratify', 'cancelled')] if 'cancelled' in df.columns else None
    )

    # Save processed data
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / 'train.parquet'
    test_path = output_path / 'test.parquet'

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    click.echo(f"💾 Saved train data: {train_path} ({len(train_df):,} records)")
    click.echo(f"💾 Saved test data: {test_path} ({len(test_df):,} records)")

    # Save data statistics
    stats = {
        'n_train': len(train_df),
        'n_test': len(test_df),
        'n_features': len(df.columns) - 1,
        'features': list(df.columns),
        'class_balance_train': train_df['cancelled'].value_counts().to_dict() if 'cancelled' in train_df.columns else {},
        'class_balance_test': test_df['cancelled'].value_counts().to_dict() if 'cancelled' in test_df.columns else {}
    }

    stats_path = output_path / 'data_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    click.echo("✅ Data preparation complete!")


@cli.command()
@click.option('--train-data', type=click.Path(exists=True), default='data/processed/train.parquet',
              help='Training data path')
@click.option('--test-data', type=click.Path(exists=True), default='data/processed/test.parquet',
              help='Test data path')
@click.option('--model-config', type=click.Path(exists=True), default='configs/model_config.yaml',
              help='Model configuration file')
@click.option('--label', default='cancelled', help='Target column name')
@click.option('--time-limit', type=int, help='Time limit in seconds')
@click.pass_context
def train(ctx, train_data, test_data, model_config, label, time_limit):
    """Train the model."""
    click.echo("🚀 Training model...")

    # Load data
    train_df = pd.read_parquet(train_data)
    test_df = pd.read_parquet(test_data)

    click.echo(f"📊 Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Load model configuration
    with open(model_config, 'r') as f:
        config = yaml.safe_load(f)

    # Create trainer
    model_cfg = config.get('model', {})
    tabpfnmix_cfg = model_cfg.get('tabpfnmix', {})

    trainer_config = {
        'model_name': model_cfg.get('name', 'TabPFNMix'),
        'model_path_regressor': tabpfnmix_cfg.get('model_path_regressor'),
        'n_ensembles': tabpfnmix_cfg.get('n_ensembles', 1),
        'max_epochs': tabpfnmix_cfg.get('max_epochs', 3),
        'eval_metric': config.get('autogluon', {}).get('eval_metric', 'roc_auc'),
        'problem_type': config.get('autogluon', {}).get('problem_type', 'binary'),
        'verbosity': config.get('autogluon', {}).get('verbosity', 2)
    }

    trainer = ModelTrainer(trainer_config)

    # Train model
    predictor = trainer.train(
        train_data=train_df,
        validation_data=test_df,
        label=label,
        time_limit=time_limit
    )

    click.echo("✅ Training complete!")

    # Evaluate model
    metrics = trainer.evaluate(test_df)

    click.echo("\n📈 Model Performance:")
    click.echo("-" * 40)
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            click.echo(f"{metric}: {value:.4f}")

    # Calculate feature importance
    click.echo("\n🔬 Calculating feature importance...")
    importance = trainer.get_feature_importance(test_df.head(1000))

    # Save feature importance
    importance_path = Path(predictor.path) / 'feature_importance.csv'
    importance.to_csv(importance_path)
    click.echo(f"💾 Feature importance saved to: {importance_path}")

    click.echo(f"\n✅ Model saved to: {predictor.path}")


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to saved model')
@click.option('--test-data', type=click.Path(exists=True), default='data/processed/test.parquet',
              help='Test data path')
@click.option('--metrics', multiple=True, default=['roc_auc', 'precision', 'recall', 'f1'],
              help='Metrics to compute')
@click.pass_context
def evaluate(ctx, model_path, test_data, metrics):
    """Evaluate a trained model."""
    click.echo("📊 Evaluating model...")

    # Load test data
    test_df = pd.read_parquet(test_data)
    click.echo(f"Loaded {len(test_df):,} test samples")

    # Load model
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(model_path)

    # Evaluate
    leaderboard = predictor.leaderboard(
        test_df,
        silent=True,
        extra_metrics=list(metrics)
    )

    click.echo("\n📈 Evaluation Results:")
    click.echo("-" * 40)

    # Display metrics
    best_model = leaderboard.iloc[0]
    for metric in metrics:
        if metric in best_model:
            click.echo(f"{metric}: {best_model[metric]:.4f}")

    # Additional info
    click.echo(f"\nModel: {best_model['model']}")
    click.echo(f"Training time: {best_model.get('fit_time', 0):.1f}s")
    click.echo(f"Inference time: {best_model.get('pred_time_test', 0):.1f}s")


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to saved model')
@click.option('--input-data', type=click.Path(exists=True), required=True,
              help='Input data for prediction')
@click.option('--output', '-o', type=click.Path(), default='predictions.csv',
              help='Output file for predictions')
@click.option('--threshold', type=float, help='Custom probability threshold')
@click.pass_context
def predict(ctx, model_path, input_data, output, threshold):
    """Make predictions on new data."""
    click.echo("🔮 Making predictions...")

    # Load input data
    if input_data.endswith('.parquet'):
        df = pd.read_parquet(input_data)
    elif input_data.endswith('.csv'):
        df = pd.read_csv(input_data)
    else:
        click.echo("❌ Unsupported file format. Use .parquet or .csv")
        return

    click.echo(f"Loaded {len(df):,} samples")

    # Load model
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(model_path)

    # Make predictions
    if threshold:
        # Get probabilities and apply threshold
        proba = predictor.predict_proba(df)
        predictions = (proba.iloc[:, 1] > threshold).astype(int)
        click.echo(f"Applied custom threshold: {threshold}")
    else:
        predictions = predictor.predict(df)

    # Get probabilities
    proba = predictor.predict_proba(df)

    # Create output dataframe
    output_df = pd.DataFrame({
        'prediction': predictions,
        'probability_class_0': proba.iloc[:, 0],
        'probability_class_1': proba.iloc[:, 1]
    })

    # Save predictions
    output_df.to_csv(output, index=False)
    click.echo(f"💾 Predictions saved to: {output}")

    # Show summary
    if 'cancelled' in df.columns:
        from sklearn.metrics import classification_report
        report = classification_report(df['cancelled'], predictions, output_dict=True)
        click.echo("\n📊 Prediction Summary:")
        click.echo(f"Accuracy: {report['accuracy']:.4f}")
        click.echo(f"Precision (class 1): {report.get('1', {}).get('precision', 0):.4f}")
        click.echo(f"Recall (class 1): {report.get('1', {}).get('recall', 0):.4f}")


@cli.command()
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to saved model')
@click.pass_context
def model_info(ctx, model_path):
    """Display model information."""
    click.echo("ℹ️  Model Information")
    click.echo("-" * 40)

    # Load model
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(model_path)

    # Display info
    click.echo(f"Path: {model_path}")
    click.echo(f"Problem type: {predictor.problem_type}")
    click.echo(f"Eval metric: {predictor.eval_metric}")
    click.echo(f"Models: {', '.join(predictor.model_names())}")
    click.echo(f"Best model: {predictor.model_best}")

    # Load metadata if exists
    metadata_path = Path(model_path) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        click.echo("\n📋 Training Metadata:")
        click.echo(f"Training samples: {metadata.get('training_samples', 'N/A')}")
        click.echo(f"Validation samples: {metadata.get('validation_samples', 'N/A')}")
        click.echo(f"Number of features: {metadata.get('n_features', 'N/A')}")
        click.echo(f"Training time: {metadata.get('training_history', {}).get('duration_seconds', 'N/A')}s")

    # Feature importance if exists
    importance_path = Path(model_path) / 'feature_importance.csv'
    if importance_path.exists():
        importance = pd.read_csv(importance_path, index_col=0)
        top_features = importance.nlargest(5, 'importance')

        click.echo("\n🔝 Top 5 Features:")
        for idx, row in top_features.iterrows():
            click.echo(f"  {idx}: {row['importance']:.4f}")


if __name__ == '__main__':
    cli(obj={})