"""
Data loading module for flight cancellation prediction.
Follows ML best practices for data handling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Data configuration class."""
    flight_data_paths: List[str]
    airport_data_path: str
    lowercase_columns: bool = True
    validate_data: bool = True


class DataLoader:
    """
    Data loader for flight cancellation prediction.

    This class handles loading and initial processing of flight and airport data,
    following ML engineering best practices.
    """

    def __init__(self, config: Union[DataConfig, Dict]):
        """
        Initialize DataLoader.

        Args:
            config: DataConfig object or dictionary with configuration
        """
        if isinstance(config, dict):
            self.config = DataConfig(**config)
        else:
            self.config = config

        self._validate_paths()
        logger.info("DataLoader initialized successfully")

    def _validate_paths(self) -> None:
        """Validate that all data paths exist."""
        for path in self.config.flight_data_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Flight data file not found: {path}")

        if not Path(self.config.airport_data_path).exists():
            raise FileNotFoundError(f"Airport data file not found: {self.config.airport_data_path}")

    def load_flight_data(self) -> pd.DataFrame:
        """
        Load and combine flight data from multiple parquet files.

        Returns:
            pd.DataFrame: Combined flight data
        """
        logger.info(f"Loading {len(self.config.flight_data_paths)} flight data files")

        dataframes = []
        for path in self.config.flight_data_paths:
            logger.debug(f"Loading {path}")
            df = pd.read_parquet(path)
            dataframes.append(df)
            logger.info(f"Loaded {len(df):,} records from {Path(path).name}")

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Combined {len(combined_df):,} total flight records")

        return combined_df

    def load_airport_data(self) -> pd.DataFrame:
        """
        Load airport reference data.

        Returns:
            pd.DataFrame: Airport data
        """
        logger.info(f"Loading airport data from {self.config.airport_data_path}")

        # Determine file format and load accordingly
        file_path = Path(self.config.airport_data_path)

        if file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Loaded {len(df):,} airport records")
        return df

    def merge_data(self, flight_df: pd.DataFrame, airport_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge flight and airport data.

        Args:
            flight_df: Flight data
            airport_df: Airport data

        Returns:
            pd.DataFrame: Merged dataset
        """
        logger.info("Merging flight and airport data")

        # Perform left join on origin airport
        merged_df = pd.merge(
            flight_df,
            airport_df,
            how='left',
            left_on='ORIGIN',
            right_on='LocID'
        )

        # Lowercase column names if configured
        if self.config.lowercase_columns:
            merged_df.columns = merged_df.columns.str.lower()
            logger.debug("Column names converted to lowercase")

        logger.info(f"Merged dataset contains {len(merged_df):,} records with {len(merged_df.columns)} columns")

        return merged_df

    def validate_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate loaded data for quality issues.

        Args:
            df: DataFrame to validate

        Returns:
            Dict: Validation results
        """
        validation_results = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'missing_values': {},
            'duplicates': 0,
            'data_types': {},
            'warnings': []
        }

        # Check for missing values
        missing = df.isnull().sum()
        validation_results['missing_values'] = missing[missing > 0].to_dict()

        if validation_results['missing_values']:
            logger.warning(f"Found missing values in {len(validation_results['missing_values'])} columns")
            validation_results['warnings'].append("Missing values detected")

        # Check for duplicates
        n_duplicates = df.duplicated().sum()
        validation_results['duplicates'] = n_duplicates

        if n_duplicates > 0:
            logger.warning(f"Found {n_duplicates:,} duplicate rows")
            validation_results['warnings'].append(f"{n_duplicates} duplicate rows found")

        # Check data types
        validation_results['data_types'] = df.dtypes.astype(str).to_dict()

        # Check for target variable
        if 'cancelled' not in df.columns:
            logger.error("Target variable 'cancelled' not found in data")
            validation_results['warnings'].append("Target variable 'cancelled' missing")
        else:
            # Check class balance
            class_counts = df['cancelled'].value_counts()
            class_balance = class_counts.min() / class_counts.max()
            validation_results['class_balance'] = class_balance

            if class_balance < 0.1:
                logger.warning(f"Severe class imbalance detected: {class_balance:.2%}")
                validation_results['warnings'].append(f"Class imbalance: {class_balance:.2%}")

        return validation_results

    def load(self) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Main method to load all data.

        Returns:
            Tuple[pd.DataFrame, Dict]: Loaded data and validation results
        """
        try:
            # Load flight data
            flight_df = self.load_flight_data()

            # Load airport data
            airport_df = self.load_airport_data()

            # Merge datasets
            merged_df = self.merge_data(flight_df, airport_df)

            # Validate if configured
            validation_results = {}
            if self.config.validate_data:
                validation_results = self.validate_data(merged_df)

                if validation_results['warnings']:
                    logger.warning(f"Data validation completed with {len(validation_results['warnings'])} warnings")
                else:
                    logger.info("Data validation completed successfully")

            return merged_df, validation_results

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise


def load_data_from_config(config_path: str) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Convenience function to load data using a configuration file.

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple[pd.DataFrame, Dict]: Loaded data and validation results
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract data configuration
    data_config = {
        'flight_data_paths': [source['path'] for source in config['data_sources']['flight_data']],
        'airport_data_path': config['data_sources']['airport_data']['path'],
        'lowercase_columns': config['processing'].get('lowercase_columns', True),
        'validate_data': config['validation'].get('check_data_types', True)
    }

    loader = DataLoader(data_config)
    return loader.load()