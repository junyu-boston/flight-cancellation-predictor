"""
Feature engineering module for flight cancellation prediction.
Implements feature creation following ML best practices.
"""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import holidays
from datetime import timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    add_holiday_features: bool = True
    holiday_country: str = "US"
    holiday_years: List[int] = field(default_factory=lambda: [2023, 2024, 2025])
    holiday_proximity_days: int = 3
    add_time_features: bool = True
    add_indicator_features: bool = True
    selected_features: Optional[List[str]] = None
    remove_noise_features: bool = False
    noise_features: Optional[List[str]] = None


class FeatureBuilder:
    """
    Feature builder for flight cancellation prediction.

    This class handles all feature engineering tasks including
    holiday features, time-based features, and custom indicators.
    """

    def __init__(self, config: Union[FeatureConfig, Dict]):
        """
        Initialize FeatureBuilder.

        Args:
            config: FeatureConfig object or dictionary with configuration
        """
        if isinstance(config, dict):
            self.config = FeatureConfig(**config)
        else:
            self.config = config

        self.holidays_calendar = None
        if self.config.add_holiday_features:
            self._initialize_holidays()

        logger.info("FeatureBuilder initialized")

    def _initialize_holidays(self) -> None:
        """Initialize holiday calendar."""
        self.holidays_calendar = holidays.country_holidays(
            self.config.holiday_country,
            years=self.config.holiday_years
        )
        logger.info(f"Initialized {self.config.holiday_country} holidays for years {self.config.holiday_years}")

    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add holiday-related features.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with holiday features
        """
        if not self.config.add_holiday_features:
            return df

        logger.info("Adding holiday features")

        # Convert flight date to datetime if not already
        if 'fl_date_dt' not in df.columns:
            df['fl_date_dt'] = pd.to_datetime(df['fl_date'])

        # Check if date is a holiday
        df['ind_is_holiday'] = df['fl_date_dt'].apply(
            lambda x: 1.0 if self.holidays_calendar.get(x.date()) is not None else 0.0
        )

        # Check if near a holiday
        days = self.config.holiday_proximity_days

        df['ind_is_near_holiday_down'] = df['fl_date_dt'].apply(
            lambda x: any((x.date() - timedelta(days=i)) in self.holidays_calendar for i in range(1, days + 1))
        )

        df['ind_is_near_holiday_up'] = df['fl_date_dt'].apply(
            lambda x: any((x.date() + timedelta(days=i)) in self.holidays_calendar for i in range(1, days + 1))
        )

        df['ind_is_near_holiday'] = np.where(
            (df['ind_is_near_holiday_down'] == True) | (df['ind_is_near_holiday_up'] == True),
            1.0, 0.0
        )

        # Days until/since holiday
        df['days_until_holiday'] = df['fl_date_dt'].apply(self._days_until_next_holiday)
        df['days_since_last_holiday'] = df['fl_date_dt'].apply(self._days_since_last_holiday)

        logger.info(f"Added {6} holiday features")

        return df

    def _days_until_next_holiday(self, flight_date: pd.Timestamp) -> Optional[int]:
        """Calculate days until next holiday."""
        if not self.holidays_calendar:
            return None

        future_holidays = sorted([d for d in self.holidays_calendar if d >= flight_date.date()])

        if not future_holidays:
            return None

        next_holiday = future_holidays[0]
        return (next_holiday - flight_date.date()).days

    def _days_since_last_holiday(self, flight_date: pd.Timestamp) -> Optional[int]:
        """Calculate days since last holiday."""
        if not self.holidays_calendar:
            return None

        past_holidays = sorted([d for d in self.holidays_calendar if d <= flight_date.date()])

        if not past_holidays:
            return None

        last_holiday = past_holidays[-1]
        return (flight_date.date() - last_holiday).days

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with time features
        """
        if not self.config.add_time_features:
            return df

        logger.info("Adding time features")

        # Extract hour from departure and arrival times
        if 'crs_dep_time' in df.columns:
            df['scheduled_hour_of_departure'] = (
                df['crs_dep_time'].astype(str)
                .str.zfill(4)
                .str.slice(0, 2)
                .astype(int)
            )

        if 'crs_arr_time' in df.columns:
            df['scheduled_hour_of_arrival'] = (
                df['crs_arr_time'].astype(str)
                .str.zfill(4)
                .str.slice(0, 2)
                .astype(int)
            )

        logger.info("Added time-based features")

        return df

    def add_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add custom indicator features.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with indicator features
        """
        if not self.config.add_indicator_features:
            return df

        logger.info("Adding indicator features")

        # Summer indicator
        if 'month' in df.columns:
            df['ind_is_summer'] = np.where((df['month'] >= 6) & (df['month'] <= 8), 1, 0)

        # Late departure indicator
        if 'scheduled_hour_of_departure' in df.columns:
            df['ind_late_scheduled_hour_of_departure'] = np.where(
                (df['scheduled_hour_of_departure'] >= 20) & (df['scheduled_hour_of_departure'] <= 22),
                1, 0
            )

        # Late arrival indicator
        if 'scheduled_hour_of_arrival' in df.columns:
            df['ind_late_scheduled_hour_of_arrival'] = np.where(
                (df['scheduled_hour_of_arrival'] >= 21) & (df['scheduled_hour_of_arrival'] <= 23),
                1, 0
            )

        # Early departure indicator
        if 'dep_delay' in df.columns:
            df['ind_early_departure'] = np.where(df['dep_delay'] < 0, 1, 0)

        # Texas long distance indicator
        if 'dest_state_abr' in df.columns and 'distance' in df.columns:
            df['ind_dest_TX_long_dist'] = np.where(
                (df['dest_state_abr'] == 'TX') & (df['distance'] >= 1000),
                1, 0
            )

        # Local airport indicator
        if 'role\n(fy25)' in df.columns:
            df['ind_is_local'] = np.where(df['role\n(fy25)'] == 'Local', 1, 0)

        logger.info("Added indicator features")

        return df

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select relevant features based on configuration.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with selected features
        """
        if not self.config.selected_features:
            return df

        logger.info(f"Selecting {len(self.config.selected_features)} configured features")

        # Keep only features that exist in the dataframe
        available_features = [f for f in self.config.selected_features if f in df.columns]
        missing_features = set(self.config.selected_features) - set(available_features)

        if missing_features:
            logger.warning(f"Features not found in data: {missing_features}")

        df_selected = df[available_features]
        logger.info(f"Selected {len(available_features)} features")

        return df_selected

    def remove_noise_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove features identified as noise.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: DataFrame without noise features
        """
        if not self.config.remove_noise_features or not self.config.noise_features:
            return df

        logger.info(f"Removing {len(self.config.noise_features)} noise features")

        # Remove noise features that exist in the dataframe
        features_to_remove = [f for f in self.config.noise_features if f in df.columns]
        df_cleaned = df.drop(columns=features_to_remove)

        logger.info(f"Removed {len(features_to_remove)} noise features")

        return df_cleaned

    def transform_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data types for optimal processing.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with corrected data types
        """
        logger.info("Transforming data types")

        # Convert target variable to category
        if 'cancelled' in df.columns:
            df['cancelled'] = df['cancelled'].astype('category')

        # Convert string columns
        string_columns = ['op_unique_carrier', 'op_carrier_airline_id', 'op_carrier', 'op_carrier_fl_num']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype('string')

        # Convert origin/dest related columns to string
        for col in df.columns:
            if col.startswith('origin') or col.startswith('dest'):
                df[col] = df[col].astype('str')

        logger.info("Data types transformed")

        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to build all features.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        try:
            # Add holiday features
            df = self.add_holiday_features(df)

            # Add time features
            df = self.add_time_features(df)

            # Add indicator features
            df = self.add_indicator_features(df)

            # Transform data types
            df = self.transform_data_types(df)

            # Select features if configured
            if self.config.selected_features:
                df = self.select_features(df)

            # Remove noise features if configured
            if self.config.remove_noise_features:
                df = self.remove_noise_features(df)

            logger.info(f"Feature engineering complete. Final shape: {df.shape}")

            return df

        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            raise


def engineer_features_from_config(df: pd.DataFrame, config_path: str) -> pd.DataFrame:
    """
    Convenience function to engineer features using a configuration file.

    Args:
        df: Input DataFrame
        config_path: Path to configuration file

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract feature configuration
    feature_config = config.get('features', {})

    builder = FeatureBuilder(feature_config)
    return builder.build_features(df)