"""
Rolling Form Features for Tournament Outcome Prediction.

This module computes time-aware rolling features with strict leakage prevention.
All features use ONLY past tournament data - the current tournament is never included.

Key Leakage Prevention:
- shift(1) before rolling ensures current tournament excluded
- Course history filtered to dates < current tournament
- All computations respect temporal ordering
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class RollingFormCalculator:
    """
    Computes rolling player form features with strict temporal ordering.

    CRITICAL LEAKAGE PREVENTION:
    - All features computed using ONLY past tournaments
    - Current tournament is NEVER included in rolling windows
    - Uses pandas shift/rolling with min_periods to handle edge cases
    """

    def __init__(self, windows: List[int] = None):
        """
        Initialize calculator with rolling window sizes.

        Args:
            windows: List of window sizes for rolling averages (default: [5, 10])
        """
        self.windows = windows or [5, 10]
        self.sg_metrics = ['sg_total', 'sg_ott', 'sg_app', 'sg_arg', 'sg_putt']

    def compute_rolling_features(
        self,
        df: pd.DataFrame,
        player_col: str = 'player_id',
        date_col: str = 'date',
        metric_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        Compute rolling averages for each player, strictly excluding current row.

        Implementation uses shift(1) before rolling to ensure the current
        tournament is NEVER included in its own features.

        Args:
            df: DataFrame with tournament-level data
            player_col: Column name for player identifier
            date_col: Column name for tournament date
            metric_cols: List of metric columns to compute rolling averages for

        Returns:
            DataFrame with original columns plus rolling features
        """
        if metric_cols is None:
            metric_cols = [col for col in self.sg_metrics if col in df.columns]

        # Ensure date is datetime
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Sort by player and date for proper rolling calculations
        df = df.sort_values([player_col, date_col]).reset_index(drop=True)

        # Compute rolling features for each window size
        for window in self.windows:
            for metric in metric_cols:
                if metric not in df.columns:
                    continue

                feature_name = f'{metric}_last_{window}'

                # CRITICAL: shift(1) excludes current row from its own calculation
                # This prevents label leakage
                df[feature_name] = (
                    df.groupby(player_col)[metric]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
                )

        # Compute standard deviation for consistency measure
        for metric in metric_cols:
            if metric not in df.columns:
                continue

            feature_name = f'{metric}_std_last_10'
            df[feature_name] = (
                df.groupby(player_col)[metric]
                .transform(lambda x: x.shift(1).rolling(10, min_periods=3).std())
            )

        return df

    def compute_form_momentum(
        self,
        df: pd.DataFrame,
        short_window: int = 5,
        long_window: int = 10
    ) -> pd.DataFrame:
        """
        Compute form momentum: how recent form compares to longer-term form.

        Positive momentum = player improving
        Negative momentum = player declining

        Formula: (short_term_avg - long_term_avg) / abs(long_term_avg)

        Args:
            df: DataFrame with rolling features already computed
            short_window: Window size for recent form
            long_window: Window size for long-term form

        Returns:
            DataFrame with momentum features added
        """
        df = df.copy()

        for metric in self.sg_metrics:
            short_col = f'{metric}_last_{short_window}'
            long_col = f'{metric}_last_{long_window}'

            if short_col not in df.columns or long_col not in df.columns:
                continue

            momentum_col = f'{metric}_momentum'

            # Compute momentum with safe division
            df[momentum_col] = np.where(
                df[long_col].abs() > 0.01,  # Avoid division by near-zero
                (df[short_col] - df[long_col]) / df[long_col].abs(),
                0
            )

            # Clip extreme values
            df[momentum_col] = df[momentum_col].clip(-5, 5)

        return df

    def compute_course_history(
        self,
        df: pd.DataFrame,
        player_col: str = 'player_id',
        course_col: str = 'course',
        date_col: str = 'date',
        metric_col: str = 'sg_total'
    ) -> pd.DataFrame:
        """
        Compute player's historical performance at each specific course.

        LEAKAGE PREVENTION: Only includes past appearances at the course,
        never the current tournament.

        Args:
            df: DataFrame with tournament-level data
            player_col: Column name for player identifier
            course_col: Column name for course identifier
            date_col: Column name for tournament date
            metric_col: Metric to average for course history

        Returns:
            DataFrame with course history features added
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Sort for proper temporal processing
        df = df.sort_values([player_col, course_col, date_col]).reset_index(drop=True)

        # Compute course-specific history (excluding current tournament)
        # Using expanding mean with shift to exclude current row
        df['course_avg_sg'] = (
            df.groupby([player_col, course_col])[metric_col]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )

        # Count of past appearances at this course
        df['course_appearances'] = (
            df.groupby([player_col, course_col]).cumcount()
        )

        # Fill NaN for first appearances with global player average
        player_avg = df.groupby(player_col)[metric_col].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
        df['course_avg_sg'] = df['course_avg_sg'].fillna(player_avg)

        # Fill remaining NaN with 0 (no history available)
        df['course_avg_sg'] = df['course_avg_sg'].fillna(0)

        return df

    def compute_all_features(
        self,
        df: pd.DataFrame,
        player_col: str = 'player_id',
        course_col: str = 'course',
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Compute all rolling features in one call.

        Combines:
        1. Rolling averages for all SG metrics
        2. Form momentum
        3. Course history

        Args:
            df: Tournament-level DataFrame
            player_col: Player identifier column
            course_col: Course identifier column
            date_col: Date column

        Returns:
            DataFrame with all rolling features added
        """
        # Step 1: Rolling averages
        df = self.compute_rolling_features(df, player_col, date_col)

        # Step 2: Form momentum
        df = self.compute_form_momentum(df)

        # Step 3: Course history
        df = self.compute_course_history(df, player_col, course_col, date_col)

        return df

    def get_feature_names(self) -> List[str]:
        """Return list of all feature names created by this calculator."""
        features = []

        # Rolling averages
        for window in self.windows:
            for metric in self.sg_metrics:
                features.append(f'{metric}_last_{window}')

        # Standard deviations
        for metric in self.sg_metrics:
            features.append(f'{metric}_std_last_10')

        # Momentum
        for metric in self.sg_metrics:
            features.append(f'{metric}_momentum')

        # Course history
        features.extend(['course_avg_sg', 'course_appearances'])

        return features


class TimeSeriesSplit:
    """
    Time-aware train/test splits for golf tournament data.

    CRITICAL: Golf tournaments have temporal structure.
    - Train on tournaments before date X
    - Test on tournaments after date X
    - Never leak future information into training
    """

    @staticmethod
    def temporal_split(
        df: pd.DataFrame,
        date_col: str = 'date',
        test_size: float = 0.2
    ) -> tuple:
        """
        Split data by date - train on past, test on future.

        Args:
            df: DataFrame with date column
            date_col: Name of date column
            test_size: Fraction of data to use for testing (most recent)

        Returns:
            (train_df, test_df) tuple
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        # Find split point
        n_total = len(df)
        n_train = int(n_total * (1 - test_size))

        # Get the date at split point
        split_date = df.iloc[n_train][date_col]

        # Split by date
        train_df = df[df[date_col] < split_date].copy()
        test_df = df[df[date_col] >= split_date].copy()

        return train_df, test_df

    @staticmethod
    def walk_forward_split(
        df: pd.DataFrame,
        date_col: str = 'date',
        train_weeks: int = 52,
        test_weeks: int = 4,
        step_weeks: int = 4
    ):
        """
        Walk-forward validation generator for backtesting.

        Yields (train, test) pairs moving forward through time.

        Args:
            df: DataFrame with date column
            date_col: Name of date column
            train_weeks: Number of weeks in training window
            test_weeks: Number of weeks in test window
            step_weeks: Number of weeks to step forward each iteration

        Yields:
            (train_df, test_df) tuples
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)

        min_date = df[date_col].min()
        max_date = df[date_col].max()

        train_start = min_date

        while True:
            train_end = train_start + pd.Timedelta(weeks=train_weeks)
            test_end = train_end + pd.Timedelta(weeks=test_weeks)

            if test_end > max_date:
                break

            train_mask = (df[date_col] >= train_start) & (df[date_col] < train_end)
            test_mask = (df[date_col] >= train_end) & (df[date_col] < test_end)

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            if len(train_df) > 0 and len(test_df) > 0:
                yield train_df, test_df

            train_start += pd.Timedelta(weeks=step_weeks)
