"""
Data preprocessing and cleaning utilities for PGA data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class PGADataPreprocessor:
    """Clean and preprocess raw PGA data."""

    def __init__(self):
        self.player_name_mappings = self._get_player_name_mappings()
        self.course_name_mappings = self._get_course_name_mappings()

    def _get_player_name_mappings(self) -> Dict[str, str]:
        """Known player name variations to standardize."""
        return {
            'J. Thomas': 'Justin Thomas',
            'R. McIlroy': 'Rory McIlroy',
            'J. Spieth': 'Jordan Spieth',
            'B. Koepka': 'Brooks Koepka',
            'D. Johnson': 'Dustin Johnson',
            'J. Rahm': 'Jon Rahm',
            'C. Morikawa': 'Collin Morikawa',
            'X. Schauffele': 'Xander Schauffele',
            'P. Cantlay': 'Patrick Cantlay',
            'S. Scheffler': 'Scottie Scheffler',
            'V. Hovland': 'Viktor Hovland',
            'T. Finau': 'Tony Finau',
            'M. Fitzpatrick': 'Matt Fitzpatrick',
            'W. Clark': 'Wyndham Clark',
            'B. Horschel': 'Billy Horschel',
        }

    def _get_course_name_mappings(self) -> Dict[str, str]:
        """Known course name variations to standardize."""
        return {
            'TPC Sawgrass': 'TPC Sawgrass',
            'TPC Sawgrass (PLAYERS Stadium)': 'TPC Sawgrass',
            'Pebble Beach GL': 'Pebble Beach GL',
            'Pebble Beach Golf Links': 'Pebble Beach GL',
            'Torrey Pines GC (South)': 'Torrey Pines (South)',
            'Torrey Pines South': 'Torrey Pines (South)',
            'Augusta National GC': 'Augusta National',
            'Augusta National Golf Club': 'Augusta National',
            'Riviera CC (Los Angeles)': 'Riviera CC',
            'Riviera Country Club': 'Riviera CC',
        }

    def clean_player_names(self, df: pd.DataFrame, col: str = 'Player_initial_last') -> pd.DataFrame:
        """
        Standardize player names for consistent matching.

        Args:
            df: DataFrame with player data
            col: Column name containing player names

        Returns:
            DataFrame with cleaned player names
        """
        df = df.copy()

        if col not in df.columns:
            return df

        # Basic cleaning
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

        # Apply known name mappings
        df[col] = df[col].replace(self.player_name_mappings)

        return df

    def clean_course_names(self, df: pd.DataFrame, col: str = 'course') -> pd.DataFrame:
        """
        Standardize course names for consistent matching.

        Args:
            df: DataFrame with course data
            col: Column name containing course names

        Returns:
            DataFrame with cleaned course names
        """
        df = df.copy()

        if col not in df.columns:
            return df

        # Basic cleaning
        df[col] = df[col].str.strip()

        # Apply known course mappings
        df[col] = df[col].replace(self.course_name_mappings)

        return df

    def handle_missing_strokes_gained(
        self,
        df: pd.DataFrame,
        strategy: str = 'impute'
    ) -> pd.DataFrame:
        """
        Handle missing Strokes Gained data.

        Args:
            df: DataFrame with strokes gained columns
            strategy: 'impute' to fill with 0, 'drop' to remove rows

        Returns:
            DataFrame with handled missing values
        """
        sg_cols = ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']
        available_cols = [col for col in sg_cols if col in df.columns]

        if not available_cols:
            return df

        df = df.copy()

        if strategy == 'impute':
            # Impute with 0 (field average)
            for col in available_cols:
                df[col] = df[col].fillna(0)
        elif strategy == 'drop':
            # Drop records without SG:Total
            if 'sg_total' in df.columns:
                df = df.dropna(subset=['sg_total'])

        return df

    def filter_valid_tournaments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to valid PGA Tour events.
        Excludes exhibitions, pro-ams, etc.

        Args:
            df: DataFrame with tournament data

        Returns:
            Filtered DataFrame
        """
        if 'tournament_name' not in df.columns:
            return df

        exclude_patterns = [
            'Pro-Am',
            'Exhibition',
            'Challenge',
            'Qualifier',
            'Monday',
            'Celebrity',
        ]

        mask = ~df['tournament_name'].str.contains(
            '|'.join(exclude_patterns),
            case=False,
            na=False
        )
        return df[mask]

    def filter_minimum_tournaments(
        self,
        df: pd.DataFrame,
        min_tournaments: int = 5,
        player_col: str = 'Player_initial_last'
    ) -> pd.DataFrame:
        """
        Filter to players with minimum tournament appearances.

        Args:
            df: DataFrame with tournament data
            min_tournaments: Minimum tournaments required
            player_col: Player identifier column

        Returns:
            Filtered DataFrame
        """
        if player_col not in df.columns:
            return df

        player_counts = df[player_col].value_counts()
        valid_players = player_counts[player_counts >= min_tournaments].index
        return df[df[player_col].isin(valid_players)]

    def aggregate_player_by_course(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple appearances by same player at same course.

        Args:
            df: DataFrame with tournament data

        Returns:
            Aggregated DataFrame
        """
        player_col = 'Player_initial_last' if 'Player_initial_last' in df.columns else 'player'
        course_col = 'course' if 'course' in df.columns else 'course_name'

        agg_dict = {
            'sg_total': 'mean',
            'sg_putt': 'mean',
            'sg_arg': 'mean',
            'sg_app': 'mean',
            'sg_ott': 'mean',
        }

        # Only include columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        if 'gir' in df.columns:
            agg_dict['gir'] = 'mean'
        if 'n_rounds' in df.columns:
            agg_dict['n_rounds'] = 'sum'

        agg_df = df.groupby([player_col, course_col]).agg(agg_dict).reset_index()

        # Add count of appearances
        agg_df['appearances'] = df.groupby([player_col, course_col]).size().values

        return agg_df

    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Generate data quality report.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with quality metrics
        """
        player_col = 'Player_initial_last' if 'Player_initial_last' in df.columns else 'player_id'
        course_col = 'course' if 'course' in df.columns else 'course_id'

        report = {
            'total_records': len(df),
            'unique_players': df[player_col].nunique() if player_col in df.columns else 0,
            'unique_courses': df[course_col].nunique() if course_col in df.columns else 0,
            'date_range': None,
            'missing_values': {},
        }

        # Check date range
        if 'date' in df.columns:
            report['date_range'] = f"{df['date'].min()} to {df['date'].max()}"
        elif 'season' in df.columns:
            report['date_range'] = f"Seasons {df['season'].min()} to {df['season'].max()}"

        # Check missing values for key columns
        key_cols = ['sg_total', 'sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'gir']
        for col in key_cols:
            if col in df.columns:
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                report['missing_values'][col] = {
                    'count': int(missing),
                    'percent': round(missing_pct, 2)
                }

        return report

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        min_tournaments: int = 3,
        sg_strategy: str = 'impute'
    ) -> pd.DataFrame:
        """
        Run full preprocessing pipeline.

        Args:
            df: Raw DataFrame
            min_tournaments: Minimum tournaments per player
            sg_strategy: Strategy for missing SG data

        Returns:
            Preprocessed DataFrame
        """
        # Determine column names
        player_col = 'Player_initial_last' if 'Player_initial_last' in df.columns else 'player'
        course_col = 'course' if 'course' in df.columns else 'course_name'

        # Clean names
        df = self.clean_player_names(df, player_col)
        df = self.clean_course_names(df, course_col)

        # Filter tournaments
        df = self.filter_valid_tournaments(df)

        # Handle missing data
        df = self.handle_missing_strokes_gained(df, strategy=sg_strategy)

        # Filter players with enough data
        df = self.filter_minimum_tournaments(df, min_tournaments, player_col)

        return df


def print_data_quality_report(report: Dict) -> None:
    """Print formatted data quality report."""
    print("\n" + "=" * 50)
    print("DATA QUALITY REPORT")
    print("=" * 50)

    print(f"\nTotal Records: {report['total_records']:,}")
    print(f"Unique Players: {report['unique_players']}")
    print(f"Unique Courses: {report['unique_courses']}")

    if report['date_range']:
        print(f"Date Range: {report['date_range']}")

    if report['missing_values']:
        print("\nMissing Values:")
        for col, info in report['missing_values'].items():
            print(f"  {col}: {info['count']:,} ({info['percent']:.1f}%)")

    print("=" * 50)
