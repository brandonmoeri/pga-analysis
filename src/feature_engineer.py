"""
Feature engineering module for Course Fit Model.
Creates course features, player skill profiles, and interaction features.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Engineer features for course fit modeling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def engineer_course_features(self, course_features: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer derived course features from raw course data.
        
        Features:
        - Difficulty metrics
        - Layout characteristics
        - Shot-making demands
        """
        df = course_features.copy()
        
        # Difficulty metrics
        df['overall_difficulty'] = (
            (df['slope_rating'] - 130) / 25 * 0.3 +
            (df['hazard_count'] - 10) / 30 * 0.3 +
            ((50 - df['fairway_width_avg']) / 25) * 0.2 +
            (df['rough_severity'] - 1) / 4 * 0.2
        )
        
        # Par distribution (spread)
        df['par_distribution_diversity'] = (
            (df['par_3'] / df['par']) ** 2 +
            (df['par_4'] / df['par']) ** 2 +
            (df['par_5'] / df['par']) ** 2
        )
        
        # Course setup characteristics
        df['is_tight_course'] = (df['fairway_width_avg'] < 35).astype(int)
        df['is_long_course'] = (df['yardage'] > 7200).astype(int)
        df['is_elevated_course'] = (df['elevation_change'] > 100).astype(int)
        
        # Green complexity
        df['green_challenge'] = (
            (8000 - df['green_size_avg']) / 3000 * 0.5 +
            (df['slope_rating'] - 130) / 25 * 0.5
        )
        
        # Hazard density
        df['hazard_density'] = df['hazard_count'] / (df['yardage'] / 100)
        
        # Approach play difficulty (combination of par 4/5 and hazards)
        df['approach_difficulty'] = (
            (df['par_4'] + df['par_5']) / df['par'] * 0.6 +
            df['hazard_density'] * 0.4
        )
        
        return df
    
    def create_player_skill_profile(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Create normalized player skill profiles.
        Groups skills into key dimensions.
        Handles missing columns and zero variance gracefully.
        """
        df = player_stats.copy()

        # Define expected skill columns
        expected_skills = [
            'driving_distance', 'driving_accuracy', 'greens_in_regulation',
            'scrambling', 'putting_average', 'short_game',
            'off_the_tee', 'approach_play'
        ]

        # Normalize available numeric skills to 0-100 scale
        skill_cols = [col for col in expected_skills if col in df.columns]

        for col in skill_cols:
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                min_val = df[col].min()
                max_val = df[col].max()
                # Handle zero variance (all same values)
                if max_val - min_val > 0:
                    df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val) * 100
                else:
                    df[f'{col}_norm'] = 50.0  # Default to middle

        # Helper to safely get normalized column or default
        def safe_get(col_name: str, default: float = 50.0) -> pd.Series:
            norm_col = f'{col_name}_norm'
            if norm_col in df.columns:
                return df[norm_col].fillna(default)
            return pd.Series([default] * len(df), index=df.index)

        # Aggregate into skill dimensions
        df['accuracy_profile'] = (
            safe_get('driving_accuracy') * 0.5 +
            safe_get('approach_play') * 0.3 +
            safe_get('short_game') * 0.2
        )

        df['distance_profile'] = (
            safe_get('driving_distance') * 0.7 +
            safe_get('off_the_tee') * 0.3
        )

        df['consistency_profile'] = (
            safe_get('greens_in_regulation') * 0.5 +
            safe_get('scrambling') * 0.3 +
            safe_get('short_game') * 0.2
        )

        df['scoring_profile'] = (
            safe_get('putting_average') * 0.4 +
            safe_get('greens_in_regulation') * 0.3 +
            safe_get('scrambling') * 0.3
        )

        return df
    
    def create_player_course_interactions(
        self,
        player_stats: pd.DataFrame,
        course_features: pd.DataFrame,
        tournament_results: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create interaction features between players and courses.
        Merge all three datasets into feature matrix.
        
        Returns:
            Tuple of (features_df, target_series)
        """
        # Engineer individual features
        course_df = self.engineer_course_features(course_features)
        player_df = self.create_player_skill_profile(player_stats)
        
        # Merge datasets
        merged = tournament_results.copy()
        merged = merged.merge(player_df, on='player_id', how='left')
        merged = merged.merge(course_df, on='course_id', how='left')
        
        # Direct compatibility scores
        merged['accuracy_tight_fit'] = (
            merged['accuracy_profile'] * (100 - merged['fairway_width_avg'] * 2) / 100
        )
        
        merged['distance_long_fit'] = (
            merged['distance_profile'] * merged['yardage'] / 7000
        )
        
        merged['consistency_difficulty_fit'] = (
            merged['consistency_profile'] * (100 - merged['overall_difficulty'] * 50) / 100
        )
        
        # Handle short_game_norm potentially missing
        if 'short_game_norm' in merged.columns:
            merged['short_game_tight_fit'] = (
                merged['short_game_norm'] * (100 - merged['green_challenge'] * 50) / 100
            )
        else:
            merged['short_game_tight_fit'] = 50 * (100 - merged['green_challenge'] * 50) / 100
        
        # Cross-interactions
        merged['hazard_experience'] = (
            merged['accuracy_profile'] * (1 - merged['hazard_density'])
        )
        
        merged['elevation_distance_fit'] = (
            merged['distance_profile'] * (100 - merged['elevation_change'] / 2) / 100
        )
        
        # Polynomial features for key interactions
        merged['accuracy_difficulty_squared'] = (
            merged['accuracy_profile'] * merged['overall_difficulty'] ** 2
        )
        
        # Handle greens_in_regulation_norm potentially missing
        if 'greens_in_regulation_norm' in merged.columns:
            merged['gir_hazard_interaction'] = (
                merged['greens_in_regulation_norm'] * (1 - merged['hazard_density'])
            )
        else:
            merged['gir_hazard_interaction'] = 50 * (1 - merged['hazard_density'])

        # Define all possible feature columns
        all_feature_cols = [
            # Player stats
            'driving_distance_norm', 'driving_accuracy_norm', 'greens_in_regulation_norm',
            'scrambling_norm', 'putting_average_norm', 'short_game_norm',
            'off_the_tee_norm', 'approach_play_norm',
            # Skill profiles
            'accuracy_profile', 'distance_profile', 'consistency_profile', 'scoring_profile',
            # Course features
            'yardage', 'par', 'par_3', 'par_4', 'par_5',
            'fairway_width_avg', 'green_size_avg', 'rough_severity',
            'hazard_count', 'elevation_change', 'slope_rating', 'course_rating',
            # Engineered course features
            'overall_difficulty', 'par_distribution_diversity',
            'is_tight_course', 'is_long_course', 'is_elevated_course',
            'green_challenge', 'hazard_density', 'approach_difficulty',
            # Interaction features
            'accuracy_tight_fit', 'distance_long_fit', 'consistency_difficulty_fit',
            'short_game_tight_fit', 'hazard_experience', 'elevation_distance_fit',
            'accuracy_difficulty_squared', 'gir_hazard_interaction',
        ]

        # Only include columns that exist in merged dataframe
        feature_cols = [col for col in all_feature_cols if col in merged.columns]
        
        # Target: score (lower is better, but we'll convert to "fit score" - lower score = better fit)
        X = merged[feature_cols].copy()
        y = merged['score'].copy()
        
        # Add identifiers for later analysis
        X['player_id'] = merged['player_id']
        X['course_id'] = merged['course_id']
        
        return X, y
