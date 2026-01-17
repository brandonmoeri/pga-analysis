"""
Data loading and preprocessing module for PGA Course Fit Model.
Handles loading player, course, and tournament data from real PGA sources.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import warnings


class RealDataLoader:
    """Load and preprocess real PGA data from Kaggle and supplemental sources."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"

    def load_kaggle_data(self, min_year: int = 2020, max_year: int = 2024) -> pd.DataFrame:
        """
        Load PGA tour data from Kaggle dataset.
        Filters to specified year range.

        Args:
            min_year: Minimum season year to include
            max_year: Maximum season year to include

        Returns:
            DataFrame with filtered tournament data
        """
        kaggle_file = self.raw_dir / "kaggle" / "ASA All PGA Raw Data - Tourn Level.csv"

        if not kaggle_file.exists():
            raise FileNotFoundError(
                f"Kaggle data not found at {kaggle_file}. "
                "Download from: https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022"
            )

        df = pd.read_csv(kaggle_file)

        # Filter by season year
        if 'season' in df.columns:
            df = df[(df['season'] >= min_year) & (df['season'] <= max_year)]
        elif 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['season'] = df['date'].dt.year
            df = df[(df['season'] >= min_year) & (df['season'] <= max_year)]

        print(f"  Loaded {len(df)} tournament records ({min_year}-{max_year})")
        return df

    def load_course_characteristics(self) -> pd.DataFrame:
        """Load course characteristics from curated CSV."""
        course_file = self.raw_dir / "courses" / "course_characteristics.csv"

        if course_file.exists():
            return pd.read_csv(course_file)
        else:
            warnings.warn("Course characteristics file not found. Using defaults.")
            return pd.DataFrame()

    def create_player_stats(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate player statistics from tournament-level data.
        Maps Kaggle columns to expected feature engineering columns.

        Args:
            raw_df: Raw Kaggle tournament data

        Returns:
            DataFrame with aggregated player stats
        """
        # Handle column name variations
        player_col = 'Player_initial_last' if 'Player_initial_last' in raw_df.columns else 'player'

        # Filter out rows with missing strokes gained data
        df = raw_df.copy()
        sg_cols = ['sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'sg_total']
        available_sg = [col for col in sg_cols if col in df.columns]

        if available_sg:
            # Keep rows that have at least sg_total
            if 'sg_total' in df.columns:
                df = df[df['sg_total'].notna()]

        # Group by player and calculate season averages
        agg_dict = {}

        if 'sg_ott' in df.columns:
            agg_dict['sg_ott'] = 'mean'
        if 'sg_app' in df.columns:
            agg_dict['sg_app'] = 'mean'
        if 'sg_arg' in df.columns:
            agg_dict['sg_arg'] = 'mean'
        if 'sg_putt' in df.columns:
            agg_dict['sg_putt'] = 'mean'
        if 'sg_t2g' in df.columns:
            agg_dict['sg_t2g'] = 'mean'
        if 'sg_total' in df.columns:
            agg_dict['sg_total'] = 'mean'
        if 'gir' in df.columns:
            agg_dict['gir'] = 'mean'

        # Count tournaments played
        agg_dict['tournament_name' if 'tournament_name' in df.columns else 'course'] = 'count'

        player_agg = df.groupby(player_col).agg(agg_dict).reset_index()
        player_agg = player_agg.rename(columns={player_col: 'player_id'})

        # Rename the count column
        count_col = 'tournament_name' if 'tournament_name' in player_agg.columns else 'course'
        if count_col in player_agg.columns:
            player_agg = player_agg.rename(columns={count_col: 'tournaments_played'})

        # Build player stats DataFrame
        player_stats = pd.DataFrame({
            'player_id': player_agg['player_id'],
        })

        # Map strokes gained columns
        if 'sg_ott' in player_agg.columns:
            player_stats['off_the_tee'] = player_agg['sg_ott'].fillna(0)
        else:
            player_stats['off_the_tee'] = 0

        if 'sg_app' in player_agg.columns:
            player_stats['approach_play'] = player_agg['sg_app'].fillna(0)
        else:
            player_stats['approach_play'] = 0

        if 'sg_arg' in player_agg.columns:
            player_stats['short_game'] = player_agg['sg_arg'].fillna(0)
        else:
            player_stats['short_game'] = 0

        # Putting average: invert SG:Putt (positive SG = good, but traditional putting avg: lower is better)
        if 'sg_putt' in player_agg.columns:
            # Scale to traditional putting average range (1.65 - 1.95)
            player_stats['putting_average'] = 1.75 - (player_agg['sg_putt'].fillna(0) * 0.05)
        else:
            player_stats['putting_average'] = 1.75

        if 'gir' in player_agg.columns:
            player_stats['greens_in_regulation'] = player_agg['gir'].fillna(65)
        else:
            player_stats['greens_in_regulation'] = 65

        # Keep sg_total for reference
        if 'sg_total' in player_agg.columns:
            player_stats['sg_total'] = player_agg['sg_total'].fillna(0)

        if 'tournaments_played' in player_agg.columns:
            player_stats['tournaments_played'] = player_agg['tournaments_played']

        # Derive estimated columns from strokes gained
        player_stats['driving_distance'] = self._estimate_driving_distance(player_stats['off_the_tee'])
        player_stats['driving_accuracy'] = self._estimate_driving_accuracy(player_stats['off_the_tee'])
        player_stats['scrambling'] = self._estimate_scrambling(player_stats['short_game'])

        print(f"  Created stats for {len(player_stats)} players")
        return player_stats

    def _estimate_driving_distance(self, sg_ott: pd.Series) -> pd.Series:
        """
        Estimate driving distance from SG:OTT.
        PGA Tour average is ~295 yards. Each 0.1 SG:OTT ~ 2-3 yards.
        """
        return 295 + (sg_ott * 25)

    def _estimate_driving_accuracy(self, sg_ott: pd.Series) -> pd.Series:
        """
        Estimate driving accuracy from SG:OTT.
        PGA Tour average is ~60%. Better SG:OTT often correlates with better accuracy.
        """
        return 60 + (sg_ott * 4)

    def _estimate_scrambling(self, sg_arg: pd.Series) -> pd.Series:
        """
        Estimate scrambling percentage from SG:ARG.
        PGA Tour average is ~58%.
        """
        return 58 + (sg_arg * 6)

    def create_course_features(
        self,
        raw_df: pd.DataFrame,
        course_chars: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create course features by merging tournament data with characteristics.

        Args:
            raw_df: Raw tournament data
            course_chars: Course characteristics DataFrame

        Returns:
            DataFrame with course features
        """
        # Get unique courses from tournament data
        course_col = 'course' if 'course' in raw_df.columns else 'course_name'
        unique_courses = raw_df[course_col].unique()

        courses = pd.DataFrame({
            'course_id': unique_courses,
            'course_name': unique_courses  # Use same as ID initially
        })

        # Merge with characteristics if available
        if not course_chars.empty:
            courses = courses.merge(
                course_chars,
                on='course_id',
                how='left',
                suffixes=('', '_char')
            )
            # Use characteristics course_name if available
            if 'course_name_char' in courses.columns:
                courses['course_name'] = courses['course_name_char'].fillna(courses['course_name'])
                courses = courses.drop(columns=['course_name_char'])

        # Fill missing values with PGA Tour averages
        defaults = {
            'yardage': 7200,
            'par': 72,
            'par_3': 4,
            'par_4': 10,
            'par_5': 4,
            'fairway_width_avg': 32,
            'green_size_avg': 6000,
            'rough_severity': 3.0,
            'hazard_count': 30,
            'elevation_change': 50,
            'slope_rating': 140,
            'course_rating': 74.5,
        }

        for col, default in defaults.items():
            if col not in courses.columns:
                courses[col] = default
            else:
                courses[col] = courses[col].fillna(default)

        print(f"  Created features for {len(courses)} courses ({len(course_chars)} with detailed characteristics)")
        return courses

    def create_tournament_results(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw tournament data into player-course results.

        Args:
            raw_df: Raw Kaggle tournament data

        Returns:
            DataFrame with tournament results
        """
        player_col = 'Player_initial_last' if 'Player_initial_last' in raw_df.columns else 'player'
        course_col = 'course' if 'course' in raw_df.columns else 'course_name'

        # Keep all records (made cut or not) but filter for those with data
        df = raw_df.copy()

        # Filter out records without strokes gained data
        if 'sg_total' in df.columns:
            df = df[df['sg_total'].notna()]

        # Build results DataFrame
        tournament_results = pd.DataFrame({
            'player_id': df[player_col],
            'course_id': df[course_col],
            'score': self._calculate_score(df),
            'rounds': df['n_rounds'] if 'n_rounds' in df.columns else 4,
        })

        # Add optional columns if available
        if 'season' in df.columns:
            tournament_results['season'] = df['season']
        if 'sg_total' in df.columns:
            tournament_results['sg_total'] = df['sg_total']
        if 'pos' in df.columns:
            tournament_results['position'] = df['pos']
        if 'made_cut' in df.columns:
            tournament_results['made_cut'] = df['made_cut']
        if 'tournament_name' in df.columns:
            tournament_results['tournament_name'] = df['tournament_name']

        print(f"  Created {len(tournament_results)} tournament results")
        return tournament_results

    def _calculate_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate score metric for modeling.
        Uses -SG:Total as score (invert so higher = worse, like golf scoring).

        Args:
            df: DataFrame with strokes gained data

        Returns:
            Series with score values
        """
        if 'sg_total' in df.columns:
            # Invert and scale: higher score = worse performance
            # Add baseline to keep positive
            return 72 - (df['sg_total'].fillna(0) * 2)
        else:
            # Fallback: use position if available
            if 'pos' in df.columns:
                return pd.to_numeric(df['pos'], errors='coerce').fillna(50)
            return 72  # Default to par

    def load_tournament_level_data(
        self,
        min_year: int = 2020,
        max_year: int = 2024
    ) -> pd.DataFrame:
        """
        Load tournament-level data for classification tasks.

        Unlike create_player_stats() which aggregates to season-level,
        this preserves individual tournament records with:
        - date (for temporal ordering and rolling features)
        - made_cut, top_10, win (classification targets)
        - SG metrics (for rolling feature computation)
        - player_id, course, tournament_name

        This data format is required for outcome prediction with
        time-aware rolling features.

        Args:
            min_year: Minimum season year to include
            max_year: Maximum season year to include

        Returns:
            DataFrame with one row per player-tournament, sorted by date
        """
        print("\n  Loading tournament-level data for outcome prediction...")

        # Load raw Kaggle data
        raw_df = self.load_kaggle_data(min_year, max_year)

        # Identify column names
        player_col = 'Player_initial_last' if 'Player_initial_last' in raw_df.columns else 'player'
        course_col = 'course' if 'course' in raw_df.columns else 'course_name'

        # Parse date
        if 'date' in raw_df.columns:
            raw_df['date'] = pd.to_datetime(raw_df['date'], errors='coerce')
        else:
            # Create synthetic dates from season if no date column
            raw_df['date'] = pd.to_datetime(raw_df['season'].astype(str) + '-06-01')

        # Filter for records with strokes gained data (needed for features)
        df = raw_df.copy()
        if 'sg_total' in df.columns:
            df = df[df['sg_total'].notna()]

        # Create binary classification targets
        # 1. made_cut - already exists in most datasets
        if 'made_cut' not in df.columns:
            # Infer from n_rounds: 4 rounds = made cut
            if 'n_rounds' in df.columns:
                df['made_cut'] = (df['n_rounds'] >= 4).astype(int)
            else:
                df['made_cut'] = 1  # Assume made cut if no data

        # 2. top_10 - derive from position
        if 'pos' in df.columns:
            # Parse position (handles "T5", "CUT", etc.)
            pos_numeric = pd.to_numeric(
                df['pos'].astype(str).str.replace('T', '').str.replace('CUT', '999'),
                errors='coerce'
            )
            df['top_10'] = (pos_numeric <= 10).astype(int)
            df['top_5'] = (pos_numeric <= 5).astype(int)
            df['win'] = (pos_numeric == 1).astype(int)
            df['position_numeric'] = pos_numeric
        else:
            df['top_10'] = 0
            df['top_5'] = 0
            df['win'] = 0
            df['position_numeric'] = 999

        # Standardize column names
        result = pd.DataFrame({
            'player_id': df[player_col],
            'course': df[course_col],
            'date': df['date'],
            'season': df['season'] if 'season' in df.columns else df['date'].dt.year,
            'tournament_name': df['tournament_name'] if 'tournament_name' in df.columns else df[course_col],
            # Strokes Gained metrics for rolling features
            'sg_total': df['sg_total'] if 'sg_total' in df.columns else 0,
            'sg_ott': df['sg_ott'] if 'sg_ott' in df.columns else 0,
            'sg_app': df['sg_app'] if 'sg_app' in df.columns else 0,
            'sg_arg': df['sg_arg'] if 'sg_arg' in df.columns else 0,
            'sg_putt': df['sg_putt'] if 'sg_putt' in df.columns else 0,
            # Classification targets
            'made_cut': df['made_cut'],
            'top_10': df['top_10'],
            'top_5': df['top_5'],
            'win': df['win'],
            'position': df['position_numeric'],
        })

        # Sort by player and date for proper rolling calculations
        result = result.sort_values(['player_id', 'date']).reset_index(drop=True)

        # Print summary
        n_players = result['player_id'].nunique()
        n_tournaments = result['tournament_name'].nunique()
        cut_rate = result['made_cut'].mean() * 100
        top10_rate = result['top_10'].mean() * 100
        win_rate = result['win'].mean() * 100

        print(f"  Tournament-Level Data Summary:")
        print(f"    - Records: {len(result)}")
        print(f"    - Players: {n_players}")
        print(f"    - Tournaments: {n_tournaments}")
        print(f"    - Date range: {result['date'].min().date()} to {result['date'].max().date()}")
        print(f"    - Made cut rate: {cut_rate:.1f}%")
        print(f"    - Top-10 rate: {top10_rate:.1f}%")
        print(f"    - Win rate: {win_rate:.2f}%")

        return result

    def load_data(
        self,
        min_year: int = 2020,
        max_year: int = 2022
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Main entry point: Load all data and return in expected format.

        Args:
            min_year: Minimum season year
            max_year: Maximum season year

        Returns:
            Tuple of (player_stats, course_features, tournament_results)
        """
        print("\n  Loading real PGA data...")

        # Load raw data
        raw_df = self.load_kaggle_data(min_year, max_year)
        course_chars = self.load_course_characteristics()

        # Transform to expected format
        player_stats = self.create_player_stats(raw_df)
        course_features = self.create_course_features(raw_df, course_chars)
        tournament_results = self.create_tournament_results(raw_df)

        # Save processed data
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        player_stats.to_csv(self.processed_dir / "player_stats.csv", index=False)
        course_features.to_csv(self.processed_dir / "course_features.csv", index=False)
        tournament_results.to_csv(self.processed_dir / "tournament_results.csv", index=False)

        # Print data quality summary
        print(f"\n  Data Quality Summary:")
        print(f"    - Players with valid SG data: {len(player_stats)}")
        print(f"    - Courses with characteristics: {len(course_chars)}")
        print(f"    - Tournament records: {len(tournament_results)}")

        return player_stats, course_features, tournament_results


class DataLoader:
    """
    Backward-compatible data loader that supports both synthetic and real data.
    Falls back to synthetic data if real data is not available.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.real_loader = RealDataLoader(data_dir)

    def create_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create sample data for demonstration.
        Used when real data is not available.

        Returns:
            Tuple of (player_stats, course_features, tournament_results)
        """
        # Sample player statistics
        np.random.seed(42)
        players = [f"Player_{i}" for i in range(1, 51)]

        player_stats = pd.DataFrame({
            'player_id': players,
            'driving_distance': np.random.normal(285, 15, len(players)),
            'driving_accuracy': np.random.uniform(45, 75, len(players)),
            'greens_in_regulation': np.random.uniform(60, 75, len(players)),
            'scrambling': np.random.uniform(40, 65, len(players)),
            'putting_average': np.random.uniform(1.65, 1.95, len(players)),
            'short_game': np.random.uniform(40, 75, len(players)),
            'off_the_tee': np.random.uniform(40, 75, len(players)),
            'approach_play': np.random.uniform(40, 75, len(players)),
        })

        # Sample course features
        courses = [f"Course_{i}" for i in range(1, 21)]

        course_features = pd.DataFrame({
            'course_id': courses,
            'course_name': [f"Championship Course {i}" for i in range(1, 21)],
            'yardage': np.random.uniform(6500, 7600, len(courses)),
            'par': np.random.choice([70, 71, 72], len(courses)),
            'par_3': np.random.choice([4, 5], len(courses)),
            'par_4': np.random.choice([10, 11], len(courses)),
            'par_5': np.random.choice([2, 3, 4], len(courses)),
            'fairway_width_avg': np.random.uniform(25, 50, len(courses)),
            'green_size_avg': np.random.uniform(5000, 8000, len(courses)),
            'rough_severity': np.random.uniform(1, 5, len(courses)),
            'hazard_count': np.random.randint(10, 40, len(courses)),
            'elevation_change': np.random.uniform(0, 200, len(courses)),
            'slope_rating': np.random.uniform(130, 155, len(courses)),
            'course_rating': np.random.uniform(69, 76, len(courses)),
        })

        # Sample tournament results (player-course pairings with scores)
        np.random.seed(123)
        tournament_data = []
        for course_id in course_features['course_id'].unique():
            for player_id in np.random.choice(player_stats['player_id'], size=25, replace=False):
                course_row = course_features[course_features['course_id'] == course_id].iloc[0]
                player_row = player_stats[player_stats['player_id'] == player_id].iloc[0]

                # Score influenced by player-course interaction
                base_score = course_row['course_rating'] + np.random.normal(0, 2)

                # Players with good GIR and short game do better on tight courses
                tight_course_bonus = -0.3 * (player_row['greens_in_regulation'] - 67) * (
                    (50 - course_row['fairway_width_avg']) / 25
                )

                # Players with high accuracy do better on courses with less hazards
                hazard_bonus = 0.05 * (player_row['driving_accuracy'] - 60) * (
                    (50 - course_row['hazard_count']) / 30
                )

                final_score = base_score + tight_course_bonus + hazard_bonus + np.random.normal(0, 1)

                tournament_data.append({
                    'player_id': player_id,
                    'course_id': course_id,
                    'score': final_score,
                    'rounds': 1,
                })

        tournament_results = pd.DataFrame(tournament_data)

        return player_stats, course_features, tournament_results

    def load_data(self, use_real_data: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load data from real sources or fall back to synthetic.

        Args:
            use_real_data: If True, attempt to load real Kaggle data

        Returns:
            Tuple of (player_stats, course_features, tournament_results)
        """
        if use_real_data:
            try:
                return self.real_loader.load_data()
            except FileNotFoundError as e:
                print(f"\n  Warning: {e}")
                print("  Falling back to synthetic data...")

        # Generate or load synthetic data
        player_file = self.data_dir / "player_stats.csv"
        course_file = self.data_dir / "course_features.csv"
        tournament_file = self.data_dir / "tournament_results.csv"

        if all([player_file.exists(), course_file.exists(), tournament_file.exists()]):
            player_stats = pd.read_csv(player_file)
            course_features = pd.read_csv(course_file)
            tournament_results = pd.read_csv(tournament_file)
        else:
            print("  Creating synthetic sample data...")
            player_stats, course_features, tournament_results = self.create_sample_data()

            # Save sample data
            self.data_dir.mkdir(exist_ok=True)
            player_stats.to_csv(player_file, index=False)
            course_features.to_csv(course_file, index=False)
            tournament_results.to_csv(tournament_file, index=False)

        return player_stats, course_features, tournament_results
