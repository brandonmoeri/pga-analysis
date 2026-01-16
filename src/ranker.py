"""
Ranking and tournament analysis module.
Generates "best fit" rankings for each course/tournament.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class CourseFitRanker:
    """Rank and analyze player-course fit."""
    
    def __init__(self, model, explainer=None):
        """
        Initialize ranker.
        
        Args:
            model: Trained CourseFitModel
            explainer: Optional ShapExplainer for insights
        """
        self.model = model
        self.explainer = explainer
    
    def rank_players_for_tournament(
        self,
        X: pd.DataFrame,
        tournament_courses: List[str],
        top_n: int = 10
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate best-fit rankings for each course in tournament.
        
        Args:
            X: Full feature matrix
            tournament_courses: List of course_ids in tournament
            top_n: Number of players to rank
        
        Returns:
            Dictionary mapping course_id to ranked DataFrame
        """
        rankings = {}
        
        for course_id in tournament_courses:
            # Get predictions for this course
            course_mask = X['course_id'] == course_id
            course_X = X[course_mask].copy()
            
            predictions = self.model.predict_fit_score(course_X)
            
            # Sort by fit score (lower = better)
            ranked = predictions.sort_values('predicted_fit_score').head(top_n).reset_index(drop=True)
            ranked['rank'] = range(1, len(ranked) + 1)
            ranked['fit_percentile'] = (1 - ranked['predicted_fit_score'] / ranked['predicted_fit_score'].max()) * 100
            
            rankings[course_id] = ranked
        
        return rankings
    
    def tournament_aggregate_ranking(
        self,
        rankings: Dict[str, pd.DataFrame],
        aggregation_method: str = 'mean'
    ) -> pd.DataFrame:
        """
        Aggregate rankings across multiple courses.
        
        Args:
            rankings: Dictionary of course rankings
            aggregation_method: 'mean', 'median', or 'min' fit score
        
        Returns:
            Overall tournament ranking
        """
        # Collect all players and their fit scores
        all_scores = []
        
        for course_id, ranked_df in rankings.items():
            for _, row in ranked_df.iterrows():
                all_scores.append({
                    'player_id': row['player_id'],
                    'course_id': course_id,
                    'fit_score': row['predicted_fit_score'],
                    'fit_percentile': row['fit_percentile']
                })
        
        scores_df = pd.DataFrame(all_scores)
        
        # Aggregate by player
        if aggregation_method == 'mean':
            agg_scores = scores_df.groupby('player_id')['fit_score'].mean()
        elif aggregation_method == 'median':
            agg_scores = scores_df.groupby('player_id')['fit_score'].median()
        elif aggregation_method == 'min':  # Best possible score
            agg_scores = scores_df.groupby('player_id')['fit_score'].min()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        # Create ranking
        tournament_ranking = pd.DataFrame({
            'player_id': agg_scores.index,
            'aggregate_fit_score': agg_scores.values
        }).sort_values('aggregate_fit_score').reset_index(drop=True)
        
        tournament_ranking['tournament_rank'] = range(1, len(tournament_ranking) + 1)
        
        # Add number of courses player was ranked in
        course_appearances = scores_df.groupby('player_id')['course_id'].nunique()
        tournament_ranking = tournament_ranking.merge(
            course_appearances.rename('courses_ranked'),
            left_on='player_id',
            right_index=True
        )
        
        return tournament_ranking
    
    def player_course_profile(
        self,
        X: pd.DataFrame,
        player_id: str
    ) -> pd.DataFrame:
        """
        Get a player's fit score across all courses.
        
        Args:
            X: Full feature matrix
            player_id: Player to analyze
        
        Returns:
            DataFrame of player's fit for each course
        """
        player_mask = X['player_id'] == player_id
        player_X = X[player_mask].copy()
        
        predictions = self.model.predict_fit_score(player_X)
        predictions = predictions.sort_values('predicted_fit_score')
        
        return predictions
    
    def course_difficulty_variance(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze how fit scores vary by course.
        Shows which courses are more/less selective.
        
        Args:
            X: Full feature matrix
        
        Returns:
            DataFrame with course variance metrics
        """
        predictions = self.model.predict_fit_score(X)
        
        course_stats = predictions.groupby('course_id').agg({
            'predicted_fit_score': ['mean', 'std', 'min', 'max', 'count']
        }).round(3)
        
        course_stats.columns = ['mean_fit_score', 'std_fit_score', 'best_fit', 'worst_fit', 'num_players']
        
        # Selectivity: higher std = more selective
        course_stats['selectivity'] = course_stats['std_fit_score']
        
        return course_stats.sort_values('mean_fit_score').reset_index()
    
    def create_fit_heatmap(
        self,
        X: pd.DataFrame,
        sample_players: int = 20,
        sample_courses: int = 10
    ) -> plt.Figure:
        """
        Create heatmap of player-course fit scores.
        
        Args:
            X: Full feature matrix
            sample_players: Number of players to display
            sample_courses: Number of courses to display
        
        Returns:
            Matplotlib figure
        """
        predictions = self.model.predict_fit_score(X)
        
        # Sample for visualization
        players = predictions['player_id'].unique()[:sample_players]
        courses = predictions['course_id'].unique()[:sample_courses]
        
        subset = predictions[
            (predictions['player_id'].isin(players)) &
            (predictions['course_id'].isin(courses))
        ]
        
        # Pivot for heatmap
        heatmap_data = subset.pivot(
            index='player_id',
            columns='course_id',
            values='predicted_fit_score'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            cmap='RdYlGn_r',  # Red for bad fits, green for good fits
            cbar_kws={'label': 'Fit Score (lower = better)'},
            ax=ax
        )
        ax.set_title('Player-Course Fit Score Matrix')
        ax.set_xlabel('Course')
        ax.set_ylabel('Player')
        
        return fig
    
    def get_summary_insights(
        self,
        X: pd.DataFrame,
        rankings: Dict[str, pd.DataFrame]
    ) -> Dict[str, any]:
        """
        Generate summary insights from analysis.
        
        Args:
            X: Full feature matrix
            rankings: Course rankings dictionary
        
        Returns:
            Dictionary of key insights
        """
        insights = {}
        
        # Overall statistics
        predictions = self.model.predict_fit_score(X)
        insights['total_player_course_pairs'] = len(predictions)
        insights['num_unique_players'] = predictions['player_id'].nunique()
        insights['num_unique_courses'] = predictions['course_id'].nunique()
        
        # Fit score statistics
        insights['mean_fit_score'] = predictions['predicted_fit_score'].mean()
        insights['std_fit_score'] = predictions['predicted_fit_score'].std()
        insights['best_fit_score'] = predictions['predicted_fit_score'].min()
        insights['worst_fit_score'] = predictions['predicted_fit_score'].max()
        
        # Course statistics
        course_difficulty = self.course_difficulty_variance(X)
        insights['easiest_course'] = course_difficulty.iloc[0]['course_id']
        insights['hardest_course'] = course_difficulty.iloc[-1]['course_id']
        insights['most_selective_course'] = course_difficulty.loc[
            course_difficulty['selectivity'].idxmax(), 'course_id'
        ]
        
        # Top players
        tournament_ranking = self.tournament_aggregate_ranking(rankings)
        insights['top_3_players'] = tournament_ranking.head(3)['player_id'].tolist()
        
        return insights
