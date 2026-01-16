"""
Main pipeline for Course Fit Model.
Orchestrates data loading, feature engineering, training, and analysis.
"""

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model import CourseFitModel
from src.explainer import ShapExplainer
from src.ranker import CourseFitRanker
import pandas as pd
import numpy as np


def run_course_fit_pipeline(
    model_type: str = 'xgboost',
    use_shap: bool = True,
    use_real_data: bool = True,
    min_year: int = 2020,
    max_year: int = 2022
):
    """
    Complete pipeline for course fit modeling.

    Args:
        model_type: 'xgboost' or 'lightgbm'
        use_shap: Whether to generate SHAP explanations
        use_real_data: Whether to use real Kaggle data (falls back to synthetic if unavailable)
        min_year: Minimum season year for real data
        max_year: Maximum season year for real data
    """

    print("=" * 70)
    print("COURSE FIT MODEL PIPELINE")
    print("=" * 70)

    # 1. Load Data
    data_mode = "REAL" if use_real_data else "SYNTHETIC"
    print(f"\n[1/5] Loading data ({data_mode} mode)...")

    loader = DataLoader()
    if use_real_data:
        try:
            player_stats, course_features, tournament_results = loader.real_loader.load_data(
                min_year=min_year, max_year=max_year
            )
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            print("  Falling back to synthetic data...")
            player_stats, course_features, tournament_results = loader.load_data(use_real_data=False)
    else:
        player_stats, course_features, tournament_results = loader.load_data(use_real_data=False)
    
    print(f"  - Players: {len(player_stats)}")
    print(f"  - Courses: {len(course_features)}")
    print(f"  - Tournament results: {len(tournament_results)}")
    
    # 2. Feature Engineering
    print("\n[2/5] Engineering features...")
    engineer = FeatureEngineer()
    X, y = engineer.create_player_course_interactions(
        player_stats, course_features, tournament_results
    )
    
    feature_cols = [col for col in X.columns if col not in ['player_id', 'course_id']]
    print(f"  - Total features created: {len(feature_cols)}")
    print(f"  - Training samples: {len(X)}")
    
    # 3. Train Model
    print(f"\n[3/5] Training {model_type.upper()} model...")
    model = CourseFitModel(model_type=model_type)
    metrics = model.train(X, y)
    
    # Get feature importance
    importance = model.get_feature_importance(top_n=15)
    print("\n  Top 15 Features:")
    for idx, row in importance.iterrows():
        print(f"    {row['feature']:35s}: {row['importance']:.4f}")
    
    # 4. SHAP Explainability (if enabled)
    explainer = None
    if use_shap:
        print("\n[4/5] Generating SHAP explanations...")
        X_features_only = X[[col for col in X.columns if col not in ['player_id', 'course_id']]]
        explainer = ShapExplainer(model.model, X_features_only)
        
        global_importance = explainer.global_feature_importance(X_features_only, top_n=10)
        print("\n  Top 10 Features (by SHAP impact):")
        for idx, row in global_importance.iterrows():
            print(f"    {row['feature']:35s}: {row['mean_abs_shap']:.4f}")
    
    # 5. Ranking and Analysis
    print("\n[5/5] Generating rankings and insights...")
    ranker = CourseFitRanker(model, explainer)
    
    # Get tournament courses
    tournament_courses = course_features['course_id'].unique()[:5]  # Analyze first 5 courses
    
    # Generate rankings
    rankings = ranker.rank_players_for_tournament(X, tournament_courses, top_n=10)
    
    print(f"\n  Generated rankings for {len(rankings)} courses")
    
    # Show sample rankings
    print("\n" + "=" * 70)
    print("SAMPLE COURSE RANKINGS")
    print("=" * 70)
    
    for course_id in list(rankings.keys())[:3]:
        print(f"\n{course_id} - Top 5 Best Fits:")
        top_5 = rankings[course_id].head(5)[['rank', 'player_id', 'predicted_fit_score', 'fit_percentile']]
        print(top_5.to_string(index=False))
    
    # Tournament aggregate
    print("\n" + "=" * 70)
    print("TOURNAMENT AGGREGATE RANKING")
    print("=" * 70)
    
    tournament_ranking = ranker.tournament_aggregate_ranking(rankings, aggregation_method='mean')
    print("\nTop 10 Players Overall:")
    print(tournament_ranking.head(10)[['tournament_rank', 'player_id', 'aggregate_fit_score', 'courses_ranked']].to_string(index=False))
    
    # Insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    insights = ranker.get_summary_insights(X, rankings)
    print(f"\nTotal Player-Course Pairs Analyzed: {insights['total_player_course_pairs']}")
    print(f"Players Evaluated: {insights['num_unique_players']}")
    print(f"Courses: {insights['num_unique_courses']}")
    print(f"\nMean Fit Score: {insights['mean_fit_score']:.4f}")
    print(f"Std Dev: {insights['std_fit_score']:.4f}")
    print(f"Best Fit Score: {insights['best_fit_score']:.4f}")
    print(f"Worst Fit Score: {insights['worst_fit_score']:.4f}")
    
    print(f"\nEasiest Course: {insights['easiest_course']}")
    print(f"Hardest Course: {insights['hardest_course']}")
    print(f"Most Selective Course: {insights['most_selective_course']}")
    print(f"\nTop 3 Overall Players: {', '.join(insights['top_3_players'])}")
    
    # Save model
    print("\n" + "=" * 70)
    model.save_model()
    
    return {
        'model': model,
        'explainer': explainer,
        'ranker': ranker,
        'player_stats': player_stats,
        'course_features': course_features,
        'X': X,
        'y': y,
        'rankings': rankings,
        'tournament_ranking': tournament_ranking,
        'insights': insights
    }


if __name__ == "__main__":
    results = run_course_fit_pipeline(model_type='xgboost', use_shap=True)
