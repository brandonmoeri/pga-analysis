"""
Main pipeline for Course Fit Model.
Orchestrates data loading, feature engineering, training, and analysis.

Includes two pipelines:
1. run_course_fit_pipeline - Regression model for course fit scoring
2. run_outcome_prediction_pipeline - Classification for tournament outcomes
"""

from src.data_loader import DataLoader, RealDataLoader
from src.feature_engineer import FeatureEngineer
from src.model import CourseFitModel
from src.explainer import ShapExplainer
from src.ranker import CourseFitRanker
from src.rolling_features import RollingFormCalculator, TimeSeriesSplit
from src.outcome_predictor import OutcomePredictor, OutcomeEvaluator
import pandas as pd
import numpy as np
from typing import Dict, Any, List


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


def run_outcome_prediction_pipeline(
    min_year: int = 2015,
    max_year: int = 2022,
    outcomes: List[str] = None,
    model_type: str = 'xgboost',
    calibration_method: str = 'isotonic',
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Complete pipeline for tournament outcome prediction.

    Predicts:
    - make_cut: Did player make the cut?
    - top_10: Did player finish top-10?
    - win: Did player win?

    Features:
    - Rolling form (last 5/10 tournaments) with leakage prevention
    - Course history (past performance at this course)
    - Form momentum (improving vs declining)
    - Calibrated probability outputs

    Args:
        min_year: Minimum season year to include
        max_year: Maximum season year to include
        outcomes: List of outcomes to predict (default: all three)
        model_type: 'xgboost' or 'logistic'
        calibration_method: 'platt' or 'isotonic'
        test_size: Fraction of data for testing (most recent)

    Returns:
        Dictionary with models, metrics, and evaluation results
    """
    if outcomes is None:
        outcomes = ['made_cut', 'top_10', 'win']

    print("=" * 70)
    print("TOURNAMENT OUTCOME PREDICTION PIPELINE")
    print("=" * 70)

    # 1. Load tournament-level data
    print(f"\n[1/6] Loading tournament-level data ({min_year}-{max_year})...")
    loader = RealDataLoader()
    tournament_df = loader.load_tournament_level_data(min_year, max_year)
    course_features = loader.load_course_characteristics()

    # 2. Compute rolling features
    print("\n[2/6] Computing rolling form features...")
    print("  (Using shift(1) to prevent label leakage)")
    rolling_calc = RollingFormCalculator(windows=[5, 10])

    # 3. Engineer classification features
    print("\n[3/6] Engineering classification features...")
    engineer = FeatureEngineer()
    X, targets = engineer.create_classification_features(
        tournament_df, course_features, rolling_calc
    )

    # 4. Time-aware train/test split
    print("\n[4/6] Creating time-aware train/test split...")
    print(f"  (Training on past, testing on most recent {test_size*100:.0f}%)")

    # Get feature columns (exclude metadata)
    metadata_cols = ['player_id', 'course', 'date', 'tournament_name']
    feature_cols = [col for col in X.columns if col not in metadata_cols]

    # Combine X with targets for splitting
    split_df = X.copy()
    for outcome, target_series in targets.items():
        split_df[outcome] = target_series.values

    # Temporal split
    train_df, test_df = TimeSeriesSplit.temporal_split(
        split_df, date_col='date', test_size=test_size
    )

    print(f"  Train: {len(train_df)} samples ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Test:  {len(test_df)} samples ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # 5. Train models for each outcome
    print(f"\n[5/6] Training {model_type.upper()} classifiers with {calibration_method} calibration...")

    results = {}
    for outcome in outcomes:
        if outcome not in targets:
            print(f"  Skipping {outcome} - target not available")
            continue

        print(f"\n  --- {outcome.upper()} ---")

        # Get train/test data
        X_train = train_df[feature_cols].copy()
        y_train = train_df[outcome].copy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df[outcome].copy()

        # Fill NaN values in features with 0 (rolling features have NaN for early tournaments)
        # This is acceptable because NaN means "no history" which is essentially neutral
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

        # Drop rows where target is NaN
        train_mask = y_train.notna()
        test_mask = y_test.notna()

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        print(f"  Train samples: {len(X_train)} (positive: {y_train.sum()}, rate: {y_train.mean()*100:.1f}%)")
        print(f"  Test samples: {len(X_test)} (positive: {y_test.sum()}, rate: {y_test.mean()*100:.1f}%)")

        # Create and train predictor
        predictor = OutcomePredictor(
            outcome_type=outcome,
            model_type=model_type,
            calibration_method=calibration_method
        )

        train_metrics = predictor.train(X_train, y_train)
        test_metrics = predictor.evaluate(X_test, y_test)

        print(f"  Train - Brier: {train_metrics['brier_score']:.4f}, ROC-AUC: {train_metrics.get('roc_auc', 'N/A')}")
        print(f"  Test  - Brier: {test_metrics['brier_score']:.4f}, ROC-AUC: {test_metrics.get('roc_auc', 'N/A')}")

        # Get feature importance
        importance = predictor.get_feature_importance(top_n=10)

        # Store results
        results[outcome] = {
            'predictor': predictor,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': importance,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': predictor.predict_proba(X_test)
        }

    # 6. Evaluation summary
    print("\n[6/6] Generating evaluation summary...")

    evaluator = OutcomeEvaluator()
    eval_results = {}
    for outcome, data in results.items():
        eval_results[outcome] = evaluator.classification_report(
            data['y_test'],
            data['y_pred_proba'],
            outcome_type=outcome
        )

    evaluator.print_evaluation_summary(eval_results)

    # Print top features for each outcome
    print("\n" + "=" * 70)
    print("TOP FEATURES BY OUTCOME")
    print("=" * 70)

    for outcome, data in results.items():
        print(f"\n{outcome.upper()} - Top 10 Features:")
        for idx, row in data['feature_importance'].iterrows():
            print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    # Save models
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)

    for outcome, data in results.items():
        filepath = f"models/outcome_{outcome}.pkl"
        data['predictor'].save_model(filepath)
        print(f"  Saved: {filepath}")

    return {
        'predictors': {k: v['predictor'] for k, v in results.items()},
        'train_metrics': {k: v['train_metrics'] for k, v in results.items()},
        'test_metrics': {k: v['test_metrics'] for k, v in results.items()},
        'eval_results': eval_results,
        'feature_cols': feature_cols,
        'results': results
    }


if __name__ == "__main__":
    results = run_course_fit_pipeline(model_type='xgboost', use_shap=True)
