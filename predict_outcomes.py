#!/usr/bin/env python
"""
Tournament Outcome Prediction Script

Predicts tournament outcomes probabilistically:
- Make/miss cut
- Top-10 finish
- Win probability

Uses rolling form features with strict leakage prevention and
calibrated probability outputs.

Usage:
    py -3.11 predict_outcomes.py --train
    py -3.11 predict_outcomes.py --train --model logistic
    py -3.11 predict_outcomes.py --evaluate
    py -3.11 predict_outcomes.py --predict --tournament "The Memorial"

Examples:
    # Train models on 2015-2022 data
    py -3.11 predict_outcomes.py --train

    # Train with logistic regression baseline
    py -3.11 predict_outcomes.py --train --model logistic

    # Evaluate saved models
    py -3.11 predict_outcomes.py --evaluate
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_outcome_prediction_pipeline
from src.outcome_predictor import OutcomePredictor, OutcomeEvaluator
from src.data_loader import RealDataLoader
from src.rolling_features import RollingFormCalculator
from src.feature_engineer import FeatureEngineer


def train_models(args):
    """Train outcome prediction models."""
    print("\n" + "=" * 70)
    print("TRAINING TOURNAMENT OUTCOME PREDICTION MODELS")
    print("=" * 70)

    # Parse outcomes
    outcomes = None
    if args.outcomes:
        outcomes = [o.strip() for o in args.outcomes.split(',')]

    # Run pipeline
    results = run_outcome_prediction_pipeline(
        min_year=args.min_year,
        max_year=args.max_year,
        outcomes=outcomes,
        model_type=args.model,
        calibration_method=args.calibration,
        test_size=args.test_size
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    # Summary
    print("\nTest Set Performance:")
    for outcome, metrics in results['test_metrics'].items():
        brier = metrics.get('brier_score', 'N/A')
        roc = metrics.get('roc_auc', 'N/A')
        if isinstance(brier, float):
            brier = f"{brier:.4f}"
        if isinstance(roc, float):
            roc = f"{roc:.4f}"
        print(f"  {outcome:15s} - Brier: {brier}, ROC-AUC: {roc}")

    return results


def evaluate_models(args):
    """Evaluate saved models on recent data."""
    print("\n" + "=" * 70)
    print("EVALUATING SAVED MODELS")
    print("=" * 70)

    outcomes = ['made_cut', 'top_10', 'win']
    evaluator = OutcomeEvaluator()

    # Load models
    predictors = {}
    for outcome in outcomes:
        model_path = Path(f"models/outcome_{outcome}.pkl")
        if model_path.exists():
            print(f"  Loading {model_path}...")
            predictors[outcome] = OutcomePredictor.load_model(str(model_path))
        else:
            print(f"  Model not found: {model_path}")

    if not predictors:
        print("\nNo models found. Run --train first.")
        return

    # Load evaluation data
    print("\n  Loading evaluation data...")
    loader = RealDataLoader()
    tournament_df = loader.load_tournament_level_data(
        min_year=args.eval_year,
        max_year=args.eval_year
    )
    course_features = loader.load_course_characteristics()

    # Compute features
    rolling_calc = RollingFormCalculator(windows=[5, 10])
    engineer = FeatureEngineer()
    X, targets = engineer.create_classification_features(
        tournament_df, course_features, rolling_calc
    )

    # Get feature columns
    metadata_cols = ['player_id', 'course', 'date', 'tournament_name']
    feature_cols = [col for col in X.columns if col not in metadata_cols]

    # Evaluate each model
    eval_results = {}
    for outcome, predictor in predictors.items():
        if outcome not in targets:
            continue

        X_eval = X[feature_cols].copy()
        y_eval = targets[outcome].copy()

        # Drop NaN
        mask = X_eval.notna().all(axis=1) & y_eval.notna()
        X_eval = X_eval[mask]
        y_eval = y_eval[mask]

        if len(X_eval) == 0:
            continue

        y_pred = predictor.predict_proba(X_eval)
        eval_results[outcome] = evaluator.classification_report(
            y_eval, y_pred, outcome_type=outcome
        )

    evaluator.print_evaluation_summary(eval_results)


def predict_player_course(args):
    """Predict outcomes for a specific player at a specific course."""
    print("\n" + "=" * 70)
    print("PLAYER-COURSE OUTCOME PREDICTION")
    print("=" * 70)

    player_name = args.player
    course_name = args.course

    # Load models
    outcomes = ['made_cut', 'top_10', 'win']
    predictors = {}
    for outcome in outcomes:
        model_path = Path(f"models/outcome_{outcome}.pkl")
        if model_path.exists():
            predictors[outcome] = OutcomePredictor.load_model(str(model_path))
        else:
            print(f"  Warning: Model not found: {model_path}")

    if not predictors:
        print("\nNo models found. Run --train first.")
        return

    # Load data to get player's recent form
    print(f"\n  Loading player data...")
    loader = RealDataLoader()

    # Load most recent data available
    tournament_df = loader.load_tournament_level_data(
        min_year=args.data_year - 2,
        max_year=args.data_year
    )
    course_features = loader.load_course_characteristics()

    # Find matching player (fuzzy match)
    all_players = tournament_df['player_id'].unique()
    matching_players = [p for p in all_players if player_name.lower() in p.lower()]

    if not matching_players:
        print(f"\n  Player '{player_name}' not found in data.")
        print(f"  Available players (sample): {list(all_players[:20])}")
        return

    if len(matching_players) > 1:
        print(f"\n  Multiple matches found: {matching_players}")
        player_id = matching_players[0]
        print(f"  Using: {player_id}")
    else:
        player_id = matching_players[0]

    # Find matching course (fuzzy match)
    all_courses = tournament_df['course'].unique()
    matching_courses = [c for c in all_courses if course_name.lower() in c.lower()]

    if not matching_courses:
        print(f"\n  Course '{course_name}' not found in tournament data.")
        print(f"  Available courses (sample): {list(all_courses[:20])}")
        # Try to use course from characteristics
        if not course_features.empty and 'course_id' in course_features.columns:
            char_courses = course_features['course_id'].unique()
            matching_courses = [c for c in char_courses if course_name.lower() in c.lower()]
            if matching_courses:
                course_id = matching_courses[0]
                print(f"  Found in course characteristics: {course_id}")
            else:
                return
        else:
            return
    else:
        course_id = matching_courses[0]

    print(f"\n  Player: {player_id}")
    print(f"  Course: {course_id}")

    # Get player's recent tournaments
    player_data = tournament_df[tournament_df['player_id'] == player_id].copy()
    player_data = player_data.sort_values('date', ascending=False)

    if len(player_data) == 0:
        print(f"\n  No tournament data found for {player_id}")
        return

    print(f"\n  Player's Recent Form (last 5 tournaments):")
    recent = player_data.head(5)
    for _, row in recent.iterrows():
        result = "Made Cut" if row['made_cut'] == 1 else "Missed Cut"
        top10 = " (Top 10)" if row['top_10'] == 1 else ""
        win = " (WIN!)" if row['win'] == 1 else ""
        print(f"    {row['date'].date()}: {row['tournament_name'][:30]:30s} SG: {row['sg_total']:+.2f} {result}{top10}{win}")

    # Compute rolling features for this player
    rolling_calc = RollingFormCalculator(windows=[5, 10])
    player_with_rolling = rolling_calc.compute_all_features(
        player_data,
        player_col='player_id',
        course_col='course',
        date_col='date'
    )

    # Get most recent row (current form)
    latest_form = player_with_rolling.iloc[-1:].copy()

    # Check if player has history at this course
    course_history = player_data[player_data['course'] == course_id]
    if len(course_history) > 0:
        print(f"\n  Player's History at {course_id}:")
        for _, row in course_history.iterrows():
            result = "Made Cut" if row['made_cut'] == 1 else "Missed Cut"
            pos = f"Pos: {int(row['position'])}" if row['position'] < 999 else ""
            print(f"    {row['date'].date()}: SG: {row['sg_total']:+.2f} {result} {pos}")

        # Compute course-specific averages
        latest_form['course_avg_sg'] = course_history['sg_total'].mean()
        latest_form['course_appearances'] = len(course_history)
    else:
        print(f"\n  No history at {course_id} - using player average")
        latest_form['course_avg_sg'] = player_data['sg_total'].mean()
        latest_form['course_appearances'] = 0

    # Merge course features
    engineer = FeatureEngineer()
    if not course_features.empty:
        course_df = engineer.engineer_course_features(course_features)
        course_row = course_df[course_df['course_id'].str.lower().str.contains(course_name.lower())]
        if len(course_row) > 0:
            for col in course_row.columns:
                if col != 'course_id':
                    latest_form[col] = course_row[col].values[0]
        else:
            # Use defaults
            print(f"  Course characteristics not found - using tour averages")
            latest_form['overall_difficulty'] = 0.5
            latest_form['is_tight_course'] = 0
            latest_form['is_long_course'] = 0
            latest_form['green_challenge'] = 0.5
            latest_form['hazard_density'] = 0.3
            latest_form['approach_difficulty'] = 0.5
            latest_form['yardage'] = 7200
            latest_form['fairway_width_avg'] = 32
            latest_form['slope_rating'] = 140

    # Create interaction features
    if 'sg_total_last_5' in latest_form.columns and 'overall_difficulty' in latest_form.columns:
        latest_form['form_difficulty_interaction'] = (
            latest_form['sg_total_last_5'] * latest_form['overall_difficulty']
        )
    if 'course_avg_sg' in latest_form.columns and 'course_appearances' in latest_form.columns:
        import numpy as np
        latest_form['course_history_boost'] = (
            latest_form['course_avg_sg'] * np.log1p(latest_form['course_appearances'])
        )
    if 'sg_total_momentum' in latest_form.columns and 'overall_difficulty' in latest_form.columns:
        latest_form['momentum_difficulty_interaction'] = (
            latest_form['sg_total_momentum'] * latest_form['overall_difficulty']
        )

    # Get feature columns from saved model
    feature_cols = predictors[list(predictors.keys())[0]].feature_names

    # Build feature vector
    X_pred = latest_form[feature_cols].copy() if all(c in latest_form.columns for c in feature_cols) else None

    if X_pred is None:
        # Build manually with available features
        X_pred = pd.DataFrame(index=[0])
        for col in feature_cols:
            if col in latest_form.columns:
                X_pred[col] = latest_form[col].values[0]
            else:
                X_pred[col] = 0

    X_pred = X_pred.fillna(0)

    # Make predictions
    print("\n" + "=" * 70)
    print(f"PREDICTED PROBABILITIES: {player_id} at {course_id}")
    print("=" * 70)

    for outcome, predictor in predictors.items():
        prob = predictor.predict_proba(X_pred)[0]
        bar_len = int(prob * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        print(f"\n  {outcome.upper():12s}: {prob*100:5.1f}%  [{bar}]")

    # Context
    print("\n" + "-" * 70)
    print("CONTEXT:")
    if 'sg_total_last_5' in latest_form.columns:
        sg5 = latest_form['sg_total_last_5'].values[0]
        print(f"  Recent Form (Last 5 SG Avg): {sg5:+.2f}")
    if 'sg_total_last_10' in latest_form.columns:
        sg10 = latest_form['sg_total_last_10'].values[0]
        print(f"  Longer Form (Last 10 SG Avg): {sg10:+.2f}")
    if 'course_appearances' in latest_form.columns:
        apps = int(latest_form['course_appearances'].values[0])
        print(f"  Course Appearances: {apps}")
    if 'course_avg_sg' in latest_form.columns:
        csg = latest_form['course_avg_sg'].values[0]
        print(f"  Course History SG Avg: {csg:+.2f}")

    return predictors


def main():
    parser = argparse.ArgumentParser(
        description="Tournament Outcome Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py -3.11 predict_outcomes.py --train
  py -3.11 predict_outcomes.py --train --model logistic --calibration platt
  py -3.11 predict_outcomes.py --evaluate --eval-year 2022
  py -3.11 predict_outcomes.py --predict --player "Scheffler" --course "Augusta"
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train outcome prediction models"
    )
    mode_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate saved models"
    )
    mode_group.add_argument(
        "--predict",
        action="store_true",
        help="Predict outcomes for a specific player at a course"
    )

    # Training options
    parser.add_argument(
        "--min-year",
        type=int,
        default=2015,
        help="Minimum year for training data (default: 2015)"
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=2022,
        help="Maximum year for training data (default: 2022)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['xgboost', 'logistic'],
        default='xgboost',
        help="Model type (default: xgboost)"
    )
    parser.add_argument(
        "--calibration",
        type=str,
        choices=['platt', 'isotonic'],
        default='isotonic',
        help="Calibration method (default: isotonic)"
    )
    parser.add_argument(
        "--outcomes",
        type=str,
        default=None,
        help="Comma-separated outcomes to predict (default: made_cut,top_10,win)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data for testing (default: 0.2)"
    )

    # Evaluation options
    parser.add_argument(
        "--eval-year",
        type=int,
        default=2022,
        help="Year to evaluate on (default: 2022)"
    )

    # Prediction options
    parser.add_argument(
        "--player",
        type=str,
        help="Player name (partial match, e.g., 'Scheffler')"
    )
    parser.add_argument(
        "--course",
        type=str,
        help="Course name (partial match, e.g., 'Augusta')"
    )
    parser.add_argument(
        "--data-year",
        type=int,
        default=2022,
        help="Year of data to use for player form (default: 2022)"
    )

    args = parser.parse_args()

    if args.train:
        train_models(args)
    elif args.evaluate:
        evaluate_models(args)
    elif args.predict:
        if not args.player or not args.course:
            parser.error("--predict requires --player and --course arguments")
        predict_player_course(args)


if __name__ == "__main__":
    main()
