#!/usr/bin/env python
"""
Tournament Outcome Prediction Script

Predicts tournament outcomes probabilistically:
- Make/miss cut
- Top-10 finish
- Win probability

Uses CURRENT season ESPN data for player form estimation and
calibrated probability outputs from trained models.

Usage:
    py -3.11 predict_outcomes.py --train
    py -3.11 predict_outcomes.py --predict --player "Scheffler" --course "Augusta"
    py -3.11 predict_outcomes.py --predict --player "Rory" --course "TPC" --update-data

Examples:
    # Train models on 2015-2022 historical data
    py -3.11 predict_outcomes.py --train

    # Predict player outcomes using current ESPN data
    py -3.11 predict_outcomes.py --predict --player "Scheffler" --course "Augusta"

    # Force refresh of ESPN data before predicting
    py -3.11 predict_outcomes.py --predict --player "Rory" --course "Sawgrass" --update-data

    # Evaluate saved models on historical data
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
from src.pga_scraper import PGATourScraper
import pandas as pd
import numpy as np


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


def fetch_current_player_stats(player_name: str, force_update: bool = False) -> pd.DataFrame:
    """
    Fetch current season stats from ESPN for a player.

    Returns DataFrame with current SG stats that can be used as
    proxies for rolling form features.

    Args:
        player_name: Player name to search for
        force_update: If True, force refresh from ESPN even if cached data exists
    """
    print(f"\n  Fetching current ESPN data...")
    scraper = PGATourScraper()

    # Force update if requested
    if force_update:
        print("  Refreshing data from ESPN...")
        current_stats = scraper.update_player_stats()
    else:
        # Try to load cached data first, then scrape if needed
        current_stats = scraper.load_latest_scraped_data()

        if current_stats.empty:
            print("  No cached data found, scraping from ESPN...")
            current_stats = scraper.update_player_stats()

    if current_stats.empty:
        print("  Warning: Could not fetch current stats from ESPN")
        return pd.DataFrame()

    # Find matching player (fuzzy match on name)
    if 'player_id' not in current_stats.columns:
        # Try to find the name column
        name_cols = [c for c in current_stats.columns if 'name' in c.lower() or 'player' in c.lower()]
        if name_cols:
            current_stats = current_stats.rename(columns={name_cols[0]: 'player_id'})
        else:
            print("  Warning: No player name column found in ESPN data")
            return pd.DataFrame()

    # Fuzzy match player name
    matching = current_stats[
        current_stats['player_id'].str.lower().str.contains(player_name.lower(), na=False)
    ]

    return matching


def predict_player_course(args):
    """Predict outcomes for a specific player at a specific course using CURRENT ESPN data."""
    print("\n" + "=" * 70)
    print("PLAYER-COURSE OUTCOME PREDICTION (CURRENT SEASON)")
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

    # Fetch CURRENT stats from ESPN
    force_update = getattr(args, 'update_data', False)
    espn_stats = fetch_current_player_stats(player_name, force_update=force_update)

    if espn_stats.empty:
        print(f"\n  Player '{player_name}' not found in current ESPN data.")
        print("  Try updating ESPN data with: py -3.11 src/pga_scraper.py")
        return

    if len(espn_stats) > 1:
        print(f"\n  Multiple matches found:")
        for _, row in espn_stats.iterrows():
            print(f"    - {row['player_id']}")
        player_row = espn_stats.iloc[0]
        print(f"  Using: {player_row['player_id']}")
    else:
        player_row = espn_stats.iloc[0]

    player_id = player_row['player_id']
    print(f"\n  Player: {player_id}")
    print(f"  Course: {course_name}")

    # Display current season stats from ESPN
    print(f"\n  Current Season Stats (ESPN):")
    stat_display = {
        'sg_total': 'SG: Total',
        'sg_ott': 'SG: Off-the-Tee',
        'sg_app': 'SG: Approach',
        'sg_arg': 'SG: Around Green',
        'sg_putt': 'SG: Putting',
        'driving_distance': 'Driving Distance',
        'driving_accuracy': 'Driving Accuracy',
        'greens_in_regulation': 'GIR %',
        'scrambling': 'Scrambling %',
    }
    for col, label in stat_display.items():
        if col in player_row.index and pd.notna(player_row[col]):
            val = player_row[col]
            if 'sg_' in col:
                print(f"    {label}: {val:+.2f}")
            elif 'accuracy' in col or 'regulation' in col or 'scrambling' in col:
                print(f"    {label}: {val:.1f}%")
            else:
                print(f"    {label}: {val:.1f}")

    # Build feature vector using current ESPN stats
    # Map ESPN season averages to rolling feature proxies
    latest_form = pd.DataFrame(index=[0])

    # Use current season SG as proxy for rolling averages
    # (assumes current season average is a good estimate of recent form)
    sg_total = player_row.get('sg_total', 0) if pd.notna(player_row.get('sg_total')) else 0
    sg_ott = player_row.get('sg_ott', player_row.get('off_the_tee', 0)) if pd.notna(player_row.get('sg_ott', player_row.get('off_the_tee'))) else 0
    sg_app = player_row.get('sg_app', player_row.get('approach_play', 0)) if pd.notna(player_row.get('sg_app', player_row.get('approach_play'))) else 0
    sg_arg = player_row.get('sg_arg', player_row.get('short_game', 0)) if pd.notna(player_row.get('sg_arg', player_row.get('short_game'))) else 0
    sg_putt = player_row.get('sg_putt', 0) if pd.notna(player_row.get('sg_putt')) else 0

    # Rolling features - use current season stats as proxy
    latest_form['sg_total_last_5'] = sg_total
    latest_form['sg_total_last_10'] = sg_total
    latest_form['sg_ott_last_5'] = sg_ott
    latest_form['sg_ott_last_10'] = sg_ott
    latest_form['sg_app_last_5'] = sg_app
    latest_form['sg_app_last_10'] = sg_app
    latest_form['sg_arg_last_5'] = sg_arg
    latest_form['sg_arg_last_10'] = sg_arg
    latest_form['sg_putt_last_5'] = sg_putt
    latest_form['sg_putt_last_10'] = sg_putt

    # Standard deviations (assume moderate consistency for current data)
    latest_form['sg_total_std_last_10'] = 1.5
    latest_form['sg_ott_std_last_10'] = 0.5
    latest_form['sg_app_std_last_10'] = 0.5
    latest_form['sg_arg_std_last_10'] = 0.3
    latest_form['sg_putt_std_last_10'] = 0.3

    # Momentum (assume neutral for season averages)
    latest_form['sg_total_momentum'] = 0
    latest_form['sg_ott_momentum'] = 0
    latest_form['sg_app_momentum'] = 0
    latest_form['sg_arg_momentum'] = 0
    latest_form['sg_putt_momentum'] = 0

    # Course history - check historical data if available
    loader = RealDataLoader()
    course_features = loader.load_course_characteristics()

    # Try to get historical course performance from Kaggle data + recent ESPN data
    course_history_list = []
    try:
        # Load Kaggle historical data (2015-2022)
        tournament_df = loader.load_tournament_level_data(min_year=2015, max_year=2022)
        # Find player in historical data (fuzzy match)
        hist_players = tournament_df['player_id'].unique()
        matching_hist = [p for p in hist_players if player_name.lower() in p.lower()]

        if matching_hist:
            hist_player_id = matching_hist[0]
            hist_courses = tournament_df['course'].unique()
            matching_courses = [c for c in hist_courses if course_name.lower() in c.lower()]

            if matching_courses:
                course_id = matching_courses[0]
                course_history = tournament_df[
                    (tournament_df['player_id'] == hist_player_id) &
                    (tournament_df['course'] == course_id)
                ]

                if len(course_history) > 0:
                    for _, row in course_history.iterrows():
                        course_history_list.append({
                            'date': row['date'],
                            'sg_total': row['sg_total'],
                            'made_cut': row['made_cut'],
                            'position': row['position'],
                            'source': 'Kaggle'
                        })

        # Try to get recent tournament history from cached data (2023+)
        scraper = PGATourScraper()
        recent_data = scraper.load_recent_tournament_data()

        # Check for recent results at this course
        if not recent_data.empty and 'player_id' in recent_data.columns:
            # Filter for matching course
            course_match = recent_data[
                recent_data['course'].str.lower().str.contains(course_name.lower(), na=False) |
                recent_data['tournament_name'].str.lower().str.contains(course_name.lower(), na=False)
            ]

            # Find player in recent data
            if not course_match.empty:
                player_recent = course_match[
                    course_match['player_id'].str.lower().str.contains(player_name.lower(), na=False)
                ]
                if len(player_recent) > 0:
                    for _, row in player_recent.iterrows():
                        year = int(row.get('year', 2024))
                        pos = row.get('position_numeric', 999)
                        if pd.isna(pos):
                            pos = 999
                        pos = int(pos)
                        made_cut = int(row.get('made_cut', 1 if pos < 100 else 0))
                        sg_est = row.get('sg_total_est', 0)
                        if pd.isna(sg_est):
                            # Estimate SG from position
                            if pos == 1:
                                sg_est = 3.5
                            elif pos <= 5:
                                sg_est = 2.5
                            elif pos <= 10:
                                sg_est = 1.5
                            elif pos <= 25:
                                sg_est = 0.5
                            elif made_cut:
                                sg_est = 0.0
                            else:
                                sg_est = -1.0

                        course_history_list.append({
                            'date': pd.Timestamp(f'{year}-04-14'),  # Masters typically mid-April
                            'sg_total': float(sg_est),
                            'made_cut': made_cut,
                            'position': pos,
                            'source': 'Recent'
                        })

        # Display combined course history
        if course_history_list:
            # Sort by date
            course_history_list.sort(key=lambda x: x['date'], reverse=True)
            print(f"\n  Course History at {course_name}:")
            for entry in course_history_list:
                result = "Made Cut" if entry['made_cut'] == 1 else "Missed Cut"
                pos = f"Pos: {int(entry['position'])}" if entry['position'] < 999 else ""
                sg_str = f"SG: {entry['sg_total']:+.2f}" if entry['source'] == 'Kaggle' else f"Est SG: {entry['sg_total']:+.1f}"
                date_str = entry['date'].date() if hasattr(entry['date'], 'date') else entry['date']
                source_tag = f"[{entry['source']}]"
                print(f"    {date_str}: {sg_str} {result} {pos} {source_tag}")

            # Calculate averages from history
            sg_values = [e['sg_total'] for e in course_history_list]
            latest_form['course_avg_sg'] = sum(sg_values) / len(sg_values)
            latest_form['course_appearances'] = len(course_history_list)
        else:
            latest_form['course_avg_sg'] = sg_total
            latest_form['course_appearances'] = 0

    except Exception as e:
        print(f"  Note: Could not load historical data: {e}")
        latest_form['course_avg_sg'] = sg_total
        latest_form['course_appearances'] = 0

    # Merge course features
    engineer = FeatureEngineer()
    if not course_features.empty:
        course_df = engineer.engineer_course_features(course_features)
        course_row = course_df[course_df['course_id'].str.lower().str.contains(course_name.lower(), na=False)]
        if len(course_row) > 0:
            print(f"\n  Course: {course_row['course_id'].values[0]}")
            for col in course_row.columns:
                if col != 'course_id':
                    latest_form[col] = course_row[col].values[0]
        else:
            print(f"\n  Course characteristics not found for '{course_name}' - using tour averages")
            latest_form['overall_difficulty'] = 0.5
            latest_form['is_tight_course'] = 0
            latest_form['is_long_course'] = 0
            latest_form['green_challenge'] = 0.5
            latest_form['hazard_density'] = 0.3
            latest_form['approach_difficulty'] = 0.5
            latest_form['yardage'] = 7200
            latest_form['fairway_width_avg'] = 32
            latest_form['slope_rating'] = 140
    else:
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
    X_pred = pd.DataFrame(index=[0])
    for col in feature_cols:
        if col in latest_form.columns:
            X_pred[col] = latest_form[col].values[0]
        else:
            X_pred[col] = 0

    X_pred = X_pred.fillna(0)

    # Make predictions
    print("\n" + "=" * 70)
    print(f"PREDICTED PROBABILITIES: {player_id}")
    print("=" * 70)

    for outcome, predictor in predictors.items():
        prob = predictor.predict_proba(X_pred)[0]
        bar_len = int(prob * 40)
        bar = "#" * bar_len + "-" * (40 - bar_len)
        print(f"\n  {outcome.upper():12s}: {prob*100:5.1f}%  [{bar}]")

    # Context
    print("\n" + "-" * 70)
    print("CONTEXT:")
    print(f"  Current Season SG Total: {sg_total:+.2f}")
    if 'course_appearances' in latest_form.columns:
        apps = int(latest_form['course_appearances'].values[0])
        print(f"  Course Appearances (historical): {apps}")
    if 'course_avg_sg' in latest_form.columns and apps > 0:
        csg = latest_form['course_avg_sg'].values[0]
        print(f"  Course History SG Avg: {csg:+.2f}")

    print("\n  Note: Using current season ESPN stats for form estimation")

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
        "--update-data",
        action="store_true",
        help="Force refresh of ESPN data before predicting"
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
