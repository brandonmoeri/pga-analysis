#!/usr/bin/env python
"""
PGA Analysis CLI - Unified command-line interface for predictions, rankings, and insights.
Replaces run.py, predict_tournament.py, and predict_outcomes.py.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent))

from backend.app.services.outcomes import OutcomeService
from backend.app.services.course_fit import CourseFitService
from backend.app.services.ranking import RankingService
from backend.app.services.explanations import ExplanationService
from backend.app.services.data import DataService
from backend.app.services.scraper import ScraperService


def predict_command(args):
    """Predict player outcomes at a course."""
    print(f"\n{'='*70}")
    print(f"PLAYER OUTCOME PREDICTION")
    print(f"{'='*70}\n")
    
    player_id = args.player
    course_id = args.course
    
    print(f"Player:  {player_id}")
    print(f"Course:  {course_id}\n")
    
    # Get course fit
    fit = CourseFitService.predict_player_course_fit(player_id, course_id)
    if fit:
        print(f"Course Fit Score: {fit['fit_score']:.2f} strokes")
        print(f"  ({fit['interpretation']})\n")
    
    # Get outcomes
    outcomes = OutcomeService.predict_player_outcomes(player_id, course_id)
    if outcomes:
        print("Tournament Outcomes:")
        print(f"  Make Cut:     {outcomes['make_cut_probability']:.1%}")
        print(f"  Top 10:       {outcomes['top_10_probability']:.1%}")
        print(f"  Win:          {outcomes['win_probability']:.1%}\n")
        print(f"Confidence:   {outcomes['overall_confidence']:.1%}")
    
    print(f"\n{'='*70}\n")


def rank_command(args):
    """Rank players for tournament."""
    print(f"\n{'='*70}")
    print(f"TOURNAMENT RANKING")
    print(f"{'='*70}\n")
    
    tournament = args.tournament or "Upcoming Tournament"
    courses = args.courses or ["augusta_national"]
    top_n = args.top_n or 50
    
    print(f"Tournament: {tournament}")
    print(f"Courses:    {', '.join(courses)}")
    print(f"Top:        {top_n} players\n")
    
    # Get rankings
    ranking = RankingService.aggregate_tournament_ranking(
        tournament_name=tournament,
        courses=courses,
        top_n=top_n
    )
    
    if ranking:
        print("Top Players:")
        print(f"{'Rank':<5} {'Player':<25} {'Fit Score':<15} {'Courses':<10}")
        print("-" * 55)
        for item in ranking["aggregate_ranking"][:10]:
            print(f"{item['rank']:<5} {item['player_id']:<25} {item['average_fit_score']:>+6.2f}      {item['courses_analyzed']:<10}")
        
        if len(ranking["aggregate_ranking"]) > 10:
            print(f"\n... and {len(ranking['aggregate_ranking']) - 10} more players")
    
    print(f"\n{'='*70}\n")


def explain_command(args):
    """Explain prediction using SHAP."""
    print(f"\n{'='*70}")
    print(f"PREDICTION EXPLANATION (SHAP)")
    print(f"{'='*70}\n")
    
    player_id = args.player
    course_id = args.course
    top_n = args.top_n or 10
    
    print(f"Player:  {player_id}")
    print(f"Course:  {course_id}")
    print(f"Top Features: {top_n}\n")
    
    explanation = ExplanationService.get_local_explanation(
        player_id=player_id,
        course_id=course_id,
        top_n=top_n
    )
    
    if explanation:
        print(f"Prediction: {explanation['prediction']:+.2f} strokes")
        print(f"Base Value: {explanation['base_value']:+.2f}\n")
        
        print("Top Contributing Features:")
        print(f"{'Feature':<30} {'SHAP Value':<15} {'Feature Value':<15}")
        print("-" * 60)
        for feature in explanation["top_features"]:
            print(f"{feature['feature']:<30} {feature['shap_value']:>+8.4f}      {feature['feature_value']:>8.2f}")
    
    print(f"\n{'='*70}\n")


def feature_importance_command(args):
    """Show global feature importance."""
    print(f"\n{'='*70}")
    print(f"GLOBAL FEATURE IMPORTANCE")
    print(f"{'='*70}\n")
    
    top_n = args.top_n or 15
    
    features = ExplanationService.get_global_feature_importance(top_n=top_n)
    
    if features:
        print(f"Top {top_n} Features by Importance:\n")
        print(f"{'Rank':<5} {'Feature':<30} {'Importance':<15}")
        print("-" * 50)
        for feature in features:
            print(f"{feature['rank']:<5} {feature['feature']:<30} {feature['importance']:>8.4f}")
    
    print(f"\n{'='*70}\n")


def update_command(args):
    """Update player statistics from web."""
    print(f"\n{'='*70}")
    print(f"UPDATING PLAYER STATISTICS")
    print(f"{'='*70}\n")
    
    result = ScraperService.update_player_stats()
    
    if result:
        print(f"Status:   {result.get('status', 'unknown').upper()}")
        print(f"Updated:  {result.get('players_updated', 0)} players")
        print(f"Time:     {result.get('timestamp', datetime.now().isoformat())}")
        if result.get('error'):
            print(f"Error:    {result['error']}")
    
    print(f"\n{'='*70}\n")


def info_command(args):
    """Show data information."""
    print(f"\n{'='*70}")
    print(f"DATA INFORMATION")
    print(f"{'='*70}\n")
    
    data_info = DataService.get_all_data()
    player_info = DataService.get_player_stats_dict()
    course_info = DataService.get_course_features_dict()
    scraper_info = ScraperService.get_latest_data_info()
    
    print("Available Data:")
    print(f"  Players:     {player_info['count']}")
    print(f"  Courses:     {course_info['count']}")
    if course_info.get('courses'):
        print(f"  Courses:     {', '.join(course_info['courses'][:3])}{'...' if len(course_info['courses']) > 3 else ''}")
    
    print(f"\nData Sources:")
    for source in scraper_info.get('data_sources', []):
        print(f"  - {source}")
    
    print(f"\nLast Updated: {scraper_info.get('last_updated', 'Unknown')}")
    
    if scraper_info.get('coverage'):
        coverage = scraper_info['coverage']
        print(f"\nCoverage:")
        print(f"  Players:     {coverage.get('players', 0)}")
        print(f"  Tournaments: {coverage.get('tournaments_year', 0)}/year")
        if coverage.get('stats_available'):
            print(f"  Stats:       {len(coverage['stats_available'])} types available")
    
    print(f"\n{'='*70}\n")


def api_info_command(args):
    """Show API information."""
    print(f"\n{'='*70}")
    print(f"API ENDPOINTS")
    print(f"{'='*70}\n")
    
    print("FastAPI server is running (if started with 'python -m uvicorn backend.app.main:app --reload')\n")
    
    print("Available Endpoints:\n")
    
    endpoints = {
        "Health": [
            "GET  /api/health                    - API health status",
            "GET  /api/health/models             - Detailed model status",
        ],
        "Predictions": [
            "POST /api/predictions/player-outcome - Predict player outcomes",
            "GET  /api/predictions/historical/{player_id} - Prediction history",
        ],
        "Rankings": [
            "POST /api/rankings/tournament       - Tournament rankings",
            "GET  /api/rankings/player/{id}/course-fits - Player course fits",
            "GET  /api/rankings/course/{id}/difficulty - Course difficulty",
        ],
        "Explanations": [
            "POST /api/explanations/local        - SHAP explanation",
            "GET  /api/explanations/feature-importance - Global importance",
        ],
        "Data": [
            "POST /api/data/update-stats         - Update player stats",
            "GET  /api/data/stats/player/{id}    - Player statistics",
            "GET  /api/data/courses              - Available courses",
            "GET  /api/data/info                 - Data metadata",
        ],
    }
    
    for category, endpoints_list in endpoints.items():
        print(f"{category}:")
        for endpoint in endpoints_list:
            print(f"  {endpoint}")
        print()
    
    print("Interactive API Docs:  http://localhost:8000/docs")
    print("Alternative Docs:      http://localhost:8000/redoc")
    
    print(f"\n{'='*70}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PGA Analysis CLI - Predictions, Rankings, and Insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict player outcomes
  python cli.py predict --player scheffler_scottie --course augusta_national
  
  # Get tournament rankings
  python cli.py rank --tournament masters_2024 --courses augusta_national --top 50
  
  # Explain prediction
  python cli.py explain --player rory_mcilroy --course pebble_beach --top 10
  
  # Show feature importance
  python cli.py features --top 15
  
  # Update data
  python cli.py update
  
  # Show data info
  python cli.py info
  
  # Show API info
  python cli.py api
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict player outcomes")
    predict_parser.add_argument("--player", required=True, help="Player ID")
    predict_parser.add_argument("--course", required=True, help="Course ID")
    predict_parser.set_defaults(func=predict_command)
    
    # Rank command
    rank_parser = subparsers.add_parser("rank", help="Rank players for tournament")
    rank_parser.add_argument("--tournament", help="Tournament name")
    rank_parser.add_argument("--courses", nargs="+", help="Course IDs")
    rank_parser.add_argument("--top", type=int, dest="top_n", default=50, help="Top N players (default: 50)")
    rank_parser.set_defaults(func=rank_command)
    
    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Explain prediction using SHAP")
    explain_parser.add_argument("--player", required=True, help="Player ID")
    explain_parser.add_argument("--course", required=True, help="Course ID")
    explain_parser.add_argument("--top", type=int, dest="top_n", default=10, help="Top features (default: 10)")
    explain_parser.set_defaults(func=explain_command)
    
    # Features command
    features_parser = subparsers.add_parser("features", help="Show global feature importance")
    features_parser.add_argument("--top", type=int, dest="top_n", default=15, help="Top features (default: 15)")
    features_parser.set_defaults(func=feature_importance_command)
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update player statistics")
    update_parser.set_defaults(func=update_command)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show data information")
    info_parser.set_defaults(func=info_command)
    
    # API command
    api_parser = subparsers.add_parser("api", help="Show API information")
    api_parser.set_defaults(func=api_info_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    try:
        args.func(args)
    except Exception as e:
        print(f"\nError: {e}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()