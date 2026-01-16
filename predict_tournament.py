#!/usr/bin/env python
"""
Weekly Tournament Prediction Script
Run this before each tournament to get course fit rankings.

Usage:
    py -3.11 predict_tournament.py --course "TPC Sawgrass"
    py -3.11 predict_tournament.py --update  # Update stats first
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pga_scraper import PGATourScraper
from src.data_loader import DataLoader, RealDataLoader
from src.feature_engineer import FeatureEngineer
from src.model import CourseFitModel
from src.ranker import CourseFitRanker
import pandas as pd
from pathlib import Path


def update_stats():
    """Scrape latest PGA Tour statistics."""
    print("\n" + "=" * 70)
    print("UPDATING PGA TOUR STATISTICS")
    print("=" * 70)

    scraper = PGATourScraper()
    df = scraper.update_player_stats()

    if df.empty:
        print("\nWarning: Could not fetch live data.")
        print("Using most recent cached data if available.")
        df = scraper.load_latest_scraped_data()

    return df


def load_course_features(course_name: str = None) -> pd.DataFrame:
    """Load course features, optionally filtering to specific course."""
    course_file = Path("data/raw/courses/course_characteristics.csv")

    if course_file.exists():
        courses = pd.read_csv(course_file)
        if course_name:
            # Find matching course (partial match)
            mask = courses['course_id'].str.lower().str.contains(course_name.lower())
            if mask.any():
                return courses[mask]
            else:
                print(f"Warning: Course '{course_name}' not found. Available courses:")
                for c in courses['course_id'].tolist()[:10]:
                    print(f"  - {c}")
        return courses
    else:
        print("Warning: Course characteristics file not found.")
        return pd.DataFrame()


def predict_for_course(player_stats: pd.DataFrame, course_name: str, top_n: int = 25):
    """
    Generate course fit predictions for a specific course.

    Args:
        player_stats: DataFrame with current player statistics
        course_name: Name of the course
        top_n: Number of top players to show
    """
    print("\n" + "=" * 70)
    print(f"COURSE FIT PREDICTIONS: {course_name.upper()}")
    print("=" * 70)

    # Load course features
    courses = load_course_features(course_name)

    if courses.empty:
        print(f"\nError: Could not find course data for '{course_name}'")
        return

    course = courses.iloc[0]
    print(f"\nCourse: {course['course_name']}")
    print(f"Yardage: {course['yardage']}, Par: {course['par']}")
    print(f"Fairway Width: {course['fairway_width_avg']}yds, Green Size: {course['green_size_avg']}sqft")
    print(f"Difficulty: Slope {course['slope_rating']}, Rating {course['course_rating']}")

    # Standardize player_id column name
    if 'player_id' not in player_stats.columns:
        # Try common column names from different sources
        for col in ['Name', 'Player', 'PLAYER', 'player', 'name']:
            if col in player_stats.columns:
                player_stats = player_stats.rename(columns={col: 'player_id'})
                break
        else:
            # Use first column as player name
            player_stats = player_stats.rename(columns={player_stats.columns[0]: 'player_id'})

    # Engineer features
    engineer = FeatureEngineer()
    player_profiles = engineer.create_player_skill_profile(player_stats)
    course_features = engineer.engineer_course_features(courses)

    # Create simple fit scores based on course characteristics
    print("\n" + "-" * 70)
    print("TOP PLAYER-COURSE FITS")
    print("-" * 70)

    # Calculate fit scores based on course demands
    course_row = course_features.iloc[0]
    fits = []

    for _, player in player_profiles.iterrows():
        fit_score = 0

        # Long course favors distance
        if course_row.get('is_long_course', 0):
            fit_score += player.get('distance_profile', 50) * 0.3

        # Tight course favors accuracy
        if course_row.get('is_tight_course', 0):
            fit_score += player.get('accuracy_profile', 50) * 0.4
        else:
            fit_score += player.get('distance_profile', 50) * 0.2

        # Difficult greens favor consistency
        if course_row.get('green_challenge', 0) > 0.5:
            fit_score += player.get('consistency_profile', 50) * 0.3

        # Overall skill
        fit_score += player.get('scoring_profile', 50) * 0.3

        # Boost for players with good SG:Total
        if 'sg_total' in player and pd.notna(player['sg_total']):
            fit_score += player['sg_total'] * 10

        # Get player name
        player_name = player.get('player_id', 'Unknown')

        fits.append({
            'player': player_name,
            'fit_score': fit_score,
            'distance_profile': player.get('distance_profile', 50),
            'accuracy_profile': player.get('accuracy_profile', 50),
            'consistency_profile': player.get('consistency_profile', 50),
            'sg_total': player.get('sg_total', 0) if pd.notna(player.get('sg_total', 0)) else 0,
        })

    # Rank players
    fits_df = pd.DataFrame(fits)
    fits_df = fits_df.sort_values('fit_score', ascending=False).reset_index(drop=True)
    fits_df['rank'] = range(1, len(fits_df) + 1)

    # Display top players
    print(f"\n{'Rank':<6}{'Player':<25}{'Fit Score':<12}{'SG:Total':<10}{'Dist':<8}{'Acc':<8}")
    print("-" * 70)

    for _, row in fits_df.head(top_n).iterrows():
        print(f"{row['rank']:<6}{row['player'][:24]:<25}{row['fit_score']:<12.1f}"
              f"{row['sg_total']:<10.2f}{row['distance_profile']:<8.1f}{row['accuracy_profile']:<8.1f}")

    # Save predictions
    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = course_name.replace(" ", "_").replace("/", "_")
    output_file = output_dir / f"fit_{safe_name}.csv"
    fits_df.to_csv(output_file, index=False)
    print(f"\n  Predictions saved to: {output_file}")

    return fits_df


def main():
    parser = argparse.ArgumentParser(description="PGA Course Fit Predictor")
    parser.add_argument("--course", type=str, help="Course name to predict for")
    parser.add_argument("--update", action="store_true", help="Update player stats first")
    parser.add_argument("--top", type=int, default=25, help="Number of top players to show")
    parser.add_argument("--list-courses", action="store_true", help="List available courses")

    args = parser.parse_args()

    # List courses if requested
    if args.list_courses:
        courses = load_course_features()
        if not courses.empty:
            print("\nAvailable courses:")
            for c in courses['course_id'].tolist():
                print(f"  - {c}")
        return

    # Update stats if requested
    if args.update:
        player_stats = update_stats()
    else:
        # Try to load existing scraped data
        scraper = PGATourScraper()
        player_stats = scraper.load_latest_scraped_data()

        if player_stats.empty:
            print("No cached data found. Fetching fresh stats...")
            player_stats = update_stats()

    if player_stats.empty:
        print("\nError: No player data available.")
        print("Try running with --update flag to fetch fresh data.")
        return

    print(f"\nLoaded stats for {len(player_stats)} players")

    # Predict for course
    if args.course:
        predict_for_course(player_stats, args.course, args.top)
    else:
        # Show available courses
        print("\nNo course specified. Use --course 'Course Name' to predict.")
        print("Use --list-courses to see available courses.")
        print("\nExample:")
        print('  py -3.11 predict_tournament.py --course "TPC Sawgrass" --update')


if __name__ == "__main__":
    main()
