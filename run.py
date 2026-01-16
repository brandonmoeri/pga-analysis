#!/usr/bin/env python
"""
Quick start script for Course Fit Model.
Run this to train both models and generate insights.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from src.pipeline import run_course_fit_pipeline
    
    print("\n" + "=" * 70)
    print("PGA COURSE FIT MODEL - Quick Start")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Load real PGA Tour data (2020-2022)")
    print("  2. Engineer course and interaction features")
    print("  3. Train XGBoost model")
    print("  4. Generate SHAP explanations")
    print("  5. Rank best-fit players for courses")
    print("  6. Display tournament insights")
    print("\n" + "-" * 70)

    # Run pipeline with real data
    results = run_course_fit_pipeline(
        model_type='xgboost',
        use_shap=True,
        use_real_data=True,
        min_year=2020, # 2015 - 2022 data available
        max_year=2022
    )
    
    print("\n" + "=" * 70)
    print("SUCCESS! Results saved to models/")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Review rankings for tournament strategy")
    print("  - Use explainer to understand predictions")
    print("  - Analyze player-course fit profiles")
    print("  - Run: python notebooks/course_fit_analysis.py")
    print("\n")
