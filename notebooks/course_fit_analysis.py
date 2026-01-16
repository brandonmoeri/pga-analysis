"""
Jupyter notebook for interactive exploration of Course Fit Model.
Run this to train models, visualize results, and generate insights.
"""

import sys
sys.path.insert(0, '.')

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model import CourseFitModel
from src.explainer import ShapExplainer
from src.ranker import CourseFitRanker
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Course Fit Model - Interactive Notebook")
print("=" * 60)

# Load Data
print("\n1. Loading Data...")
loader = DataLoader()
player_stats, course_features, tournament_results = loader.load_data()

print(f"   Loaded {len(player_stats)} players")
print(f"   Loaded {len(course_features)} courses")
print(f"   Loaded {len(tournament_results)} tournament results")

# Feature Engineering
print("\n2. Engineering Features...")
engineer = FeatureEngineer()
X, y = engineer.create_player_course_interactions(
    player_stats, course_features, tournament_results
)

print(f"   Created {len(X.columns) - 2} features")
print(f"   Data shape: {X.shape}")

# Train XGBoost
print("\n3. Training XGBoost Model...")
xgb_model = CourseFitModel(model_type='xgboost')
xgb_metrics = xgb_model.train(X, y)

print("\nXGBoost Feature Importance (Top 10):")
xgb_importance = xgb_model.get_feature_importance(top_n=10)
print(xgb_importance)

# Train LightGBM
print("\n4. Training LightGBM Model...")
lgb_model = CourseFitModel(model_type='lightgbm')
lgb_metrics = lgb_model.train(X, y)

print("\nLightGBM Feature Importance (Top 10):")
lgb_importance = lgb_model.get_feature_importance(top_n=10)
print(lgb_importance)

# SHAP Analysis
print("\n5. Generating SHAP Explanations...")
X_features = X[[col for col in X.columns if col not in ['player_id', 'course_id']]]
xgb_explainer = ShapExplainer(xgb_model.model, X_features)

shap_importance = xgb_explainer.global_feature_importance(X_features, top_n=15)
print("\nGlobal Feature Importance (SHAP):")
print(shap_importance)

# Ranking Analysis
print("\n6. Ranking Players by Course Fit...")
ranker = CourseFitRanker(xgb_model, xgb_explainer)

# Get sample courses
sample_courses = course_features['course_id'].unique()[:3]
rankings = ranker.rank_players_for_tournament(X, sample_courses, top_n=10)

for course_id in sample_courses:
    print(f"\n{course_id} - Top 5 Best Fits:")
    print(rankings[course_id].head(5)[['rank', 'player_id', 'predicted_fit_score']])

# Tournament Ranking
print("\n7. Tournament Aggregate Ranking...")
tournament_ranking = ranker.tournament_aggregate_ranking(rankings)
print("\nTop 10 Players Overall:")
print(tournament_ranking.head(10))

# Player-Specific Analysis
print("\n8. Player Fit Analysis...")
sample_player = player_stats['player_id'].iloc[0]
player_profile = ranker.player_course_profile(X, sample_player)
print(f"\n{sample_player} - Fit Across Courses:")
print(player_profile.head(10))

# Course Difficulty
print("\n9. Course Difficulty Analysis...")
course_difficulty = ranker.course_difficulty_variance(X)
print("\nCourse Difficulty Metrics:")
print(course_difficulty)

# Model Comparison
print("\n10. Model Performance Comparison")
print("\n" + "=" * 60)
print("Metric           | XGBoost    | LightGBM")
print("-" * 60)
print(f"Test RMSE        | {xgb_metrics['test_rmse']:10.4f} | {lgb_metrics['test_rmse']:10.4f}")
print(f"Test MAE         | {xgb_metrics['test_mae']:10.4f} | {lgb_metrics['test_mae']:10.4f}")
print(f"Test R²          | {xgb_metrics['test_r2']:10.4f} | {lgb_metrics['test_r2']:10.4f}")

print("\n" + "=" * 60)
print("Pipeline complete! Results saved to models/ directory")
