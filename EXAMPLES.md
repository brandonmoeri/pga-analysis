# Advanced Examples & Workflows

This document provides advanced examples for using the Course Fit Model.

## Example 1: Complete End-to-End Pipeline

```python
import sys
sys.path.insert(0, '.')

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model import CourseFitModel
from src.explainer import ShapExplainer
from src.ranker import CourseFitRanker

# 1. Load Data
print("Loading data...")
loader = DataLoader()
player_stats, course_features, tournament_results = loader.load_data()

# 2. Engineer Features
print("Engineering features...")
engineer = FeatureEngineer()
X, y = engineer.create_player_course_interactions(
    player_stats, course_features, tournament_results
)
print(f"Created {len(X.columns)} features from {len(X)} samples")

# 3. Train Model
print("Training model...")
model = CourseFitModel(model_type='xgboost')
metrics = model.train(X, y, test_size=0.2)
print(f"Test R²: {metrics['test_r2']:.4f}")

# 4. Create Explainer
print("Generating explanations...")
X_features = X[[col for col in X.columns if col not in ['player_id', 'course_id']]]
explainer = ShapExplainer(model.model, X_features)

# 5. Rank Players
print("Creating rankings...")
ranker = CourseFitRanker(model, explainer)
courses_to_rank = course_features['course_id'].unique()[:5]
rankings = ranker.rank_players_for_tournament(X, courses_to_rank, top_n=10)

# 6. Tournament Analysis
tournament_ranking = ranker.tournament_aggregate_ranking(rankings)
print("\nTop 5 Players Overall:")
print(tournament_ranking.head(5))

# 7. Save Model
model.save_model()
print("Model saved!")
```

---

## Example 2: Analyzing Individual Player Fit

```python
# Find a specific player
target_player = 'Player_15'

# Get their fit across all courses
player_profile = ranker.player_course_profile(X, target_player)

# Best 5 courses
print(f"\n{target_player} - Best Fit Courses:")
best_courses = player_profile.nsmallest(5, 'predicted_fit_score')
print(best_courses[['course_id', 'predicted_fit_score']])

# Worst 5 courses
print(f"\n{target_player} - Worst Fit Courses:")
worst_courses = player_profile.nlargest(5, 'predicted_fit_score')
print(worst_courses[['course_id', 'predicted_fit_score']])

# Why does this player excel at their best course?
best_course_id = best_courses.iloc[0]['course_id']
best_explanation = explainer.player_course_interaction_explanation(
    X, (target_player, best_course_id), X
)
print(f"\nWhy {target_player} excels at {best_course_id}:")
print(best_explanation['top_contributing_features'])

# Why do they struggle at their worst course?
worst_course_id = worst_courses.iloc[0]['course_id']
worst_explanation = explainer.player_course_interaction_explanation(
    X, (target_player, worst_course_id), X
)
print(f"\nWhy {target_player} struggles at {worst_course_id}:")
print(worst_explanation['top_contributing_features'])
```

---

## Example 3: Comparing Two Players

```python
player_a = 'Player_10'
player_b = 'Player_20'

# Get profiles
profile_a = ranker.player_course_profile(X, player_a)
profile_b = ranker.player_course_profile(X, player_b)

# Head-to-head for specific course
test_course = 'Course_5'
score_a = profile_a[profile_a['course_id'] == test_course]['predicted_fit_score'].values[0]
score_b = profile_b[profile_b['course_id'] == test_course]['predicted_fit_score'].values[0]

print(f"\n{test_course} - Head-to-Head:")
print(f"{player_a}: {score_a:.2f} (fit rank: {profile_a[profile_a['course_id']==test_course]['rank'].values[0]})")
print(f"{player_b}: {score_b:.2f}")

winner = player_a if score_a < score_b else player_b
print(f"\nExpected: {winner} fits {test_course} better by {abs(score_a - score_b):.2f} strokes")

# Explain why
winner_explanation = explainer.player_course_interaction_explanation(
    X, (winner, test_course), X
)
print(f"\nKey factors helping {winner}:")
print(winner_explanation['top_contributing_features'].head(5))
```

---

## Example 4: Course Difficulty Analysis

```python
# Analyze course characteristics
course_stats = ranker.course_difficulty_variance(X)

print("Course Difficulty Ranking:")
print(course_stats[['course_id', 'mean_fit_score', 'std_fit_score', 'selectivity']].sort_values('mean_fit_score'))

# Find courses with specific difficulty levels
easy_courses = course_stats[course_stats['mean_fit_score'] < course_stats['mean_fit_score'].quantile(0.25)]
hard_courses = course_stats[course_stats['mean_fit_score'] > course_stats['mean_fit_score'].quantile(0.75)]

print(f"\nEasiest courses: {easy_courses['course_id'].tolist()}")
print(f"Hardest courses: {hard_courses['course_id'].tolist()}")

# Which courses are most selective (high variance)?
print(f"\nMost selective courses (high variance):")
print(course_stats.nlargest(5, 'selectivity')[['course_id', 'selectivity']])

print(f"\nLeast selective courses (low variance):")
print(course_stats.nsmallest(5, 'selectivity')[['course_id', 'selectivity']])
```

---

## Example 5: Feature Importance Analysis

```python
# Get traditional feature importance
importance = model.get_feature_importance(top_n=20)
print("XGBoost Feature Importance (Top 20):")
print(importance)

# Get SHAP-based importance
shap_importance = explainer.global_feature_importance(X_features, top_n=20)
print("\nSHAP Global Feature Importance (Top 20):")
print(shap_importance)

# Identify interaction features
interaction_features = [col for col in X.columns if 'fit' in col or 'interaction' in col]
print(f"\nInteraction features impact:")
shap_interaction = shap_importance[shap_importance['feature'].isin(interaction_features)]
print(shap_interaction)

# These show how much player-course compatibility matters
```

---

## Example 6: Model Comparison (XGBoost vs LightGBM)

```python
from src.model import CourseFitModel

# Train XGBoost
print("Training XGBoost...")
xgb_model = CourseFitModel(model_type='xgboost')
xgb_metrics = xgb_model.train(X, y)
xgb_importance = xgb_model.get_feature_importance(top_n=10)

# Train LightGBM
print("\nTraining LightGBM...")
lgb_model = CourseFitModel(model_type='lightgbm')
lgb_metrics = lgb_model.train(X, y)
lgb_importance = lgb_model.get_feature_importance(top_n=10)

# Compare metrics
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"{'Metric':<20} {'XGBoost':<15} {'LightGBM':<15}")
print("-"*60)
print(f"{'Test RMSE':<20} {xgb_metrics['test_rmse']:<15.4f} {lgb_metrics['test_rmse']:<15.4f}")
print(f"{'Test MAE':<20} {xgb_metrics['test_mae']:<15.4f} {lgb_metrics['test_mae']:<15.4f}")
print(f"{'Test R²':<20} {xgb_metrics['test_r2']:<15.4f} {lgb_metrics['test_r2']:<15.4f}")
print(f"{'Train R²':<20} {xgb_metrics['train_r2']:<15.4f} {lgb_metrics['train_r2']:<15.4f}")

# Compare feature importance
print("\nTop 5 Features - XGBoost:")
print(xgb_importance.head())
print("\nTop 5 Features - LightGBM:")
print(lgb_importance.head())
```

---

## Example 7: Visualization Examples

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Player-Course Fit Heatmap
fig = ranker.create_fit_heatmap(X, sample_players=15, sample_courses=10)
plt.title("Player-Course Fit Score Heatmap\n(Lower = Better)")
plt.tight_layout()
plt.show()

# 2. Feature Importance Bar Plot
importance = model.get_feature_importance(top_n=15)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importance['feature'], importance['importance'])
ax.set_xlabel("Importance Score")
ax.set_title("Top 15 Features (XGBoost)")
plt.tight_layout()
plt.show()

# 3. Course Difficulty Distribution
course_stats = ranker.course_difficulty_variance(X)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.barh(course_stats['course_id'], course_stats['mean_fit_score'])
ax1.set_xlabel("Mean Fit Score (lower = harder)")
ax1.set_title("Course Difficulty Ranking")

ax2.barh(course_stats['course_id'], course_stats['std_fit_score'])
ax2.set_xlabel("Standard Deviation (selectivity)")
ax2.set_title("Course Selectivity")
plt.tight_layout()
plt.show()

# 4. SHAP Summary Plot
fig = explainer.summary_plot(X_features, plot_type='bar')
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()

# 5. Player Fit Distribution
player_fits = ranker.player_course_profile(X, 'Player_15')
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(player_fits['predicted_fit_score'], bins=20, edgecolor='black')
ax.set_xlabel("Fit Score")
ax.set_ylabel("Number of Courses")
ax.set_title("Player_15 - Fit Score Distribution Across Courses")
plt.tight_layout()
plt.show()
```

---

## Example 8: Custom Tournament Scenario

```python
# Scenario: Select 5-course tournament where we want to pick best players

# 1. Define tournament courses
tournament_courses = ['Course_3', 'Course_7', 'Course_12', 'Course_18', 'Course_5']

# 2. Get rankings for each course
rankings = ranker.rank_players_for_tournament(X, tournament_courses, top_n=20)

# 3. Aggregate results
tournament_ranking = ranker.tournament_aggregate_ranking(
    rankings,
    aggregation_method='mean'  # Average fit across courses
)

# 4. Select top 12 players for tournament
selected_players = tournament_ranking.head(12)
print("Selected Players for Tournament:")
print(selected_players[['tournament_rank', 'player_id', 'aggregate_fit_score']])

# 5. Create detailed breakdown for each course
print("\nDetailed Breakdown by Course:")
for course in tournament_courses:
    print(f"\n{course} - Best Fits:")
    course_ranking = rankings[course].head(3)
    for idx, row in course_ranking.iterrows():
        print(f"  {row['rank']}. {row['player_id']}: {row['predicted_fit_score']:.2f}")

# 6. Identify tournament dynamics
print("\nTournament Insights:")
print(f"- Tournament is well-suited for: {selected_players.iloc[0]['player_id']}")
print(f"- Most variable player: ", end="")
player_variance = tournament_ranking.set_index('player_id').join(
    X.groupby('player_id')['predicted_fit_score'].std().rename('variance'),
    how='left'
)
print(player_variance['variance'].idxmax())
```

---

## Example 9: SHAP Local Explanations

```python
# Get detailed explanation for a single prediction

# Find a prediction to explain
test_idx = 100
player_id = X.iloc[test_idx]['player_id']
course_id = X.iloc[test_idx]['course_id']

print(f"Explaining: {player_id} fit for {course_id}")
print("="*60)

# Get SHAP explanation
explanation = explainer.local_explanation(X, test_idx, top_n=15)
print("\nTop Contributing Features:")
print(explanation[['feature', 'value', 'shap_value']])

# Interpretation
print("\nInterpretation:")
for idx, row in explanation.head(5).iterrows():
    direction = "helps" if row['shap_value'] > 0 else "hurts"
    magnitude = abs(row['shap_value'])
    print(f"- {row['feature']}: {row['value']:.2f} ({direction} fit by {magnitude:.2f})")
```

---

## Example 10: Production Deployment

```python
# Save everything for deployment
import joblib
import json

# Save trained model
model.save_model('models/course_fit_model.pkl')

# Save feature names and preprocessing info
metadata = {
    'model_type': 'xgboost',
    'features': model.feature_names,
    'test_r2': metrics['test_r2'],
    'test_rmse': metrics['test_rmse'],
    'version': '1.0'
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Model saved for production!")

# Later, load and use:
model_loaded = CourseFitModel(model_type='xgboost')
model_loaded.load_model('models/course_fit_model.pkl')

# Get predictions on new data
predictions = model_loaded.predict_fit_score(X_new)
```

---

## Performance Tips

1. **Faster Training**: Use `use_shap=False` to skip SHAP explanations
2. **Large Datasets**: Use LightGBM instead of XGBoost (faster)
3. **SHAP Sampling**: Use a sample of data for SHAP analysis
4. **Parallel Processing**: Use `n_jobs=-1` in model hyperparameters

```python
# Fast pipeline for large data
results = run_course_fit_pipeline(
    model_type='lightgbm',  # Faster
    use_shap=False           # Skip explanations
)
```

---

See README.md and SETUP.md for more information!
