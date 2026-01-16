# Course Fit Model - PGA Tournament Analysis

A comprehensive machine learning project for predicting player-course compatibility in PGA tournaments.

## Project Overview

This project builds a **Gradient Boosting-based course fit model** that predicts which golfers are best suited for specific courses. It demonstrates:

- **Tabular ML** with XGBoost and LightGBM
- **Feature engineering** for complex sports analytics
- **Interaction features** between player skills and course characteristics
- **Model interpretability** using SHAP explainability
- **Real-world ranking** of top players for tournaments

## Features

### Course Feature Engineering
- **Yardage & Par Distribution**: Course length metrics
- **Fairway Width & Green Size**: Course playability characteristics
- **Hazard Density**: Water/bunker severity
- **Elevation & Slope**: Topographical difficulty
- **Difficulty Scoring**: Composite difficulty metrics

### Player Skill Profiles
- Driving Distance & Accuracy
- Greens in Regulation (GIR)
- Scrambling Ability
- Putting Performance
- Short Game & Approach Play

### Interaction Features
- **Accuracy-Tight Course Fit**: Accurate players excel on tight courses
- **Distance-Long Course Fit**: Long hitters suit lengthy courses
- **Consistency-Difficulty Fit**: Consistent players handle hard courses
- **Hazard Experience**: Accurate players navigate hazards better

## Project Structure

```
pga-analysis/
├── data/                          # Data directory
│   ├── player_stats.csv
│   ├── course_features.csv
│   └── tournament_results.csv
├── src/
│   ├── data_loader.py            # Data loading & sample generation
│   ├── feature_engineer.py       # Feature engineering (course & interaction)
│   ├── model.py                  # XGBoost/LightGBM training
│   ├── explainer.py              # SHAP interpretability
│   ├── ranker.py                 # Ranking & tournament analysis
│   └── pipeline.py               # Full end-to-end pipeline
├── notebooks/
│   └── course_fit_analysis.py    # Interactive analysis script
├── models/                        # Trained models saved here
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Installation

```bash
# Create virtual environment (optional)
python -m venv venv
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Option 1: Run Full Pipeline
```python
from src.pipeline import run_course_fit_pipeline

results = run_course_fit_pipeline(model_type='xgboost', use_shap=True)
```

### Option 2: Interactive Analysis
```bash
python notebooks/course_fit_analysis.py
```

### Option 3: Step-by-Step in Python
```python
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model import CourseFitModel
from src.explainer import ShapExplainer
from src.ranker import CourseFitRanker

# Load data
loader = DataLoader()
player_stats, course_features, tournament_results = loader.load_data()

# Engineer features
engineer = FeatureEngineer()
X, y = engineer.create_player_course_interactions(player_stats, course_features, tournament_results)

# Train model
model = CourseFitModel(model_type='xgboost')
model.train(X, y)

# Explain predictions
explainer = ShapExplainer(model.model, X)
global_importance = explainer.global_feature_importance(X)

# Rank players
ranker = CourseFitRanker(model, explainer)
rankings = ranker.rank_players_for_tournament(X, course_features['course_id'].unique()[:5])
```

## Key Components

### 1. Data Loading (`data_loader.py`)
- Generates synthetic PGA data (players, courses, tournament results)
- Supports real data loading from CSV files
- Realistic score simulation based on player-course interactions

### 2. Feature Engineering (`feature_engineer.py`)
- **Course Features**: Difficulty, layout, green challenge, hazard density
- **Player Profiles**: Skill aggregation (accuracy, distance, consistency, scoring)
- **Interaction Features**: 8+ interaction terms between player skills and course traits

### 3. Model Training (`model.py`)
- XGBoost & LightGBM support
- Automatic hyperparameter selection
- Comprehensive evaluation metrics (RMSE, MAE, R²)
- Model persistence (save/load)

### 4. Explainability (`explainer.py`)
- SHAP-based model interpretability
- Global feature importance
- Local prediction explanations
- Interaction analysis
- Force plots & summary plots

### 5. Ranking Engine (`ranker.py`)
- Per-course player rankings
- Tournament aggregate rankings
- Player-course compatibility analysis
- Course difficulty assessment
- Heatmap visualization
- Strategic insights

## Example Output

```
COURSE RANKINGS:
Course_1 - Top 5 Best Fits:
  Rank  Player         Fit Score  Fit Percentile
  1     Player_15      68.32      95%
  2     Player_42      69.18      92%
  3     Player_7       69.75      88%

TOURNAMENT RANKING:
  Rank  Player         Aggregate Score  Courses Ranked
  1     Player_15      69.12           5
  2     Player_42      70.05           5
  3     Player_7       70.18           5

KEY INSIGHTS:
- Total Player-Course Pairs: 1,250
- Players Evaluated: 50
- Courses: 20
- Easiest Course: Course_5 (mean fit: 71.2)
- Hardest Course: Course_15 (mean fit: 68.9)
- Most Selective: Course_12 (std: 1.8)
```

## SHAP Explainability Examples

### Global Importance
Shows which features most influence predictions across all cases:
```python
shap_importance = explainer.global_feature_importance(X, top_n=15)
```

### Local Explanation
Explains why a specific player-course combination was rated a certain way:
```python
explanation = explainer.local_explanation(X, instance_idx=42, top_n=10)
# Shows: "This player scored well because high accuracy (SHAP: +2.3) 
#         combined with tight course (SHAP: +1.8) are perfect fit"
```

### Force Plot
Visualizes all feature contributions to a single prediction:
```python
explainer.force_plot(X, instance_idx=0)
```

## ML Techniques Used

### Gradient Boosting
- **XGBoost**: Industry standard for tabular data
- **LightGBM**: Fast, memory-efficient alternative
- Captures non-linear relationships between features
- Handles feature interactions automatically

### Feature Engineering
- Domain-specific features (course difficulty, fairway width)
- Interaction terms (accuracy × fairway width)
- Polynomial features for non-linearity
- Normalized skill profiles

### Model Interpretability
- **SHAP (SHapley Additive exPlanations)**: Game theory-based explanations
- Global feature importance
- Local prediction reasoning
- Feature interaction analysis

## Advanced Features

### Tournament Strategy
```python
# Find best-fit players for a 5-course tournament
tournament_ranking = ranker.tournament_aggregate_ranking(
    rankings,
    aggregation_method='mean'  # or 'median' or 'min'
)
```

### Player Analysis
```python
# Analyze player's performance across all courses
player_profile = ranker.player_course_profile(X, 'Player_15')
print(player_profile.sort_values('predicted_fit_score'))
```

### Course Analysis
```python
# How selective is each course?
course_difficulty = ranker.course_difficulty_variance(X)
```

### Visualization
```python
# Heatmap of all player-course fits
fig = ranker.create_fit_heatmap(X, sample_players=20, sample_courses=10)
plt.show()
```

## Model Performance

Example metrics on sample data:
- **XGBoost Test RMSE**: ~0.85 strokes
- **LightGBM Test RMSE**: ~0.88 strokes
- **R² Score**: 0.72+

Better performance with:
- Real historical tournament data
- Extended feature set (weather, player form, course conditions)
- Time-series features (recent performance trends)

## Use Cases

1. **Tournament Strategy**: Identify best-suited players for specific courses
2. **Player Development**: Understand where players excel and struggle
3. **Course Design**: Predict how course changes affect different players
4. **Betting Models**: Incorporate course fit into odds calculations
5. **Fantasy Golf**: Optimize lineup selection for weekly tournaments

## Dependencies

- pandas: Data manipulation
- numpy: Numerical computing
- scikit-learn: ML preprocessing & metrics
- xgboost: Gradient boosting model
- lightgbm: Alternative boosting model
- shap: Model explainability
- matplotlib & seaborn: Visualization

## Future Enhancements

- [ ] Weather feature integration
- [ ] Player form/momentum tracking
- [ ] Crowd size & noise impact
- [ ] Historical head-to-head performance
- [ ] Real-time prediction updates
- [ ] Web API for deployment
- [ ] Bayesian uncertainty estimation
- [ ] Multi-task learning (multiple tournaments)

## License

MIT License - Feel free to use for learning and analysis.

---

**Author**: PGA Analytics  
**Version**: 1.0  
**Date**: January 2026
