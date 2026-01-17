# PGA Course Fit & Tournament Outcome Prediction

A comprehensive machine learning project for predicting player-course compatibility and tournament outcomes in professional golf.

## Project Overview

This project combines **two predictive models**:

1. **Course Fit Model** - Regression model predicting which golfers are best suited for specific courses
2. **Tournament Outcome Predictor** - Classification model predicting probabilities of making the cut, top-10 finishes, and wins

### Key ML Techniques Demonstrated

- **Gradient Boosting** with XGBoost/LightGBM
- **Probability Calibration** (Platt scaling, isotonic regression)
- **Rolling Form Features** with strict label leakage prevention
- **Time-Aware Train/Test Splits** for temporal data
- **SHAP Explainability** for model interpretation
- **Feature Engineering** for sports analytics

## Quick Start

### Train Outcome Prediction Models
```bash
py -3.11 predict_outcomes.py --train
```

### Predict Player Outcomes at a Course
```bash
py -3.11 predict_outcomes.py --predict --player "Scheffler" --course "Augusta"
```

Example output:
```
PREDICTED PROBABILITIES: S. Scheffler at Augusta National Golf Club
======================================================================

  MADE_CUT    :  87.7%  [###################################-----]
  TOP_10      :  35.7%  [##############--------------------------]
  WIN         :   6.8%  [##--------------------------------------]

CONTEXT:
  Recent Form (Last 5 SG Avg): +1.04
  Course Appearances: 1
  Course History SG Avg: +4.36
```

### Run Course Fit Pipeline
```bash
py -3.11 predict_tournament.py --course "TPC Sawgrass"
```

## Features

### Tournament Outcome Prediction (NEW)

Predicts three binary outcomes with calibrated probabilities:

| Outcome | Description | Base Rate |
|---------|-------------|-----------|
| **Made Cut** | Did player make the cut? | ~58% |
| **Top-10** | Did player finish top-10? | ~9% |
| **Win** | Did player win? | ~0.8% |

**Rolling Form Features** (with leakage prevention):
- `sg_total_last_5` / `sg_total_last_10` - Recent strokes gained averages
- `sg_*_momentum` - Form trend (improving vs declining)
- `course_avg_sg` - Historical performance at specific course
- `course_appearances` - Experience at the course

**Leakage Prevention**: Uses `shift(1)` before rolling calculations to ensure current tournament is never included in its own features.

### Course Fit Prediction

Predicts player-course compatibility using:
- **Player Skill Profiles**: Driving distance/accuracy, GIR, scrambling, putting
- **Course Characteristics**: Yardage, fairway width, hazard density, slope rating
- **Interaction Features**: Accuracy × tight fairways, distance × long courses

## Project Structure

```
pga-analysis/
├── data/
│   ├── raw/
│   │   ├── kaggle/                    # PGA Tour 2015-2022 data
│   │   └── courses/                   # Course characteristics
│   └── processed/                     # Generated data files
├── src/
│   ├── data_loader.py                 # Data loading (season & tournament level)
│   ├── feature_engineer.py            # Feature engineering
│   ├── model.py                       # XGBoost/LightGBM regression
│   ├── outcome_predictor.py           # Classification with calibration (NEW)
│   ├── rolling_features.py            # Time-aware rolling features (NEW)
│   ├── explainer.py                   # SHAP interpretability
│   ├── ranker.py                      # Player ranking engine
│   └── pipeline.py                    # Training pipelines
├── models/                            # Saved models
│   ├── course_fit_model.pkl
│   ├── outcome_made_cut.pkl
│   ├── outcome_top_10.pkl
│   └── outcome_win.pkl
├── predict_tournament.py              # Course fit predictions
├── predict_outcomes.py                # Outcome predictions (NEW)
├── requirements.txt
└── README.md
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download Kaggle data (2015-2022 PGA Tour stats)
# Place in: data/raw/kaggle/ASA All PGA Raw Data - Tourn Level.csv
```

## Usage

### 1. Train Outcome Prediction Models

```bash
# Train on 2015-2022 data with XGBoost
py -3.11 predict_outcomes.py --train

# Train with logistic regression baseline
py -3.11 predict_outcomes.py --train --model logistic

# Custom year range
py -3.11 predict_outcomes.py --train --min-year 2018 --max-year 2022
```

### 2. Predict Player Outcomes

```bash
# Predict Scottie Scheffler at Augusta National
py -3.11 predict_outcomes.py --predict --player "Scheffler" --course "Augusta"

# Predict Rory McIlroy at TPC Sawgrass
py -3.11 predict_outcomes.py --predict --player "Rory" --course "Sawgrass"

# Any partial name match works
py -3.11 predict_outcomes.py --predict --player "DJ" --course "Torrey"
```

### 3. Course Fit Rankings

```bash
# Get course fit rankings for a tournament
py -3.11 predict_tournament.py --course "TPC Sawgrass"

# Update with latest ESPN data
py -3.11 predict_tournament.py --course "Augusta" --update
```

### 4. Python API

```python
from src.pipeline import run_outcome_prediction_pipeline, run_course_fit_pipeline

# Train outcome models
results = run_outcome_prediction_pipeline(
    min_year=2018,
    max_year=2022,
    model_type='xgboost',
    calibration_method='isotonic'
)

# Train course fit model
fit_results = run_course_fit_pipeline(
    model_type='xgboost',
    use_shap=True
)
```

## Model Performance

### Outcome Prediction (Test Set)

| Outcome | Brier Score | ROC-AUC |
|---------|-------------|---------|
| Made Cut | 0.232 | 0.62 |
| Top-10 | 0.081 | 0.65 |
| Win | 0.008 | 0.61 |

**Top Predictive Features**:
1. `sg_total_last_10` - Rolling 10-tournament strokes gained
2. `course_avg_sg` - Player's history at the course
3. `sg_total_last_5` - Recent form
4. `sg_total_momentum` - Improving vs declining trend

### Course Fit Model

- **Test RMSE**: ~0.85 strokes
- **R² Score**: 0.72+

## Key Components

### Rolling Features (`rolling_features.py`)

```python
class RollingFormCalculator:
    """Time-aware rolling features with leakage prevention."""

    def compute_rolling_features(df, windows=[5, 10]):
        # Uses shift(1).rolling(window).mean()
        # to exclude current tournament
```

### Outcome Predictor (`outcome_predictor.py`)

```python
class OutcomePredictor:
    """Calibrated classification for tournament outcomes."""

    def __init__(self, outcome_type='made_cut',
                 model_type='xgboost',
                 calibration_method='isotonic'):
        # Uses CalibratedClassifierCV for probability calibration
```

### Time-Aware Splitting

```python
class TimeSeriesSplit:
    """Temporal train/test splits - train on past, test on future."""

    @staticmethod
    def temporal_split(df, test_size=0.2):
        # Never leaks future data into training
```

## Interview Talking Points

This project demonstrates several key ML concepts:

1. **Label Leakage Prevention**: "I use `shift(1)` before rolling calculations to ensure the current tournament is never included in its own features."

2. **Probability Calibration**: "Raw classifier probabilities are often poorly calibrated. I use `CalibratedClassifierCV` with isotonic regression to ensure predicted probabilities match actual outcome rates."

3. **Class Imbalance**: "Wins are ~0.7% of outcomes. I use `scale_pos_weight` and evaluate with average precision rather than accuracy."

4. **Temporal Structure**: "Golf tournaments have strong temporal patterns. I always split by date, never randomly, to prevent the model from learning 'future' form."

## Data Sources

- **Training Data**: Kaggle PGA Tour 2015-2022 (strokes gained, tournament results)
- **Current Stats**: ESPN Golf scraper for real-time player statistics
- **Course Data**: Curated course characteristics (yardage, fairway width, etc.)

## Dependencies

- pandas, numpy - Data manipulation
- scikit-learn - ML preprocessing, calibration, metrics
- xgboost - Gradient boosting
- lightgbm - Alternative boosting model
- shap - Model explainability
- matplotlib, seaborn - Visualization
- requests, beautifulsoup4 - Web scraping

## Future Enhancements

- [ ] Real-time prediction API
- [ ] Weather feature integration
- [ ] Betting odds comparison
- [ ] Historical prediction accuracy tracking
- [ ] Web dashboard for predictions

## License

MIT License

---

**Author**: Brandon Moeri
**Version**: 2.0
**Updated**: January 2026
