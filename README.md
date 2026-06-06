# PGA Course Fit & Tournament Outcome Prediction

A full-stack machine learning application for predicting player-course compatibility and tournament outcomes in professional golf.

## Project Overview

This project combines **two predictive models** served through a REST API and interactive web dashboard:

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

### Option 1: Docker (recommended)

```bash
docker compose up
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

### Option 2: Local development

```bash
# Backend
pip install -r requirements.txt
uvicorn backend.app.main:app --reload

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Option 3: CLI

```bash
# Predict player outcomes at a course
python cli.py predict --player scheffler_scottie --course augusta_national

# Get tournament rankings
python cli.py rank --tournament masters_2024 --courses augusta_national --top 50

# Explain a prediction (SHAP)
python cli.py explain --player rory_mcilroy --course pebble_beach --top 10

# Show global feature importance
python cli.py features --top 15

# Update player statistics from PGA Tour / ESPN
python cli.py update

# Show data info
python cli.py info

# List API endpoints
python cli.py api
```

Example CLI output:

```
======================================================================
PLAYER OUTCOME PREDICTION
======================================================================

Player:  scheffler_scottie
Course:  augusta_national

Course Fit Score: +0.45 strokes

Tournament Outcomes:
  Make Cut:     87.7%
  Top 10:       35.7%
  Win:           6.8%

Confidence:   72.0%
======================================================================
```

## Features

### Web Dashboard

Four pages served by the React frontend:

| Page | Description |
|------|-------------|
| **Dashboard** | API health status, data coverage, top features |
| **Predictions** | Player outcome prediction with course history (sg_avg, appearances) and probability bars |
| **Rankings** | Tournament field rankings across one or more courses |
| **Explanations** | SHAP feature explanations and global feature importance |

### Tournament Outcome Prediction

Predicts three binary outcomes with calibrated probabilities:

| Outcome | Description | Base Rate |
|---------|-------------|-----------|
| **Made Cut** | Did player make the cut? | ~58% |
| **Top-10** | Did player finish top-10? | ~9% |
| **Win** | Did player win? | ~0.8% |

**Rolling Form Features** (with leakage prevention):
- `sg_total_last_5` / `sg_total_last_10` - Recent strokes gained averages
- `sg_*_momentum` - Form trend (improving vs declining)
- `course_avg_sg` - Historical strokes gained at specific course
- `rounds_at_course` - Experience at the course

**Leakage Prevention**: Uses `shift(1)` before rolling calculations to ensure the current tournament is never included in its own features.

### Course Fit Prediction

Predicts player-course compatibility using:
- **Player Skill Profiles**: Driving distance/accuracy, GIR, scrambling, putting
- **Course Characteristics**: Yardage, fairway width, hazard density, slope rating
- **Interaction Features**: Accuracy × tight fairways, distance × long courses

## Project Structure

```
pga-analysis/
├── backend/
│   └── app/
│       ├── main.py                    # FastAPI entry point
│       ├── models.py                  # Pydantic request/response models
│       ├── config.py                  # App settings
│       ├── routers/
│       │   ├── predictions.py         # POST /api/predictions/player-outcome
│       │   ├── rankings.py            # POST /api/rankings/tournament
│       │   ├── explanations.py        # POST /api/explanations/local
│       │   └── data.py                # GET  /api/data/*
│       ├── services/
│       │   ├── course_fit.py          # Course fit scoring
│       │   ├── outcomes.py            # Outcome probability prediction
│       │   ├── ranking.py             # Tournament ranking engine
│       │   ├── features.py            # Feature engineering
│       │   ├── explanations.py        # SHAP explanations
│       │   ├── data.py                # Data loading
│       │   └── scraper.py             # PGA Tour / ESPN stats scraper
│       ├── websockets/
│       │   └── ranking_stream.py      # WebSocket streaming for live rankings
│       └── utils/
│           └── model_loader.py        # Model loading and caching
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── Dashboard.tsx
│       │   ├── Predictions.tsx        # Includes course history display
│       │   ├── Rankings.tsx
│       │   └── Explanations.tsx
│       ├── components/
│       │   ├── PredictionForm.tsx
│       │   ├── ProbabilityBar.tsx
│       │   ├── RankingTable.tsx
│       │   └── FeatureImportance.tsx
│       ├── hooks/useApi.ts            # Generic API request hook
│       └── services/api.ts            # Typed API client
├── data/
│   ├── raw/
│   │   ├── kaggle/                    # PGA Tour 2015-2022 data
│   │   └── courses/                   # Course characteristics
│   └── processed/                     # Generated data files
├── models/                            # Saved model files (.pkl)
├── tests/
│   └── test_endpoints.py
├── cli.py                             # Unified CLI
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── nginx.conf
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | API health and model status |
| POST | `/api/predictions/player-outcome` | Predict outcomes for a player at a course |
| POST | `/api/rankings/tournament` | Rank players for a tournament |
| GET | `/api/rankings/player/{id}/course-fits` | Player fit across courses |
| POST | `/api/explanations/local` | SHAP explanation for a prediction |
| GET | `/api/explanations/feature-importance` | Global feature importance |
| GET | `/api/data/courses` | List available courses |
| GET | `/api/data/stats/player/{id}` | Player statistics |
| POST | `/api/data/update-stats` | Refresh stats from PGA Tour / ESPN |
| WS | `/ws/rankings/stream` | Live ranking updates via WebSocket |

Interactive docs available at http://localhost:8000/docs when the server is running.

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

## Interview Talking Points

1. **Label Leakage Prevention**: "I use `shift(1)` before rolling calculations to ensure the current tournament is never included in its own features."

2. **Probability Calibration**: "Raw classifier probabilities are often poorly calibrated. I use `CalibratedClassifierCV` with isotonic regression to ensure predicted probabilities match actual outcome rates."

3. **Class Imbalance**: "Wins are ~0.7% of outcomes. I use `scale_pos_weight` and evaluate with average precision rather than accuracy."

4. **Temporal Structure**: "Golf tournaments have strong temporal patterns. I always split by date, never randomly, to prevent the model from learning 'future' form."

## Data Sources

- **Training Data**: Kaggle PGA Tour 2015-2022 (strokes gained, tournament results)
- **Current Stats**: PGA Tour API with ESPN fallback for current season statistics
- **Course Data**: Curated course characteristics (yardage, fairway width, etc.)

## Dependencies

**Backend**
- `fastapi`, `uvicorn` - REST API and ASGI server
- `websockets` - Live ranking streaming
- `pydantic` - Request/response validation
- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - ML preprocessing, calibration, metrics
- `xgboost`, `lightgbm` - Gradient boosting
- `shap` - Model explainability
- `requests` - Web scraping

**Frontend**
- React + TypeScript + Vite
- Tailwind CSS
- Axios

## Future Enhancements

- [ ] Weather feature integration
- [ ] Betting odds comparison
- [ ] Historical prediction accuracy tracking
- [ ] Deploy to Vercel (frontend) + Railway/Render (backend)

## License

MIT License

---

**Author**: Brandon Moeri
**Version**: 3.0
**Updated**: June 2026
