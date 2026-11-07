# PGA Course Fit Analysis

Does a golfer's game suit a particular course, once you account for how well they are playing right now? This project tests that question on PGA Tour strokes-gained data from 2015 to 2022, and includes a FastAPI + React app built to serve the results.

**The short answer:** recent form does most of the work. Course information may add a little, but one season of holdout data can't confirm it.

> **Project status:** the analysis runs end to end on real data. The app is an interface prototype: its API, dashboard and CLI are not connected to a trained model yet. See [App status](#app-status).

## Findings

The analysis lives in [notebooks/course_fit_analysis.py](notebooks/course_fit_analysis.py). It predicts `sg_total`, a player's strokes gained per round relative to the field, for 25 courses with curated data. Models train on 2015-2021 (13,196 player-events) and are scored on the 2022 season (1,798 player-events across 15 events).

| Model | RMSE | MAE | R² | Within-event rank corr. |
|-------|------|-----|----|-------------------------|
| Baseline (training mean) | 1.860 | 1.450 | 0.000 | n/a |
| XGBoost: form only | 1.773 | 1.384 | 0.091 | 0.314 |
| XGBoost: form + course history | 1.769 | 1.381 | 0.094 | 0.324 |
| XGBoost: all features | 1.758 | 1.377 | 0.106 | 0.331 |
| LightGBM: all features | 1.760 | 1.379 | 0.104 | 0.334 |

- **Form is the main signal.** It improves RMSE by 0.087 over the baseline and accounts for 61% of the SHAP attribution. `sg_total_last_10` is the strongest single feature.
- **The course effect is small and unproven.** Adding course features improves RMSE by another 0.015, but the 95% event-level bootstrap interval is -0.002 to 0.031, which includes zero.
- **Low R² is expected.** A single tournament is mostly noise, so the within-event rank correlation is the more useful measure: about 0.33 between predicted and actual results.

The course results are also limited by the data: 25 courses, 15 test events, and hand-curated course attributes.

## Method

- **No label leakage:** rolling features apply `shift(1)` before the window, so an event never contributes to its own features.
- **Time-based validation:** the test set is a full later season, not a random sample of rows.
- **Features:** 10-event rolling strokes gained by category, 5-event form and momentum, prior results at the course, course attributes (length, fairway width, green size, rough, hazards, elevation, slope), and skill × course interactions.
- **Course fit metric:** a player's prediction at a course minus their own average across all courses, which separates fit from overall ability.
- **Interpretation:** SHAP values for global importance, grouped by feature family.

## Running the analysis

Requires Python 3.11+.

1. Download the [PGA Tour Golf Data 2015-2022](https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022) dataset from Kaggle.
2. Save `ASA All PGA Raw Data - Tourn Level.csv` in `data/raw/kaggle/`.
3. Install dependencies and run the notebook:

```bash
pip install -r requirements.txt
python notebooks/course_fit_analysis.py
```

In VS Code you can also run it cell by cell using the `# %%` markers.

## Running the app

The API reads `data/processed/*.csv` and `models/*.pkl`. Neither is committed, and the pipeline that generated them is no longer in the repo (it was last present in commit `2678e88~1`), so a fresh clone will have empty data endpoints.

```bash
# Backend: http://localhost:8000 (interactive docs at /docs)
pip install -r requirements.txt
uvicorn backend.app.main:app --reload

# Frontend, in a second terminal
cd frontend
npm install
npm run dev
```

`docker compose up` is also configured, serving the frontend on port 3000 and the API on port 8000. The compose file does not mount `models/`.

IDs use the dataset's formats: players look like `A. Ancer`, and courses look like `Muirfield Village Golf Club - Dublin, OH`.

### Tests

```bash
pip install pytest
pytest
```

The 11 endpoint tests need the processed CSVs. They replace the prediction services with mocks, so they cover request handling and response shapes, not model output.

## App status

| Area | Current behavior |
|------|------------------|
| `GET /api/data/courses` | **Works:** served from the processed CSV |
| `GET /api/health`, `/api/health/models` | **Works:** reports model load state |
| `POST /api/predictions/player-outcome` | **Not working yet:** returns 500 because the outcome models don't load. The service behind it returns random placeholder values |
| `POST /api/explanations/local` | **Not working yet:** returns 500 on its placeholder SHAP values |
| `POST /api/rankings/tournament` | **Placeholder:** returns an empty ranking |
| `GET /api/explanations/feature-importance` | **Placeholder:** fixed values |
| `GET /api/rankings/player/{id}/course-fits` | **Placeholder:** fixed values |
| `GET /api/rankings/course/{id}/difficulty` | **Placeholder:** fixed values |
| `GET /api/data/stats/player/{id}` | **Placeholder:** fixed values |
| `POST /api/data/update-stats` | **Placeholder:** reports success without fetching data |
| `GET /api/predictions/historical/{id}` | **Not implemented:** returns an empty list |
| WebSocket ranking stream | **Untested:** built on the placeholder services, at `/ws/ws/rankings/tournament` |
| CLI (`cli.py`) | `info`, `api` and `update` run, with placeholder data. `predict`, `rank`, `explain` and `features` show no results because the CLI never loads the models |

The React app has Dashboard, Predictions, Rankings and Explanations pages, and it inherits the behavior above.

## Project structure

```
pga-analysis/
├── notebooks/
│   └── course_fit_analysis.py   # The analysis: features, models, SHAP, rankings
├── backend/app/
│   ├── main.py                  # FastAPI entry point
│   ├── routers/                 # API routes
│   ├── services/                # Prediction, ranking and data services
│   ├── websockets/              # Ranking stream
│   └── utils/model_loader.py    # Model cache
├── frontend/src/                # React + TypeScript dashboard
├── data/raw/courses/            # Curated course characteristics
├── tests/test_endpoints.py      # API tests
├── cli.py                       # Command-line interface
└── docker-compose.yml           # Backend and frontend containers
```

## Data sources

- **Results and strokes gained:** [PGA Tour Golf Data 2015-2022](https://www.kaggle.com/datasets/robikscube/pga-tour-golf-data-20152022) on Kaggle.
- **Course characteristics:** a curated file of 30 courses, committed at `data/raw/courses/course_characteristics.csv`. The notebook uses the 25 that match tournament data.

## Tech stack

- **Analysis:** pandas, scikit-learn, XGBoost, LightGBM, SHAP
- **Backend:** FastAPI, Pydantic, WebSockets
- **Frontend:** React, TypeScript, Vite, Tailwind CSS

## Roadmap

- [ ] Serve the notebook's model through the API in place of the placeholder services
- [ ] Add a repeatable pipeline to build the data files and models
- [ ] Extend course coverage and test the course effect on more seasons
- [ ] Add tests that exercise real model output
- [ ] Add weather and betting odds comparisons
- [ ] Finish deployment (`vercel.json` is in place, but `pyproject.toml` lacks the ML dependencies)

## License

MIT License

**Author:** Brandon Moeri
