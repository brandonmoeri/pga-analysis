# %% [markdown]
# # Course fit analysis
#
# **Question:** once you know a player's recent form, do course characteristics
# help predict how they will play at a given course?
#
# Run it cell by cell in VS Code (the `# %%` markers) or as a plain script from
# anywhere in the repo. It is self-contained: it needs the Kaggle CSV in
# `data/raw/kaggle/` (see the README), the curated course CSV, and the packages
# in `requirements.txt`. It does not read or write anything in `models/`.
#
# - **Target:** `sg_total`, strokes gained per round relative to the field.
# - **Validation:** train on 2015-2021, test on the 2022 season.
# - **Leakage:** every form feature uses `shift(1)`, so an event never sees itself.

# %%
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

start = Path(__file__).parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in [start, *start.parents] if (p / "data" / "raw").is_dir())

TEST_SEASON = 2022
MIN_PRIOR_EVENTS = 5
MIN_FIELD_EVENTS = 20
SHOWCASE_COURSES = ["Augusta National", "Pebble Beach GL", "TPC Sawgrass"]
SEED = 42

SG_COLS = ["sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt"]

# Kaggle course name -> course_id in course_characteristics.csv. The two files
# spell names differently, so an exact join silently matches nothing.
COURSE_ALIASES = {
    "Augusta National Golf Club": "Augusta National",
    "Bay Hill": "Bay Hill Club",
    "Colonial": "Colonial CC",
    "Congaree Golf Club": "Congaree GC",
    "East Lake Golf Club": "East Lake GC",
    "Harbour Town Golf Links": "Harbour Town GL",
    "Muirfield Village Golf Club": "Muirfield Village",
    "Olympia Fields Country Club": "Olympia Fields (North)",
    "PGA National": "PGA National (Champion)",
    "Pebble Beach Resort": "Pebble Beach GL",
    "Quail Hollow Club": "Quail Hollow",
    "Riviera Country Club": "Riviera CC",
    "Sedgefield Country Club": "Sedgefield CC",
    "Sherwood CC": "Sherwood CC",
    "Silverado Resort": "Silverado Resort (North)",
    "Southern Hills Country Club": "Southern Hills CC",
    "TPC Craig Ranch": "TPC Craig Ranch",
    "TPC River Highlands": "TPC River Highlands",
    "TPC San Antonio": "TPC San Antonio (Oaks)",
    "TPC Sawgrass": "TPC Sawgrass",
    "TPC Scottsdale": "TPC Scottsdale",
    "TPC Southwind": "TPC Southwind",
    "TPC Twin Cities": "TPC Twin Cities",
    "Torrey Pines South": "Torrey Pines (South)",
    "Waialae CC": "Waialae CC",
}

# %% [markdown]
# ## 1. Load data
#
# Rows without strokes gained are dropped. Form is computed on every remaining
# event; only the modelling table is limited to courses with curated data.

# %%
raw = pd.read_csv(
    ROOT / "data/raw/kaggle/ASA All PGA Raw Data - Tourn Level.csv",
    usecols=["player id", "Player_initial_last", "player", "tournament id", "course", "date", "season", *SG_COLS],
    parse_dates=["date"],
).rename(columns={"player id": "player_key", "tournament id": "event"})

raw["player"] = raw["Player_initial_last"].fillna(raw["player"]).fillna("Player " + raw["player_key"].astype(str))
events = raw.dropna(subset=SG_COLS).sort_values(["player_key", "date"]).reset_index(drop=True)
events["course_id"] = events["course"].str.split(" - ").str[0].map(COURSE_ALIASES)

courses = pd.read_csv(ROOT / "data/raw/courses/course_characteristics.csv").drop(columns="course_name")
COURSE_FEATURES = [
    "yardage", "par", "fairway_width_avg", "green_size_avg", "rough_severity",
    "hazard_count", "elevation_change", "slope_rating", "course_rating",
]

print(f"{len(events):,} player-events, {events['player_key'].nunique()} players, "
      f"seasons {events['season'].min()}-{events['season'].max()}")
print(f"{events['course_id'].notna().mean():.0%} of events map to one of "
      f"{events['course_id'].nunique()} curated courses")

# %% [markdown]
# ## 2. Engineer features

# %%
def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling form from events strictly before each row. Expects player/date order."""
    df = df.copy()
    grouped = df.groupby("player_key")

    for col in SG_COLS:
        df[f"{col}_last_10"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(10, min_periods=MIN_PRIOR_EVENTS).mean()
        )
    df["sg_total_last_5"] = grouped["sg_total"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=3).mean()
    )
    df["sg_total_momentum"] = df["sg_total_last_5"] - df["sg_total_last_10"]
    df["prior_events"] = grouped.cumcount()
    return df


def add_course_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join curated course data and add player-skill x course interactions."""
    df = df.merge(courses, on="course_id", how="inner")
    z = {c: (df[c] - courses[c].mean()) / courses[c].std() for c in COURSE_FEATURES}
    df["ott_x_length"] = df["sg_ott_last_10"] * z["yardage"]
    df["ott_x_tight_fairways"] = df["sg_ott_last_10"] * -z["fairway_width_avg"]
    df["app_x_small_greens"] = df["sg_app_last_10"] * -z["green_size_avg"]
    df["arg_x_rough"] = df["sg_arg_last_10"] * z["rough_severity"]
    df["putt_x_small_greens"] = df["sg_putt_last_10"] * -z["green_size_avg"]
    return df


events = add_form_features(events)
events["course_avg_sg"] = events.groupby(["player_key", "course"])["sg_total"].transform(
    lambda s: s.shift(1).expanding().mean()
)
events["course_appearances"] = events.groupby(["player_key", "course"]).cumcount()

FORM_FEATURES = [f"{c}_last_10" for c in SG_COLS] + ["sg_total_last_5", "sg_total_momentum", "prior_events"]
HISTORY_FEATURES = ["course_avg_sg", "course_appearances"]
INTERACTIONS = ["ott_x_length", "ott_x_tight_fairways", "app_x_small_greens", "arg_x_rough", "putt_x_small_greens"]
ALL_FEATURES = FORM_FEATURES + HISTORY_FEATURES + COURSE_FEATURES + INTERACTIONS

model_df = add_course_features(events[events["prior_events"] >= MIN_PRIOR_EVENTS].dropna(subset=["course_id"]))
train = model_df[model_df["season"] < TEST_SEASON]
test = model_df[model_df["season"] == TEST_SEASON]

print(f"Train: {len(train):,} rows ({train['season'].min()}-{train['season'].max()})")
print(f"Test:  {len(test):,} rows ({TEST_SEASON})")

# %% [markdown]
# ## 3. Does course information add signal?
#
# Every model is scored on the same 2022 rows. Compare each one to the baseline
# of predicting the training mean. The R² is low by nature: a single event is
# mostly noise, even for the best player in the world.

# %%
XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=SEED,
)
LGB_PARAMS = dict(
    n_estimators=300, max_depth=3, num_leaves=8, learning_rate=0.03,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1,
)


def score(name: str, predictions: np.ndarray) -> dict:
    y = test["sg_total"]
    rank_corr = np.nan
    if np.ptp(predictions) > 0:
        scored = test.assign(pred=predictions)
        rank_corr = np.mean([g["pred"].corr(g["sg_total"], method="spearman") for _, g in scored.groupby("event")])
    return {
        "model": name,
        "RMSE": np.sqrt(mean_squared_error(y, predictions)),
        "MAE": mean_absolute_error(y, predictions),
        "R²": r2_score(y, predictions),
        "event rank corr": rank_corr,
    }


def fit(features: list[str], model) -> np.ndarray:
    model.fit(train[features], train["sg_total"])
    return model.predict(test[features])


xgb_full = xgb.XGBRegressor(**XGB_PARAMS)
lgb_full = lgb.LGBMRegressor(**LGB_PARAMS)
baseline = np.full(len(test), train["sg_total"].mean())

results = pd.DataFrame([
    score("Baseline (training mean)", baseline),
    score("XGBoost: form", fit(FORM_FEATURES, xgb.XGBRegressor(**XGB_PARAMS))),
    score("XGBoost: form + course history", fit(FORM_FEATURES + HISTORY_FEATURES, xgb.XGBRegressor(**XGB_PARAMS))),
    score("XGBoost: all features", fit(ALL_FEATURES, xgb_full)),
    score("LightGBM: all features", fit(ALL_FEATURES, lgb_full)),
]).set_index("model")

print(results.round(4).to_string())

# %% [markdown]
# ## 4. Which features matter?
#
# Mean absolute SHAP value on the test season, in strokes gained.

# %%
sample = test[ALL_FEATURES].sample(min(1000, len(test)), random_state=SEED)
shap_values = shap.TreeExplainer(xgb_full).shap_values(sample)

importance = (
    pd.Series(np.abs(shap_values).mean(axis=0), index=ALL_FEATURES, name="mean |SHAP|")
    .sort_values(ascending=False)
)
print(importance.head(15).round(4).to_string())

group_share = pd.Series({
    "form": importance[FORM_FEATURES].sum(),
    "course history": importance[HISTORY_FEATURES].sum(),
    "course characteristics": importance[COURSE_FEATURES].sum(),
    "skill x course interactions": importance[INTERACTIONS].sum(),
}) / importance.sum()
print("\nShare of total SHAP attribution:")
print(group_share.round(3).to_string())

# %% [markdown]
# ## 5. Score the field at every course
#
# A placeholder row after each player's last event gives their form going into
# the next one. **Fit** is a player's prediction at a course minus their own
# average prediction across the modelled courses, so it isolates the course
# effect from how good the player is overall.

# %%
xgb_final = xgb.XGBRegressor(**XGB_PARAMS).fit(model_df[ALL_FEATURES], model_df["sg_total"])

recent = events[events["season"] == TEST_SEASON].groupby("player_key")["player"].last()
next_event = (
    events[events["player_key"].isin(recent.index)]
    .groupby("player_key", as_index=False)
    .last()
    .assign(**{col: np.nan for col in SG_COLS})
)
history = pd.concat([events[["player_key", "date", *SG_COLS]], next_event[["player_key", "date", *SG_COLS]]])
form_now = add_form_features(history.sort_values("player_key", kind="stable")).groupby("player_key").last()
form_now = form_now[form_now.index.isin(recent.index) & (form_now["prior_events"] >= MIN_FIELD_EVENTS)].reset_index()

course_history = (
    events.dropna(subset=["course_id"])
    .groupby(["player_key", "course_id"], as_index=False)
    .agg(course_avg_sg=("sg_total", "mean"), course_appearances=("sg_total", "size"))
)

observed = pd.DataFrame({"course_id": model_df["course_id"].unique()})
field = form_now[["player_key", *FORM_FEATURES]].merge(observed, how="cross")
field = add_course_features(field.merge(course_history, on=["player_key", "course_id"], how="left"))
field["course_appearances"] = field["course_appearances"].fillna(0)
field["player"] = field["player_key"].map(recent)
field["predicted_sg"] = xgb_final.predict(field[ALL_FEATURES])
field["fit"] = field["predicted_sg"] - field.groupby("player_key")["predicted_sg"].transform("mean")

print(f"Scored {field['player_key'].nunique()} players at {field['course_id'].nunique()} courses")

# %%
for course_id in SHOWCASE_COURSES:
    best = field[field["course_id"] == course_id].nlargest(5, "predicted_sg")
    print(f"\n{course_id}: top 5 by predicted strokes gained")
    print(best[["player", "predicted_sg", "fit", "course_appearances"]].round(3).to_string(index=False))

# %% [markdown]
# ## 6. Player and course views

# %%
top_player = field.groupby("player")["predicted_sg"].mean().idxmax()
player_view = field[field["player"] == top_player].sort_values("fit", ascending=False)
print(f"{top_player}: best and worst course fits")
print(pd.concat([player_view.head(3), player_view.tail(3)])[["course_id", "predicted_sg", "fit"]].round(3).to_string(index=False))

course_view = (
    model_df.groupby("course_id")
    .agg(events=("event", "nunique"), rows=("sg_total", "size"), spread_of_results=("sg_total", "std"))
    .join(field.groupby("course_id").agg(mean_fit=("fit", "mean"), spread_of_fit=("fit", "std")))
    .sort_values("spread_of_fit", ascending=False)
)
print("\nCourses that separate players most by fit:")
print(course_view.head(10).round(3).to_string())
