"""
Model training and evaluation for Course Fit Prediction.
Uses XGBoost and LightGBM with interaction features.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path


class CourseFitModel:
    """Train and evaluate course fit models."""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize model.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.identifier_cols = ['player_id', 'course_id']
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
        **model_params
    ) -> Dict[str, float]:
        """
        Train the course fit model.
        
        Args:
            X: Feature matrix
            y: Target variable (course score)
            test_size: Proportion for test set
            random_state: Random seed
            **model_params: Model-specific parameters
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Separate identifiers from features
        self.feature_names = [col for col in X.columns if col not in self.identifier_cols]
        X_features = X[self.feature_names].copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y,
            test_size=test_size,
            random_state=random_state
        )
        
        # Default parameters for each model type
        if self.model_type == 'xgboost':
            default_params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 200,
                'random_state': random_state,
                'verbosity': 0,
            }
            default_params.update(model_params)
            
            self.model = xgb.XGBRegressor(**default_params)
        
        elif self.model_type == 'lightgbm':
            default_params = {
                'objective': 'regression',
                'metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 200,
                'random_state': random_state,
                'verbose': -1,
            }
            default_params.update(model_params)
            
            self.model = lgb.LGBMRegressor(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
        }
        
        print(f"\n{self.model_type.upper()} Model Evaluation")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"{metric:15s}: {value:.4f}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.model_type == 'xgboost':
            importance_dict = self.model.get_booster().get_score(importance_type='weight')
        elif self.model_type == 'lightgbm':
            importance_dict = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        
        # Sort and return top features
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def predict_fit_score(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict course fit scores for player-course combinations.
        Lower scores = better fit.
        
        Args:
            X: Feature matrix with player_id and course_id
        
        Returns:
            DataFrame with predictions and identifiers
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_features = X[self.feature_names].copy()
        predictions = self.model.predict(X_features)
        
        result = X[self.identifier_cols].copy()
        result['predicted_fit_score'] = predictions
        
        return result
    
    def rank_best_fits(
        self,
        X: pd.DataFrame,
        course_id: str,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Rank best-fit players for a specific course.
        
        Args:
            X: Full feature matrix with predictions
            course_id: Course to analyze
            top_n: Number of top players to return
        
        Returns:
            Sorted DataFrame of top player fits
        """
        course_data = X[X['course_id'] == course_id].copy()
        predictions = self.predict_fit_score(course_data)
        
        # Lower score = better fit
        top_fits = predictions.sort_values('predicted_fit_score').head(top_n)
        
        return top_fits
    
    def save_model(self, filepath: str = "models/course_fit_model.pkl"):
        """Save trained model."""
        Path(filepath).parent.mkdir(exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = "models/course_fit_model.pkl"):
        """Load trained model."""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
