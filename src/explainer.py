"""
SHAP explainability analysis for Course Fit Model.
Provides interpretable insights into model predictions.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ShapExplainer:
    """SHAP-based model interpretability."""
    
    def __init__(self, model, X_train: pd.DataFrame):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained XGBoost or LightGBM model
            X_train: Training data for background
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize appropriate SHAP explainer based on model type."""
        model_class = type(self.model).__name__
        
        if 'XGB' in model_class:
            self.explainer = shap.TreeExplainer(self.model)
        elif 'LGBM' in model_class:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Fallback to KernelExplainer
            self.explainer = shap.KernelExplainer(
                self.model.predict,
                shap.sample(self.X_train, min(100, len(self.X_train)))
            )
    
    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for predictions.
        
        Args:
            X: Feature matrix
        
        Returns:
            SHAP values array
        """
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
    
    def global_feature_importance(self, X: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.
        
        Args:
            X: Feature matrix
            top_n: Number of features to show
        
        Returns:
            DataFrame of top important features
        """
        shap_values = self.calculate_shap_values(X)
        
        # Use mean absolute SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': np.abs(shap_values).mean(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        return feature_importance.head(top_n)
    
    def local_explanation(
        self,
        X: pd.DataFrame,
        instance_idx: int,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get SHAP explanation for a single prediction.
        Shows which features most influenced this specific prediction.
        
        Args:
            X: Feature matrix
            instance_idx: Index of instance to explain
            top_n: Number of top features to show
        
        Returns:
            DataFrame of contributing features
        """
        shap_values = self.calculate_shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        instance_shap = shap_values[instance_idx]
        
        explanation = pd.DataFrame({
            'feature': X.columns,
            'value': X.iloc[instance_idx].values,
            'shap_value': instance_shap,
            'abs_shap_value': np.abs(instance_shap)
        }).sort_values('abs_shap_value', ascending=False)
        
        return explanation.head(top_n)
    
    def player_course_interaction_explanation(
        self,
        X: pd.DataFrame,
        player_course_pair: Tuple[str, str],
        X_full: pd.DataFrame
    ) -> dict:
        """
        Explain a specific player-course fit prediction.
        
        Args:
            X: Feature matrix for prediction
            player_course_pair: Tuple of (player_id, course_id)
            X_full: Full feature matrix with identifiers
        
        Returns:
            Dictionary with explanation details
        """
        player_id, course_id = player_course_pair
        
        # Find the row matching this pair
        mask = (X_full['player_id'] == player_id) & (X_full['course_id'] == course_id)
        if not mask.any():
            return {'error': f"Pair {player_course_pair} not found"}
        
        idx = np.where(mask)[0][0]
        
        # Get prediction
        prediction = self.model.predict(X[[col for col in X.columns if col not in ['player_id', 'course_id']]].iloc[[idx]])[0]
        
        # Get SHAP explanation
        explanation = self.local_explanation(X, idx, top_n=15)
        
        return {
            'player_id': player_id,
            'course_id': course_id,
            'predicted_fit_score': prediction,
            'top_contributing_features': explanation
        }
    
    def force_plot(self, X: pd.DataFrame, instance_idx: int = 0):
        """
        Create SHAP force plot for visualization.
        
        Args:
            X: Feature matrix
            instance_idx: Index to visualize
        """
        shap_values = self.calculate_shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return shap.force_plot(
            self.explainer.expected_value,
            shap_values[instance_idx],
            X.iloc[instance_idx],
            show=False
        )
    
    def summary_plot(self, X: pd.DataFrame, plot_type: str = 'bar'):
        """
        Create SHAP summary plot.
        
        Args:
            X: Feature matrix
            plot_type: 'bar', 'beeswarm', or 'violin'
        """
        shap_values = self.calculate_shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type=plot_type, show=False)
        return plt.gcf()
    
    def get_interaction_explanation(
        self,
        X: pd.DataFrame,
        feature1: str,
        feature2: str
    ) -> np.ndarray:
        """
        Analyze interaction between two features using SHAP.
        
        Args:
            X: Feature matrix
            feature1: First feature name
            feature2: Second feature name
        
        Returns:
            Array of interaction effects
        """
        shap_values = self.calculate_shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        idx1 = list(X.columns).index(feature1)
        idx2 = list(X.columns).index(feature2)
        
        # Simple interaction: product of SHAP values
        interaction = shap_values[:, idx1] * shap_values[:, idx2]
        
        return interaction
