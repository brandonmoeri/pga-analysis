"""
Tournament Outcome Predictor with Calibrated Probabilities.

Predicts tournament outcomes probabilistically:
- make_cut: Did player make the cut? (~70% base rate)
- top_10: Did player finish top-10? (~7% base rate)
- win: Did player win? (~0.7% base rate)

Features:
- Multiple model types: logistic regression, XGBoost
- Probability calibration via Platt scaling or isotonic regression
- Class imbalance handling for rare outcomes (wins)
- Evaluation with Brier score, ROC-AUC, calibration curves
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import joblib
import warnings

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)
import xgboost as xgb

warnings.filterwarnings('ignore', category=UserWarning)


class OutcomePredictor:
    """
    Predicts tournament outcomes with calibrated probabilities.

    Outcomes:
    - make_cut (binary): Most common, ~70% base rate
    - top_10 (binary): Rarer, ~7% base rate
    - win (binary): Very rare, ~0.7% base rate

    Uses CalibratedClassifierCV for probability calibration.
    """

    # Base rates for class weight calculation
    BASE_RATES = {
        'make_cut': 0.70,
        'top_10': 0.07,
        'win': 0.007
    }

    def __init__(
        self,
        outcome_type: str = 'make_cut',
        model_type: str = 'xgboost',
        calibration_method: str = 'isotonic'
    ):
        """
        Initialize predictor.

        Args:
            outcome_type: 'make_cut', 'top_10', or 'win'
            model_type: 'logistic' or 'xgboost'
            calibration_method: 'platt' (sigmoid) or 'isotonic'
        """
        self.outcome_type = outcome_type
        self.model_type = model_type
        self.calibration_method = calibration_method

        self.model = None
        self.calibrated_model = None
        self.feature_names = None
        self.is_trained = False

    def _get_class_weight(self) -> float:
        """Calculate scale_pos_weight for class imbalance."""
        base_rate = self.BASE_RATES.get(self.outcome_type, 0.5)
        # scale_pos_weight = (negative samples) / (positive samples)
        return (1 - base_rate) / base_rate

    def _create_base_model(self):
        """Create uncalibrated base model."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='lbfgs',
                random_state=42
            )
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                scale_pos_weight=self._get_class_weight(),
                max_depth=6,
                learning_rate=0.1,
                n_estimators=200,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                verbosity=0
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Train model with calibration.

        If X_val/y_val provided, uses them for calibration.
        Otherwise uses cross-validation on training set.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Optional validation features for calibration
            y_val: Optional validation labels for calibration
            cv_folds: Number of CV folds if no validation set

        Returns:
            Dictionary of training metrics
        """
        # Store feature names
        self.feature_names = list(X_train.columns)

        # Handle NaN values
        X_train = X_train.fillna(0)
        if X_val is not None:
            X_val = X_val.fillna(0)

        # Create base model
        self.model = self._create_base_model()

        # Calibration method mapping
        cal_method = 'sigmoid' if self.calibration_method == 'platt' else 'isotonic'

        # Always fit the base model first (needed for feature importance)
        self.model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            # Create calibrated wrapper with prefit model
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method=cal_method,
                cv='prefit'
            )
            self.calibrated_model.fit(X_val, y_val)
        else:
            # CV-based calibration on training set
            # Use a clone of the fitted model for calibration
            self.calibrated_model = CalibratedClassifierCV(
                self._create_base_model(),
                method=cal_method,
                cv=cv_folds
            )
            self.calibrated_model.fit(X_train, y_train)

        self.is_trained = True

        # Return training metrics
        return self.evaluate(X_train, y_train)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return calibrated probabilities.

        Args:
            X: Features to predict

        Returns:
            Array of probabilities for positive class
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        X = X.fillna(0)
        return self.calibrated_model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Return binary predictions.

        Args:
            X: Features to predict
            threshold: Probability threshold for positive class

        Returns:
            Array of binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute classification metrics.

        Args:
            X: Features
            y: True labels
            threshold: Threshold for binary predictions

        Returns:
            Dictionary of metrics
        """
        X = X.fillna(0)
        probs = self.predict_proba(X)
        preds = (probs >= threshold).astype(int)

        metrics = {
            'brier_score': brier_score_loss(y, probs),
            'log_loss': log_loss(y, probs),
        }

        # ROC-AUC requires both classes present
        if len(np.unique(y)) > 1:
            metrics['roc_auc'] = roc_auc_score(y, probs)
            metrics['avg_precision'] = average_precision_score(y, probs)

        # Classification metrics at threshold
        metrics['precision'] = precision_score(y, preds, zero_division=0)
        metrics['recall'] = recall_score(y, preds, zero_division=0)
        metrics['f1'] = f1_score(y, preds, zero_division=0)

        # Base rate for reference
        metrics['base_rate'] = y.mean()
        metrics['predicted_rate'] = probs.mean()

        return metrics

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance from base model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")

        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        elif self.model_type == 'logistic':
            importance = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'calibrated_model': self.calibrated_model,
            'base_model': self.model,
            'feature_names': self.feature_names,
            'outcome_type': self.outcome_type,
            'model_type': self.model_type,
            'calibration_method': self.calibration_method
        }
        joblib.dump(model_data, filepath)

    @classmethod
    def load_model(cls, filepath: str) -> 'OutcomePredictor':
        """Load trained model from file."""
        model_data = joblib.load(filepath)

        predictor = cls(
            outcome_type=model_data['outcome_type'],
            model_type=model_data['model_type'],
            calibration_method=model_data['calibration_method']
        )
        predictor.calibrated_model = model_data['calibrated_model']
        predictor.model = model_data['base_model']
        predictor.feature_names = model_data['feature_names']
        predictor.is_trained = True

        return predictor


class OutcomeEvaluator:
    """Evaluation metrics and visualizations for outcome predictions."""

    @staticmethod
    def compute_calibration_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute calibration curve data.

        Returns:
            fraction_of_positives: Actual positive rate per bin
            mean_predicted_value: Mean predicted probability per bin
        """
        return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    @staticmethod
    def classification_report(
        y_true: pd.Series,
        y_prob: np.ndarray,
        outcome_type: str = 'unknown'
    ) -> Dict[str, Any]:
        """
        Comprehensive classification report.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            outcome_type: Name of outcome for display

        Returns:
            Dictionary with metrics and calibration data
        """
        metrics = {
            'outcome': outcome_type,
            'n_samples': len(y_true),
            'n_positive': int(y_true.sum()),
            'base_rate': float(y_true.mean()),
            'brier_score': brier_score_loss(y_true, y_prob),
        }

        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['avg_precision'] = average_precision_score(y_true, y_prob)

            # Calibration curve
            try:
                frac_pos, mean_pred = calibration_curve(
                    y_true, y_prob, n_bins=10, strategy='uniform'
                )
                metrics['calibration_curve'] = {
                    'fraction_of_positives': frac_pos.tolist(),
                    'mean_predicted_value': mean_pred.tolist()
                }
            except Exception:
                pass

        return metrics

    @staticmethod
    def print_evaluation_summary(results: Dict[str, Dict]):
        """
        Print formatted evaluation summary for all outcomes.

        Args:
            results: Dictionary mapping outcome names to metric dictionaries
        """
        print("\n" + "=" * 70)
        print("TOURNAMENT OUTCOME PREDICTION - EVALUATION SUMMARY")
        print("=" * 70)

        for outcome, metrics in results.items():
            print(f"\n{outcome.upper()}")
            print("-" * 40)
            print(f"  Samples: {metrics.get('n_samples', 'N/A')}")
            print(f"  Positive: {metrics.get('n_positive', 'N/A')} "
                  f"({metrics.get('base_rate', 0)*100:.1f}% base rate)")
            print(f"  Brier Score: {metrics.get('brier_score', 'N/A'):.4f} (lower is better)")

            if 'roc_auc' in metrics:
                print(f"  ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
            if 'avg_precision' in metrics:
                print(f"  Avg Precision: {metrics.get('avg_precision', 'N/A'):.4f}")

        print("\n" + "=" * 70)

    @staticmethod
    def plot_calibration_curves(
        results: Dict[str, Dict],
        save_path: str = None
    ):
        """
        Plot calibration curves for all outcomes.

        Args:
            results: Dictionary with calibration curve data
            save_path: Optional path to save figure
        """
        try:
            import matplotlib.pyplot as plt

            n_outcomes = len(results)
            fig, axes = plt.subplots(1, n_outcomes, figsize=(5 * n_outcomes, 4))

            if n_outcomes == 1:
                axes = [axes]

            for ax, (outcome, metrics) in zip(axes, results.items()):
                if 'calibration_curve' not in metrics:
                    ax.text(0.5, 0.5, 'No calibration data',
                            ha='center', va='center')
                    ax.set_title(outcome)
                    continue

                frac_pos = metrics['calibration_curve']['fraction_of_positives']
                mean_pred = metrics['calibration_curve']['mean_predicted_value']

                ax.plot(mean_pred, frac_pos, 's-', label='Model')
                ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                ax.set_xlabel('Mean predicted probability')
                ax.set_ylabel('Fraction of positives')
                ax.set_title(f'{outcome}\nBrier: {metrics.get("brier_score", 0):.4f}')
                ax.legend(loc='lower right')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"  Calibration plot saved to: {save_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            print("  Warning: matplotlib not available for plotting")
