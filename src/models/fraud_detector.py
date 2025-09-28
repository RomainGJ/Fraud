import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.model_selection import cross_val_score
import joblib
import logging
from typing import Tuple, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from mlflow_integration.experiment_tracker import MLflowExperimentTracker
except ImportError:
    MLflowExperimentTracker = None
    logging.warning("MLflow integration not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self, model_type='random_forest', use_mlflow=True):
        self.model_type = model_type
        self.model = None
        self.anomaly_detector = None
        self.is_trained = False
        self.feature_importance = None
        self.use_mlflow = use_mlflow and MLflowExperimentTracker is not None
        self.mlflow_tracker = None

        if self.use_mlflow:
            try:
                self.mlflow_tracker = MLflowExperimentTracker()
                logger.info("MLflow tracking enabled")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {str(e)}")
                self.use_mlflow = False

    def _initialize_model(self):
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Initialize anomaly detector
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Dict[str, Any]:
        self._initialize_model()

        logger.info(f"Training {self.model_type} model...")

        # Train main classifier
        self.model.fit(X_train, y_train)

        # Train anomaly detector on normal transactions
        normal_transactions = X_train[y_train == 0]
        self.anomaly_detector.fit(normal_transactions)

        self.is_trained = True

        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        # Evaluate with cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')

        training_results = {
            'model_type': self.model_type,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0],
            'fraud_rate': y_train.mean()
        }

        # MLflow tracking
        if self.use_mlflow and self.mlflow_tracker and X_test is not None and y_test is not None:
            try:
                # Get evaluation results
                evaluation_results = self.evaluate(X_test, y_test)

                # Log complete training session
                self.mlflow_run_id = self.mlflow_tracker.log_training_session(
                    model=self.model,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    training_params=training_results,
                    evaluation_results=evaluation_results,
                    feature_importance=self.feature_importance
                )

                if self.mlflow_run_id:
                    training_results['mlflow_run_id'] = self.mlflow_run_id

            except Exception as e:
                logger.warning(f"MLflow logging failed: {str(e)}")

        logger.info(f"Training completed. CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        return training_results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        probabilities = self.model.predict_proba(X)
        return probabilities[:, 1]  # Return probability of fraud class

    def detect_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies")

        anomaly_scores = self.anomaly_detector.decision_function(X)
        anomalies = self.anomaly_detector.predict(X)
        return anomalies == -1  # True for anomalies

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        anomalies = self.detect_anomalies(X_test)

        # Metrics
        auc_score = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = np.trapz(precision, recall)

        # Anomaly detection performance
        anomaly_precision = np.mean(y_test[anomalies] == 1) if np.sum(anomalies) > 0 else 0

        evaluation_results = {
            'auc_score': auc_score,
            'pr_auc': pr_auc,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'confusion_matrix': conf_matrix.tolist(),
            'anomaly_detection_precision': anomaly_precision,
            'anomalies_detected': np.sum(anomalies),
            'total_samples': len(y_test)
        }

        logger.info(f"Evaluation completed. AUC: {auc_score:.3f}, Precision: {report['1']['precision']:.3f}, Recall: {report['1']['recall']:.3f}")

        return evaluation_results

    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train the model first.")
        return self.feature_importance

    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'model': self.model,
            'anomaly_detector': self.anomaly_detector,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.anomaly_detector = model_data['anomaly_detector']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data['feature_importance']
        self.is_trained = model_data['is_trained']

        logger.info(f"Model loaded from {filepath}")

    def predict_single_transaction(self, transaction_features: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Convert to DataFrame
        X = pd.DataFrame([transaction_features])

        # Get predictions
        fraud_probability = self.predict_proba(X)[0]
        is_fraud = self.predict(X)[0]
        is_anomaly = self.detect_anomalies(X)[0]

        # Risk assessment
        risk_level = 'LOW'
        if fraud_probability > 0.7 or is_anomaly:
            risk_level = 'HIGH'
        elif fraud_probability > 0.3:
            risk_level = 'MEDIUM'

        result = {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_probability),
            'is_anomaly': bool(is_anomaly),
            'risk_level': risk_level,
            'confidence': float(max(fraud_probability, 1 - fraud_probability))
        }

        return result