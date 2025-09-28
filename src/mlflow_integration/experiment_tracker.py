import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
import joblib
import json
import os
from datetime import datetime
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowExperimentTracker:
    def __init__(self, experiment_name: str = "fraud-detection", tracking_uri: str = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri or "http://localhost:5000"
        self.experiment_id = None
        self.setup_mlflow()

    def setup_mlflow(self):
        """Initialize MLflow tracking"""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)

            # Create or get experiment
            try:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created new experiment: {self.experiment_name}")
            except mlflow.exceptions.MlflowException:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name}")

            mlflow.set_experiment(self.experiment_name)

        except Exception as e:
            logger.warning(f"MLflow setup failed: {str(e)}. Running without tracking.")
            self.experiment_id = None

    def start_run(self, run_name: str = None) -> str:
        """Start a new MLflow run"""
        if self.experiment_id is None:
            logger.warning("MLflow not initialized. Skipping run creation.")
            return None

        try:
            run = mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run.info.run_id
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {str(e)}")
            return None

    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        if mlflow.active_run() is None:
            logger.warning("No active MLflow run. Skipping parameter logging.")
            return

        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            logger.info(f"Logged {len(params)} parameters to MLflow")
        except Exception as e:
            logger.error(f"Failed to log parameters: {str(e)}")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow"""
        if mlflow.active_run() is None:
            logger.warning("No active MLflow run. Skipping metrics logging.")
            return

        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
            logger.info(f"Logged {len(metrics)} metrics to MLflow")
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")

    def log_model(self, model, model_name: str, X_sample: pd.DataFrame = None,
                  signature=None, input_example=None):
        """Log model to MLflow"""
        if mlflow.active_run() is None:
            logger.warning("No active MLflow run. Skipping model logging.")
            return

        try:
            # Infer signature if not provided
            if signature is None and X_sample is not None:
                try:
                    y_sample = model.predict(X_sample[:5])  # Small sample for signature
                    signature = infer_signature(X_sample[:5], y_sample)
                except Exception as e:
                    logger.warning(f"Could not infer signature: {str(e)}")

            # Log the model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                signature=signature,
                input_example=input_example
            )

            logger.info(f"Model '{model_name}' logged to MLflow")

        except Exception as e:
            logger.error(f"Failed to log model: {str(e)}")

    def log_artifacts(self, artifacts: Dict[str, Any]):
        """Log additional artifacts like plots, feature importance, etc."""
        if mlflow.active_run() is None:
            logger.warning("No active MLflow run. Skipping artifacts logging.")
            return

        try:
            for name, artifact in artifacts.items():
                if isinstance(artifact, pd.DataFrame):
                    # Save DataFrame as CSV
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                        artifact.to_csv(f.name, index=False)
                        mlflow.log_artifact(f.name, f"{name}.csv")
                        os.unlink(f.name)

                elif isinstance(artifact, dict):
                    # Save dict as JSON
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(artifact, f, indent=2, default=str)
                        mlflow.log_artifact(f.name, f"{name}.json")
                        os.unlink(f.name)

                elif isinstance(artifact, str) and os.path.exists(artifact):
                    # Log file directly
                    mlflow.log_artifact(artifact, name)

            logger.info(f"Logged {len(artifacts)} artifacts to MLflow")

        except Exception as e:
            logger.error(f"Failed to log artifacts: {str(e)}")

    def log_training_session(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series,
                           training_params: Dict, evaluation_results: Dict,
                           feature_importance: pd.DataFrame = None):
        """Log complete training session to MLflow"""

        run_name = f"fraud-detection-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run_id = self.start_run(run_name)

        if run_id is None:
            logger.warning("Could not start MLflow run. Training will continue without tracking.")
            return None

        try:
            # Log parameters
            params = {
                "model_type": training_params.get("model_type", "unknown"),
                "n_features": X_train.shape[1],
                "n_train_samples": X_train.shape[0],
                "n_test_samples": X_test.shape[0],
                "fraud_rate_train": float(y_train.mean()),
                "fraud_rate_test": float(y_test.mean()),
                "training_date": datetime.now().isoformat()
            }
            params.update(training_params)
            self.log_parameters(params)

            # Log metrics
            metrics = {
                "test_auc": evaluation_results.get("auc_score", 0),
                "test_precision": evaluation_results.get("precision", 0),
                "test_recall": evaluation_results.get("recall", 0),
                "test_f1": evaluation_results.get("f1_score", 0),
                "test_pr_auc": evaluation_results.get("pr_auc", 0),
                "anomaly_detection_precision": evaluation_results.get("anomaly_detection_precision", 0)
            }
            self.log_metrics(metrics)

            # Log model with signature
            self.log_model(
                model=model,
                model_name="fraud_detector",
                X_sample=X_test.head(10),
                input_example=X_test.head(1).to_dict(orient="records")[0]
            )

            # Log artifacts
            artifacts = {
                "evaluation_results": evaluation_results,
                "training_params": training_params
            }

            if feature_importance is not None:
                artifacts["feature_importance"] = feature_importance

            self.log_artifacts(artifacts)

            # Add tags
            mlflow.set_tags({
                "model_type": training_params.get("model_type", "unknown"),
                "data_version": "v1.0",
                "pipeline_version": "v2.0",
                "environment": "development"
            })

            logger.info(f"Complete training session logged to MLflow run: {run_id}")
            return run_id

        except Exception as e:
            logger.error(f"Failed to log training session: {str(e)}")
            return None
        finally:
            if mlflow.active_run():
                mlflow.end_run()

    def register_model(self, run_id: str, model_name: str = "fraud-detection-model",
                      stage: str = "Staging"):
        """Register model in MLflow Model Registry"""
        try:
            model_uri = f"runs:/{run_id}/fraud_detector"

            # Register the model
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )

            # Transition to specified stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )

            logger.info(f"Model registered as {model_name} v{model_version.version} in {stage}")
            return model_version

        except Exception as e:
            logger.error(f"Failed to register model: {str(e)}")
            return None

    def load_model_from_registry(self, model_name: str = "fraud-detection-model",
                                stage: str = "Production"):
        """Load model from MLflow Model Registry"""
        try:
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.sklearn.load_model(model_uri)

            logger.info(f"Loaded model {model_name}/{stage} from registry")
            return model

        except Exception as e:
            logger.error(f"Failed to load model from registry: {str(e)}")
            return None

    def compare_models(self, metric_name: str = "test_auc", limit: int = 10):
        """Compare models from recent runs"""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                logger.warning("Experiment not found")
                return None

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} DESC"],
                max_results=limit
            )

            if runs.empty:
                logger.warning("No runs found for comparison")
                return None

            comparison = runs[[
                'run_id', 'status', 'start_time',
                f'metrics.{metric_name}', 'metrics.test_precision',
                'metrics.test_recall', 'metrics.test_f1',
                'params.model_type'
            ]].copy()

            comparison = comparison.sort_values(f'metrics.{metric_name}', ascending=False)

            logger.info(f"Retrieved {len(comparison)} runs for comparison")
            return comparison

        except Exception as e:
            logger.error(f"Failed to compare models: {str(e)}")
            return None

    def get_best_model(self, metric_name: str = "test_auc"):
        """Get the best performing model based on a metric"""
        try:
            comparison = self.compare_models(metric_name=metric_name, limit=1)
            if comparison is None or comparison.empty:
                return None

            best_run_id = comparison.iloc[0]['run_id']
            model_uri = f"runs:/{best_run_id}/fraud_detector"

            model = mlflow.sklearn.load_model(model_uri)

            logger.info(f"Loaded best model from run {best_run_id}")
            return model, best_run_id

        except Exception as e:
            logger.error(f"Failed to get best model: {str(e)}")
            return None, None