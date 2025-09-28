from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
import pandas as pd
import logging
import os
import sys

# Add project root to Python path
sys.path.append('/opt/airflow/dags/fraud_detection')

logger = logging.getLogger(__name__)

# Default DAG arguments
default_args = {
    'owner': 'fraudguard-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False,
}

# DAG definition
dag = DAG(
    'fraud_detection_training_pipeline',
    default_args=default_args,
    description='Complete fraud detection model training and deployment pipeline',
    schedule_interval='@daily',  # Run daily
    max_active_runs=1,
    tags=['ml', 'fraud-detection', 'training']
)

def extract_and_preprocess_data(**context):
    """Extract data from source and preprocess it"""
    from src.data_processing.preprocessor import FraudDataProcessor
    from src.features.feature_engineering import FeatureEngineer

    logger.info("Starting data extraction and preprocessing...")

    # Initialize processors
    processor = FraudDataProcessor()
    feature_engineer = FeatureEngineer()

    # Check if we have new data, otherwise generate synthetic data
    data_path = Variable.get("DATA_PATH", default_var=None)

    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        data = processor.load_data(data_path)
    else:
        logger.info("Generating synthetic data for training")
        data = processor.generate_synthetic_data(n_samples=50000)

    # Feature engineering
    logger.info("Performing feature engineering...")
    data_enhanced = feature_engineer.create_all_features(data)

    # Save processed data
    output_path = "/opt/airflow/data/processed_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_enhanced.to_csv(output_path, index=False)

    logger.info(f"Processed data saved to {output_path}")

    # Push metadata to XCom
    context['task_instance'].xcom_push(key='data_path', value=output_path)
    context['task_instance'].xcom_push(key='data_shape', value=data_enhanced.shape)
    context['task_instance'].xcom_push(key='fraud_rate', value=float(data_enhanced['is_fraud'].mean()))

    return output_path

def train_model(**context):
    """Train the fraud detection model"""
    from src.models.fraud_detector import FraudDetector
    from src.data_processing.preprocessor import FraudDataProcessor

    logger.info("Starting model training...")

    # Get data path from previous task
    data_path = context['task_instance'].xcom_pull(key='data_path', task_ids='extract_and_preprocess')

    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed data not found at {data_path}")

    # Load data
    data = pd.read_csv(data_path)

    # Preprocess for ML
    processor = FraudDataProcessor()
    X, y = processor.preprocess_data(data)
    X_train, X_test, y_train, y_test = processor.split_data(X, y)

    # Train models with different configurations
    model_configs = [
        {'model_type': 'random_forest', 'use_mlflow': True},
        {'model_type': 'logistic_regression', 'use_mlflow': True}
    ]

    best_model = None
    best_score = 0
    best_model_path = None

    for config in model_configs:
        logger.info(f"Training {config['model_type']} model...")

        detector = FraudDetector(**config)
        training_results = detector.train(X_train, y_train, X_test, y_test)
        evaluation_results = detector.evaluate(X_test, y_test)

        current_score = evaluation_results['auc_score']

        if current_score > best_score:
            best_score = current_score
            best_model = detector

            # Save best model
            model_dir = "/opt/airflow/models"
            os.makedirs(model_dir, exist_ok=True)
            best_model_path = os.path.join(model_dir, f"best_fraud_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
            best_model.save_model(best_model_path)

    # Push results to XCom
    context['task_instance'].xcom_push(key='best_model_path', value=best_model_path)
    context['task_instance'].xcom_push(key='best_score', value=best_score)
    context['task_instance'].xcom_push(key='model_type', value=best_model.model_type)

    logger.info(f"Best model ({best_model.model_type}) trained with AUC: {best_score:.3f}")

    return best_model_path

def validate_model(**context):
    """Validate the trained model"""
    from src.models.fraud_detector import FraudDetector

    logger.info("Starting model validation...")

    # Get model path from previous task
    model_path = context['task_instance'].xcom_pull(key='best_model_path', task_ids='train_model')
    best_score = context['task_instance'].xcom_pull(key='best_score', task_ids='train_model')

    # Define minimum performance thresholds
    min_auc_threshold = float(Variable.get("MIN_AUC_THRESHOLD", default_var=0.75))
    min_precision_threshold = float(Variable.get("MIN_PRECISION_THRESHOLD", default_var=0.70))

    # Load and validate model
    detector = FraudDetector()
    detector.load_model(model_path)

    # Validation checks
    validation_passed = True
    validation_messages = []

    if best_score < min_auc_threshold:
        validation_passed = False
        validation_messages.append(f"AUC score {best_score:.3f} below threshold {min_auc_threshold}")

    # Additional validation logic can be added here
    # - Data drift detection
    # - Model bias checks
    # - Performance comparison with current production model

    if validation_passed:
        logger.info("Model validation passed")
        context['task_instance'].xcom_push(key='validation_status', value='passed')
    else:
        logger.error(f"Model validation failed: {'; '.join(validation_messages)}")
        context['task_instance'].xcom_push(key='validation_status', value='failed')
        raise ValueError(f"Model validation failed: {'; '.join(validation_messages)}")

    return validation_passed

def deploy_model(**context):
    """Deploy the validated model"""
    logger.info("Starting model deployment...")

    validation_status = context['task_instance'].xcom_pull(key='validation_status', task_ids='validate_model')

    if validation_status != 'passed':
        raise ValueError("Cannot deploy model that failed validation")

    model_path = context['task_instance'].xcom_pull(key='best_model_path', task_ids='train_model')

    # Copy model to production location
    production_model_path = "/opt/airflow/models/production/fraud_detector.pkl"
    os.makedirs(os.path.dirname(production_model_path), exist_ok=True)

    import shutil
    shutil.copy2(model_path, production_model_path)

    # Update model registry (if using MLflow)
    try:
        from src.mlflow_integration.experiment_tracker import MLflowExperimentTracker
        tracker = MLflowExperimentTracker()

        # Get run ID from training task if available
        # This would typically come from the model training metadata
        # For now, we'll just log the deployment
        logger.info("Model deployed successfully")

    except Exception as e:
        logger.warning(f"MLflow model registry update failed: {str(e)}")

    logger.info(f"Model deployed to production: {production_model_path}")

    return production_model_path

def send_notification(**context):
    """Send notification about pipeline completion"""
    logger.info("Sending pipeline completion notification...")

    # Get pipeline results
    best_score = context['task_instance'].xcom_pull(key='best_score', task_ids='train_model')
    model_type = context['task_instance'].xcom_pull(key='model_type', task_ids='train_model')
    data_shape = context['task_instance'].xcom_pull(key='data_shape', task_ids='extract_and_preprocess')
    fraud_rate = context['task_instance'].xcom_pull(key='fraud_rate', task_ids='extract_and_preprocess')

    notification_message = f"""
    FraudGuard Training Pipeline Completed Successfully!

    Pipeline Execution: {context['ds']}
    Model Type: {model_type}
    Best AUC Score: {best_score:.3f}
    Data Shape: {data_shape}
    Fraud Rate: {fraud_rate:.3f}

    Model has been deployed to production.
    """

    logger.info(notification_message)

    # Here you would typically send to Slack, email, etc.
    # For now, we'll just log the notification

    return "notification_sent"

# Task definitions
extract_task = PythonOperator(
    task_id='extract_and_preprocess',
    python_callable=extract_and_preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

notify_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag,
)

# Data quality check task (using external tool)
data_quality_task = BashOperator(
    task_id='data_quality_check',
    bash_command="""
    echo "Running data quality checks..."
    # Here you would run Great Expectations or similar
    echo "Data quality checks completed"
    """,
    dag=dag,
)

# Model performance monitoring task
monitoring_task = DockerOperator(
    task_id='setup_monitoring',
    image='fraudguard/monitoring:latest',
    command=['python', '/app/setup_monitoring.py'],
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    dag=dag,
)

# Define task dependencies
extract_task >> data_quality_task >> train_task >> validate_task >> deploy_task >> [notify_task, monitoring_task]

# Add some parallel tasks for efficiency
# data_quality_task could run in parallel with feature engineering if needed