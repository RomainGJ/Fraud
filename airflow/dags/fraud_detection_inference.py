from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.email import EmailOperator
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

# Default DAG arguments
default_args = {
    'owner': 'fraudguard-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'catchup': False,
}

# DAG for batch inference
inference_dag = DAG(
    'fraud_detection_batch_inference',
    default_args=default_args,
    description='Batch fraud detection inference pipeline',
    schedule_interval='@hourly',  # Run every hour
    max_active_runs=3,
    tags=['ml', 'fraud-detection', 'inference']
)

def process_batch_transactions(**context):
    """Process batch of transactions for fraud detection"""
    logger.info("Starting batch transaction processing...")

    # Look for new transaction files
    input_dir = "/opt/airflow/data/incoming"
    output_dir = "/opt/airflow/data/predictions"

    os.makedirs(output_dir, exist_ok=True)

    # Find new transaction files
    transaction_files = []
    if os.path.exists(input_dir):
        for file in os.listdir(input_dir):
            if file.endswith('.csv') and 'transactions' in file:
                transaction_files.append(os.path.join(input_dir, file))

    if not transaction_files:
        logger.info("No new transaction files found")
        return "no_files"

    # Load production model
    model_path = "/opt/airflow/models/production/fraud_detector.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Production model not found at {model_path}")

    from src.models.fraud_detector import FraudDetector
    detector = FraudDetector()
    detector.load_model(model_path)

    total_processed = 0
    total_fraud_detected = 0

    for file_path in transaction_files:
        logger.info(f"Processing file: {file_path}")

        # Load transactions
        transactions = pd.read_csv(file_path)

        # Make predictions
        predictions = []
        for _, transaction in transactions.iterrows():
            try:
                result = detector.predict_single_transaction(transaction.to_dict())
                predictions.append({
                    'transaction_id': transaction.get('transaction_id', ''),
                    'is_fraud': result['is_fraud'],
                    'fraud_probability': result['fraud_probability'],
                    'risk_level': result['risk_level'],
                    'prediction_timestamp': datetime.now().isoformat()
                })

                if result['is_fraud']:
                    total_fraud_detected += 1

            except Exception as e:
                logger.error(f"Error processing transaction: {str(e)}")

        total_processed += len(transactions)

        # Save predictions
        predictions_df = pd.DataFrame(predictions)
        output_file = os.path.join(
            output_dir,
            f"predictions_{os.path.basename(file_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        predictions_df.to_csv(output_file, index=False)

        # Move processed file to archive
        archive_dir = "/opt/airflow/data/processed"
        os.makedirs(archive_dir, exist_ok=True)
        import shutil
        shutil.move(file_path, os.path.join(archive_dir, os.path.basename(file_path)))

    # Push results to XCom
    context['task_instance'].xcom_push(key='total_processed', value=total_processed)
    context['task_instance'].xcom_push(key='total_fraud_detected', value=total_fraud_detected)

    logger.info(f"Batch processing completed. Processed: {total_processed}, Fraud detected: {total_fraud_detected}")

    return f"processed_{total_processed}_transactions"

def generate_fraud_alerts(**context):
    """Generate alerts for detected fraud"""
    logger.info("Generating fraud alerts...")

    total_fraud_detected = context['task_instance'].xcom_pull(key='total_fraud_detected', task_ids='process_transactions')

    if total_fraud_detected == 0:
        logger.info("No fraud detected in this batch")
        return "no_alerts"

    # Load recent predictions with fraud
    predictions_dir = "/opt/airflow/data/predictions"
    alert_threshold = 0.8  # High confidence fraud threshold

    high_risk_transactions = []

    # Scan recent prediction files
    for file in os.listdir(predictions_dir):
        if file.startswith('predictions_') and file.endswith('.csv'):
            file_path = os.path.join(predictions_dir, file)

            # Check if file is from this batch (within last hour)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            if datetime.now() - file_time < timedelta(hours=1):
                predictions = pd.read_csv(file_path)

                # Filter high-risk transactions
                high_risk = predictions[
                    (predictions['is_fraud'] == True) &
                    (predictions['fraud_probability'] >= alert_threshold)
                ]

                high_risk_transactions.extend(high_risk.to_dict('records'))

    if high_risk_transactions:
        # Save alert file
        alerts_dir = "/opt/airflow/data/alerts"
        os.makedirs(alerts_dir, exist_ok=True)

        alert_file = os.path.join(
            alerts_dir,
            f"fraud_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        pd.DataFrame(high_risk_transactions).to_csv(alert_file, index=False)

        logger.info(f"Generated {len(high_risk_transactions)} high-risk fraud alerts")

    context['task_instance'].xcom_push(key='high_risk_count', value=len(high_risk_transactions))

    return f"generated_{len(high_risk_transactions)}_alerts"

def update_model_monitoring(**context):
    """Update model performance monitoring metrics"""
    logger.info("Updating model monitoring metrics...")

    total_processed = context['task_instance'].xcom_pull(key='total_processed', task_ids='process_transactions')
    total_fraud_detected = context['task_instance'].xcom_pull(key='total_fraud_detected', task_ids='process_transactions')

    if total_processed == 0:
        return "no_metrics_to_update"

    fraud_rate = total_fraud_detected / total_processed

    # Update Prometheus metrics (in a real setup)
    # Here we'll just log the metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_transactions_processed': total_processed,
        'fraud_detected': total_fraud_detected,
        'fraud_rate': fraud_rate,
        'batch_id': context['dag_run'].run_id
    }

    # Save metrics to file (would be sent to Prometheus in production)
    metrics_dir = "/opt/airflow/data/metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file = os.path.join(
        metrics_dir,
        f"batch_metrics_{datetime.now().strftime('%Y%m%d_%H')}.json"
    )

    import json
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')

    logger.info(f"Metrics updated: {metrics}")

    # Check for anomalies in fraud rate
    if fraud_rate > 0.1:  # If fraud rate is unusually high
        logger.warning(f"High fraud rate detected: {fraud_rate:.3f}")
        context['task_instance'].xcom_push(key='anomaly_detected', value=True)
    else:
        context['task_instance'].xcom_push(key='anomaly_detected', value=False)

    return metrics

# Task definitions for inference DAG
wait_for_transactions = FileSensor(
    task_id='wait_for_new_transactions',
    filepath='/opt/airflow/data/incoming',
    fs_conn_id='fs_default',
    poke_interval=300,  # Check every 5 minutes
    timeout=3600,  # Timeout after 1 hour
    dag=inference_dag,
)

process_transactions = PythonOperator(
    task_id='process_transactions',
    python_callable=process_batch_transactions,
    dag=inference_dag,
)

generate_alerts = PythonOperator(
    task_id='generate_fraud_alerts',
    python_callable=generate_fraud_alerts,
    dag=inference_dag,
)

update_monitoring = PythonOperator(
    task_id='update_monitoring',
    python_callable=update_model_monitoring,
    dag=inference_dag,
)

cleanup_old_files = BashOperator(
    task_id='cleanup_old_files',
    bash_command="""
    # Clean up files older than 7 days
    find /opt/airflow/data/processed -name "*.csv" -mtime +7 -delete
    find /opt/airflow/data/predictions -name "*.csv" -mtime +7 -delete
    echo "Cleanup completed"
    """,
    dag=inference_dag,
)

# Define task dependencies for inference DAG
wait_for_transactions >> process_transactions >> [generate_alerts, update_monitoring] >> cleanup_old_files