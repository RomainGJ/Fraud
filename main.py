#!/usr/bin/env python3

import os
import sys
import pandas as pd
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.preprocessor import FraudDataProcessor
from models.fraud_detector import FraudDetector
from features.feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_model(data_path=None, model_type='random_forest', save_model=True):
    logger.info("Starting fraud detection model training...")

    # Initialize components
    processor = FraudDataProcessor()
    feature_engineer = FeatureEngineer()
    detector = FraudDetector(model_type=model_type)

    # Load or generate data
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        data = processor.load_data(data_path)
    else:
        logger.info("Generating synthetic data for demonstration")
        data = processor.generate_synthetic_data(n_samples=10000)

    # Feature engineering
    logger.info("Performing feature engineering...")
    data_enhanced = feature_engineer.create_all_features(data)

    # Preprocess data
    logger.info("Preprocessing data...")
    X, y = processor.preprocess_data(data_enhanced)

    # Split data
    X_train, X_test, y_train, y_test = processor.split_data(X, y)

    # Train model
    logger.info("Training model...")
    training_results = detector.train(X_train, y_train)

    # Evaluate model
    logger.info("Evaluating model...")
    evaluation_results = detector.evaluate(X_test, y_test)

    # Print results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    for key, value in training_results.items():
        print(f"{key}: {value}")

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in evaluation_results.items():
        if key != 'confusion_matrix':
            print(f"{key}: {value}")

    print("\nConfusion Matrix:")
    print(evaluation_results['confusion_matrix'])

    # Feature importance
    if detector.feature_importance is not None:
        print("\n" + "="*50)
        print("TOP 10 FEATURE IMPORTANCE")
        print("="*50)
        print(detector.feature_importance.head(10).to_string(index=False))

    # Save model if requested
    if save_model:
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)

        model_path = models_dir / 'fraud_detector.pkl'
        processor_path = models_dir / 'data_processor.pkl'

        detector.save_model(str(model_path))

        import joblib
        joblib.dump(processor, str(processor_path))

        logger.info(f"Models saved to {models_dir}")

    return detector, processor, evaluation_results

def predict_single_transaction():
    # Example transaction for testing
    sample_transaction = {
        'transaction_amount': 850.0,
        'account_age_days': 15,
        'merchant_category': 'online',
        'time_of_day': 23,
        'day_of_week': 6,
        'transaction_count_last_hour': 3,
        'average_transaction_amount': 120.0,
        'location_risk_score': 0.8,
        'payment_method': 'credit'
    }

    logger.info("Loading trained model for prediction...")

    try:
        import joblib
        detector = joblib.load('models/fraud_detector.pkl')

        logger.info("Making prediction on sample transaction...")
        result = detector.predict_single_transaction(sample_transaction)

        print("\n" + "="*50)
        print("FRAUD PREDICTION RESULT")
        print("="*50)
        print(f"Transaction: {sample_transaction}")
        print(f"\nPrediction Results:")
        for key, value in result.items():
            print(f"{key}: {value}")

    except FileNotFoundError:
        print("No trained model found. Please train a model first using --train")
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Fraud Detection System')
    parser.add_argument('--train', action='store_true', help='Train the fraud detection model')
    parser.add_argument('--predict', action='store_true', help='Make a prediction on sample data')
    parser.add_argument('--data', type=str, help='Path to training data CSV file')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'logistic_regression'],
                       help='Type of model to train')
    parser.add_argument('--api', action='store_true', help='Start the API server')

    args = parser.parse_args()

    if args.train:
        train_model(data_path=args.data, model_type=args.model_type)
    elif args.predict:
        predict_single_transaction()
    elif args.api:
        logger.info("Starting API server...")
        from api.app import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Please specify --train, --predict, or --api")
        print("Use --help for more information")

if __name__ == "__main__":
    main()