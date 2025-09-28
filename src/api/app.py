from flask import Flask, request, jsonify
import pandas as pd
import joblib
import logging
from typing import Dict, Any
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model
fraud_detector = None
data_processor = None

def load_models():
    global fraud_detector, data_processor
    try:
        # Load models (paths would be configured in production)
        model_path = os.path.join('models', 'fraud_detector.pkl')
        processor_path = os.path.join('models', 'data_processor.pkl')

        if os.path.exists(model_path):
            fraud_detector = joblib.load(model_path)
            logger.info("Fraud detector model loaded successfully")

        if os.path.exists(processor_path):
            data_processor = joblib.load(processor_path)
            logger.info("Data processor loaded successfully")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': fraud_detector is not None,
        'processor_loaded': data_processor is not None
    })

@app.route('/predict', methods=['POST'])
def predict_fraud():
    try:
        if fraud_detector is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get transaction data from request
        transaction_data = request.json

        if not transaction_data:
            return jsonify({'error': 'No transaction data provided'}), 400

        # Validate required fields
        required_fields = [
            'transaction_amount', 'merchant_category', 'time_of_day',
            'account_age_days', 'location_risk_score'
        ]

        missing_fields = [field for field in required_fields if field not in transaction_data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400

        # Make prediction
        result = fraud_detector.predict_single_transaction(transaction_data)

        # Add transaction ID to response if provided
        if 'transaction_id' in transaction_data:
            result['transaction_id'] = transaction_data['transaction_id']

        logger.info(f"Prediction made: {result}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_fraud_batch():
    try:
        if fraud_detector is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get batch transaction data
        batch_data = request.json

        if not batch_data or 'transactions' not in batch_data:
            return jsonify({'error': 'No transaction data provided'}), 400

        transactions = batch_data['transactions']
        results = []

        for i, transaction in enumerate(transactions):
            try:
                result = fraud_detector.predict_single_transaction(transaction)
                result['transaction_index'] = i
                if 'transaction_id' in transaction:
                    result['transaction_id'] = transaction['transaction_id']
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing transaction {i}: {str(e)}")
                results.append({
                    'transaction_index': i,
                    'error': str(e)
                })

        return jsonify({
            'results': results,
            'total_processed': len(results),
            'successful_predictions': len([r for r in results if 'error' not in r])
        })

    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': 'Batch prediction failed', 'details': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    try:
        if fraud_detector is None:
            return jsonify({'error': 'Model not loaded'}), 500

        info = {
            'model_type': fraud_detector.model_type,
            'is_trained': fraud_detector.is_trained,
            'features_count': len(fraud_detector.feature_importance) if fraud_detector.feature_importance is not None else 0
        }

        return jsonify(info)

    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': 'Failed to get model info', 'details': str(e)}), 500

@app.route('/model/features', methods=['GET'])
def get_feature_importance():
    try:
        if fraud_detector is None:
            return jsonify({'error': 'Model not loaded'}), 500

        if fraud_detector.feature_importance is None:
            return jsonify({'error': 'Feature importance not available'}), 400

        # Convert to dict for JSON serialization
        features = fraud_detector.feature_importance.to_dict('records')

        return jsonify({
            'feature_importance': features,
            'total_features': len(features)
        })

    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return jsonify({'error': 'Failed to get feature importance', 'details': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load models on startup
    load_models()

    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)