import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None

    def load_data(self, file_path, target_column='is_fraud'):
        try:
            data = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, data, target_column='is_fraud'):
        data_clean = data.copy()

        # Handle missing values
        numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
        data_clean[numeric_columns] = data_clean[numeric_columns].fillna(data_clean[numeric_columns].median())

        categorical_columns = data_clean.select_dtypes(include=['object']).columns
        categorical_columns = categorical_columns.drop(target_column, errors='ignore')

        for col in categorical_columns:
            data_clean[col] = data_clean[col].fillna(data_clean[col].mode()[0] if not data_clean[col].mode().empty else 'Unknown')

        # Encode categorical variables
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            data_clean[col] = self.label_encoders[col].fit_transform(data_clean[col].astype(str))

        # Prepare features and target
        X = data_clean.drop(target_column, axis=1)
        y = data_clean[target_column] if target_column in data_clean.columns else None

        self.feature_columns = X.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        logger.info(f"Data preprocessing completed. Features shape: {X_scaled.shape}")

        return X_scaled, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def generate_synthetic_data(self, n_samples=10000, fraud_rate=0.05):
        np.random.seed(42)

        # Generate synthetic transaction features
        data = {
            'transaction_amount': np.random.exponential(100, n_samples),
            'account_age_days': np.random.randint(1, 3650, n_samples),
            'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'online', 'retail'], n_samples),
            'time_of_day': np.random.randint(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'transaction_count_last_hour': np.random.poisson(1, n_samples),
            'average_transaction_amount': np.random.normal(75, 25, n_samples),
            'location_risk_score': np.random.beta(2, 5, n_samples),
            'payment_method': np.random.choice(['credit', 'debit', 'online', 'mobile'], n_samples)
        }

        df = pd.DataFrame(data)

        # Generate fraud labels (biased towards certain conditions)
        fraud_probability = (
            (df['transaction_amount'] > 500) * 0.3 +
            (df['account_age_days'] < 30) * 0.2 +
            (df['location_risk_score'] > 0.8) * 0.4 +
            (df['transaction_count_last_hour'] > 5) * 0.3
        )

        fraud_probability = np.clip(fraud_probability, 0, 0.8)

        n_fraud = int(n_samples * fraud_rate)
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False,
                                       p=fraud_probability/fraud_probability.sum())

        df['is_fraud'] = 0
        df.loc[fraud_indices, 'is_fraud'] = 1

        logger.info(f"Synthetic data generated. Shape: {df.shape}, Fraud rate: {df['is_fraud'].mean():.3f}")

        return df