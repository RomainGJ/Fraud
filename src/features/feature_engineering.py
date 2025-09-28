import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_history = {}

    def create_temporal_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        df_enhanced = df.copy()

        if timestamp_col in df.columns:
            # Convert to datetime if not already
            df_enhanced[timestamp_col] = pd.to_datetime(df_enhanced[timestamp_col])

            # Extract temporal features
            df_enhanced['hour'] = df_enhanced[timestamp_col].dt.hour
            df_enhanced['day_of_week'] = df_enhanced[timestamp_col].dt.dayofweek
            df_enhanced['month'] = df_enhanced[timestamp_col].dt.month
            df_enhanced['is_weekend'] = (df_enhanced['day_of_week'] >= 5).astype(int)
            df_enhanced['is_night'] = ((df_enhanced['hour'] >= 22) | (df_enhanced['hour'] <= 6)).astype(int)

        return df_enhanced

    def create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enhanced = df.copy()

        if 'transaction_amount' in df.columns:
            # Amount-based features
            df_enhanced['amount_log'] = np.log1p(df_enhanced['transaction_amount'])
            df_enhanced['amount_zscore'] = (df_enhanced['transaction_amount'] - df_enhanced['transaction_amount'].mean()) / df_enhanced['transaction_amount'].std()

            # Binning amount into categories
            df_enhanced['amount_category'] = pd.cut(
                df_enhanced['transaction_amount'],
                bins=[0, 50, 200, 1000, float('inf')],
                labels=['small', 'medium', 'large', 'very_large']
            )

        return df_enhanced

    def create_user_behavior_features(self, df: pd.DataFrame, user_id_col: str = 'user_id') -> pd.DataFrame:
        df_enhanced = df.copy()

        if user_id_col in df.columns and 'transaction_amount' in df.columns:
            # User aggregated features
            user_stats = df.groupby(user_id_col)['transaction_amount'].agg([
                'count', 'mean', 'std', 'min', 'max', 'sum'
            ]).add_prefix('user_')

            # Merge back to original dataframe
            df_enhanced = df_enhanced.merge(user_stats, left_on=user_id_col, right_index=True, how='left')

            # User behavior features
            df_enhanced['user_amount_deviation'] = abs(df_enhanced['transaction_amount'] - df_enhanced['user_mean'])
            df_enhanced['user_amount_ratio'] = df_enhanced['transaction_amount'] / (df_enhanced['user_mean'] + 1e-8)

        return df_enhanced

    def create_merchant_features(self, df: pd.DataFrame, merchant_col: str = 'merchant_category') -> pd.DataFrame:
        df_enhanced = df.copy()

        if merchant_col in df.columns and 'transaction_amount' in df.columns:
            # Merchant risk score based on historical fraud rate
            merchant_fraud_rate = df.groupby(merchant_col)['is_fraud'].mean() if 'is_fraud' in df.columns else None

            if merchant_fraud_rate is not None:
                df_enhanced['merchant_risk_score'] = df_enhanced[merchant_col].map(merchant_fraud_rate)
            else:
                # Default risk scores for different merchant categories
                default_risk = {
                    'online': 0.3,
                    'gas': 0.1,
                    'grocery': 0.05,
                    'restaurant': 0.15,
                    'retail': 0.2
                }
                df_enhanced['merchant_risk_score'] = df_enhanced[merchant_col].map(default_risk).fillna(0.2)

            # Merchant transaction volume
            merchant_volume = df.groupby(merchant_col).size()
            df_enhanced['merchant_transaction_volume'] = df_enhanced[merchant_col].map(merchant_volume)

        return df_enhanced

    def create_velocity_features(self, df: pd.DataFrame, user_id_col: str = 'user_id',
                               timestamp_col: str = 'timestamp', window_hours: int = 24) -> pd.DataFrame:
        df_enhanced = df.copy()

        if all(col in df.columns for col in [user_id_col, timestamp_col, 'transaction_amount']):
            df_enhanced[timestamp_col] = pd.to_datetime(df_enhanced[timestamp_col])
            df_enhanced = df_enhanced.sort_values([user_id_col, timestamp_col])

            # Calculate velocity features within time windows
            velocity_features = []

            for window in [1, 6, 24]:  # 1 hour, 6 hours, 24 hours
                window_td = timedelta(hours=window)

                # Count of transactions in window
                count_col = f'tx_count_{window}h'
                df_enhanced[count_col] = 0

                # Sum of amounts in window
                amount_col = f'tx_amount_{window}h'
                df_enhanced[amount_col] = 0

                # This is a simplified version - in production, you'd use rolling windows
                for user_id in df_enhanced[user_id_col].unique():
                    user_mask = df_enhanced[user_id_col] == user_id
                    user_data = df_enhanced[user_mask].copy()

                    for idx in user_data.index:
                        current_time = user_data.loc[idx, timestamp_col]
                        window_start = current_time - window_td

                        window_mask = (
                            (user_data[timestamp_col] >= window_start) &
                            (user_data[timestamp_col] < current_time)
                        )

                        df_enhanced.loc[idx, count_col] = window_mask.sum()
                        df_enhanced.loc[idx, amount_col] = user_data.loc[window_mask, 'transaction_amount'].sum()

        return df_enhanced

    def create_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enhanced = df.copy()

        # Mock location features - in real scenarios, you'd have actual lat/lon
        if 'location_risk_score' not in df.columns:
            # Create synthetic location risk based on other features
            np.random.seed(42)
            df_enhanced['location_risk_score'] = np.random.beta(2, 5, len(df))

        # High-risk location indicator
        df_enhanced['is_high_risk_location'] = (df_enhanced['location_risk_score'] > 0.7).astype(int)

        return df_enhanced

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting comprehensive feature engineering...")

        df_enhanced = df.copy()

        # Apply all feature engineering steps
        df_enhanced = self.create_temporal_features(df_enhanced)
        df_enhanced = self.create_transaction_features(df_enhanced)
        df_enhanced = self.create_user_behavior_features(df_enhanced)
        df_enhanced = self.create_merchant_features(df_enhanced)
        df_enhanced = self.create_location_features(df_enhanced)

        # Add velocity features if timestamp is available
        if 'timestamp' in df.columns:
            df_enhanced = self.create_velocity_features(df_enhanced)

        # Create interaction features
        df_enhanced = self.create_interaction_features(df_enhanced)

        logger.info(f"Feature engineering completed. Original features: {df.shape[1]}, Enhanced features: {df_enhanced.shape[1]}")

        return df_enhanced

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_enhanced = df.copy()

        # Amount and time interactions
        if all(col in df.columns for col in ['transaction_amount', 'hour']):
            df_enhanced['amount_hour_interaction'] = df_enhanced['transaction_amount'] * df_enhanced['hour']

        # Risk score interactions
        if all(col in df.columns for col in ['merchant_risk_score', 'location_risk_score']):
            df_enhanced['combined_risk_score'] = df_enhanced['merchant_risk_score'] * df_enhanced['location_risk_score']

        # User behavior and amount
        if all(col in df.columns for col in ['user_amount_ratio', 'merchant_risk_score']):
            df_enhanced['user_merchant_risk'] = df_enhanced['user_amount_ratio'] * df_enhanced['merchant_risk_score']

        return df_enhanced

    def select_top_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> list:
        from sklearn.feature_selection import SelectKBest, f_classif

        # Select top features using ANOVA F-test
        selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
        selector.fit(X, y)

        selected_features = X.columns[selector.get_support()].tolist()

        logger.info(f"Selected top {len(selected_features)} features from {X.shape[1]} total features")

        return selected_features