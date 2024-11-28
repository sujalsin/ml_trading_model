import pandas as pd
import numpy as np
from typing import List, Tuple
import ta
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, prediction_horizon: int = 5):
        """
        Initialize DataProcessor with prediction horizon.
        
        Args:
            prediction_horizon: Number of days to look ahead for prediction
        """
        self.prediction_horizon = prediction_horizon
        self.scaler = MinMaxScaler()
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        """
        try:
            # Trend Indicators
            df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['ema_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['adx'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
            df['macd'] = ta.trend.macd_diff(df['Close'])
            
            # Momentum Indicators
            df['rsi'] = ta.momentum.rsi(df['Close'])
            df['stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
            df['cci'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
            df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
            
            # Volatility Indicators
            df['bbands_upper'] = ta.volatility.bollinger_hband(df['Close'])
            df['bbands_lower'] = ta.volatility.bollinger_lband(df['Close'])
            df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            # Volume Indicators
            df['obv'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            df['cmf'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Price Transforms
            df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
            df['return_volatility'] = df['log_return'].rolling(window=20).std()
            
            # Custom Features
            df['price_momentum'] = df['Close'] - df['Close'].shift(5)
            df['volume_momentum'] = df['Volume'] - df['Volume'].shift(5)
            df['high_low_diff'] = df['High'] - df['Low']
            
            # Clean up NaN values
            df = df.dropna()
            
            logger.info("Technical indicators added successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            raise
    
    def create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable based on future returns.
        """
        try:
            # Calculate future returns
            future_returns = df['Close'].shift(-self.prediction_horizon) / df['Close'] - 1
            
            # Create binary target (1 for positive returns, 0 for negative)
            df['target'] = (future_returns > 0).astype(int)
            
            # Remove rows with NaN targets
            df = df.dropna()
            
            logger.info("Target variable created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating target variable: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for model training.
        """
        try:
            # Select all features except Date and target
            feature_columns = [col for col in df.columns if col not in ['Date', 'target']]
            
            # Scale features
            X = self.scaler.fit_transform(df[feature_columns])
            y = df['target'].values
            
            logger.info("Features prepared successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def analyze_feature_importance(self, X: np.ndarray, y: np.ndarray, 
                                 feature_names: List[str]) -> pd.DataFrame:
        """
        Analyze feature importance using various methods.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import mutual_info_classif
            
            # Random Forest importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = rf.feature_importances_
            
            # Mutual Information
            mi_importance = mutual_info_classif(X, y)
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'RF_Importance': rf_importance,
                'MI_Importance': mi_importance
            })
            
            # Sort by Random Forest importance
            importance_df = importance_df.sort_values('RF_Importance', ascending=False)
            
            logger.info("Feature importance analysis completed")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            raise
