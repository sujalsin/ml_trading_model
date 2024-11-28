import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import logging
from typing import Tuple, List, Dict, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.model = None
        
    def create_sequences(self, 
                        data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM model.
        
        Args:
            data: Input data array
            
        Returns:
            Tuple of (X, y) where X is the sequence data and y is the target
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, -1])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]):
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
        """
        self.model = Sequential([
            LSTM(units=32, return_sequences=True, 
                 input_shape=input_shape,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(units=16,
                 kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.2),
            Dense(units=1, activation='sigmoid',
                  kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ])
        
        # Custom loss function that considers both direction and magnitude
        def custom_loss(y_true, y_pred):
            # Binary cross-entropy for direction
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            # Mean squared error for magnitude
            mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
            # Add L2 regularization loss
            reg_loss = tf.reduce_sum(self.model.losses)
            return bce + 0.2 * mse + reg_loss
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss=custom_loss,
            metrics=['accuracy']
        )
        
        logger.info("Model built successfully")
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X, y) for training
        """
        # Select features
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume',
                         'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line',
                         'BB_middle', 'BB_upper', 'BB_lower', 'Returns',
                         'Volatility']
        
        # Scale each feature independently
        scaled_data = []
        for col in feature_columns:
            scaler = StandardScaler()
            scaled_col = scaler.fit_transform(df[col].values.reshape(-1, 1))
            scaled_data.append(scaled_col)
        
        # Combine scaled features
        data = np.hstack(scaled_data)
        
        # Add target
        data = np.column_stack([data, df['Target'].values])
        
        # Create sequences
        X, y = self.create_sequences(data)
        
        return X, y
    
    def train(self, 
              df: pd.DataFrame,
              validation_split: float = 0.2,
              epochs: int = 50,
              batch_size: int = 32):
        """
        Train the model.
        
        Args:
            df: Input DataFrame
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        # Prepare data
        X, y = self.prepare_data(df)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, shuffle=False
        )
        
        # Build model if not already built
        if self.model is None:
            self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Define callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.00001,
            min_delta=0.001
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        logger.info("Model training completed")
        return history
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Prepare data
        X, _ = self.prepare_data(df)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions
    
    def save_model(self, filepath: str):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Load processed data
    df = pd.read_csv("data/sp500_processed.csv", index_col=0)
    
    # Create and train model
    predictor = StockPredictor(sequence_length=60)
    history = predictor.train(df, epochs=50)
    
    # Save model
    predictor.save_model("models/lstm_model")
