import pandas as pd
import numpy as np
from data_processor import DataProcessor
from transformer_model import TransformerModel
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Load and process data
    logger.info("Loading and processing data...")
    data_path = "data/^GSPC_processed.csv"
    dp = DataProcessor()
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    df = dp.add_technical_indicators(df)
    df = dp.create_target(df)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Get feature columns (exclude date and target)
    feature_cols = [col for col in df.columns if col not in ['Date', 'target']]
    
    # Prepare features
    X, y = dp.prepare_features(df)
    
    # Initialize and build model
    sequence_length = 30  # Look back period
    n_features = len(feature_cols)
    
    logger.info(f"Building model with {n_features} features and sequence length {sequence_length}")
    model = TransformerModel(sequence_length=sequence_length, n_features=n_features)
    model.build_model(
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_transformer_blocks=3,
        mlp_units=[128, 64],
        dropout=0.2,
        mlp_dropout=0.3
    )
    
    # Prepare sequences for training
    X_sequences = []
    y_sequences = []
    
    # Create sequences
    for i in range(len(X) - sequence_length + 1):
        X_sequences.append(X[i:(i + sequence_length)])
        y_sequences.append(y[i + sequence_length - 1])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Split data into train and validation sets
    train_size = int(len(X_sequences) * 0.8)
    X_train, X_val = X_sequences[:train_size], X_sequences[train_size:]
    y_train, y_val = y_sequences[:train_size], y_sequences[train_size:]
    
    # Train model
    logger.info("Training model...")
    history = model.train(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32
    )
    
    # Evaluate on validation set
    logger.info("Evaluating model on validation set...")
    val_loss, val_accuracy = model.model.evaluate(X_val, y_val)
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = model.predict(X_val)
    
    # Calculate prediction accuracy
    pred_classes = (predictions > 0.5).astype(int)
    pred_accuracy = np.mean(pred_classes.flatten() == y_val)
    logger.info(f"Prediction Accuracy: {pred_accuracy:.4f}")

if __name__ == "__main__":
    main()
