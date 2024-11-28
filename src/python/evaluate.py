import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import StockPredictor
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the saved model
        """
        self.predictor = StockPredictor()
        self.predictor.load_model(model_path)
        
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            df: Test data DataFrame
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        predictions = self.predictor.predict(df)
        predictions_binary = (predictions > 0.5).astype(int)
        
        # Get actual values
        _, y_true = self.predictor.prepare_data(df)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, predictions_binary),
            'precision': precision_score(y_true, predictions_binary),
            'recall': recall_score(y_true, predictions_binary),
            'f1': f1_score(y_true, predictions_binary)
        }
        
        logger.info("Evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
        return metrics
    
    def plot_predictions(self, df: pd.DataFrame, save_path: str = None):
        """
        Plot actual vs predicted values.
        
        Args:
            df: Test data DataFrame
            save_path: Optional path to save the plot
        """
        predictions = self.predictor.predict(df)
        _, y_true = self.predictor.prepare_data(df)
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual')
        plt.plot(predictions, label='Predicted')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel('Price Movement')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_confusion_matrix(self, df: pd.DataFrame, save_path: str = None):
        """
        Plot confusion matrix.
        
        Args:
            df: Test data DataFrame
            save_path: Optional path to save the plot
        """
        predictions = self.predictor.predict(df)
        predictions_binary = (predictions > 0.5).astype(int)
        _, y_true = self.predictor.prepare_data(df)
        
        cm = pd.crosstab(y_true, predictions_binary.ravel(),
                        rownames=['Actual'],
                        colnames=['Predicted'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()

if __name__ == "__main__":
    # Example usage
    # Load test data
    df = pd.read_csv("data/sp500_processed.csv", index_col=0)
    
    # Create evaluator
    evaluator = ModelEvaluator("models/lstm_model")
    
    # Evaluate model
    metrics = evaluator.evaluate(df)
    
    # Plot results
    evaluator.plot_predictions(df, "models/predictions.png")
    evaluator.plot_confusion_matrix(df, "models/confusion_matrix.png")
