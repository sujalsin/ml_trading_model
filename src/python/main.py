import argparse
import logging
from data_collection import DataCollector
from model import StockPredictor
from evaluate import ModelEvaluator
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = ['data', 'models', 'results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"Created directory: {d}")

def train_model(args):
    """Train the predictive model."""
    # Initialize data collector
    collector = DataCollector()
    
    # Fetch and process data
    data = collector.fetch_stock_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if data is None:
        logger.error("Failed to fetch data")
        return
    
    # Calculate indicators and preprocess
    data = collector.calculate_technical_indicators(data)
    data = collector.preprocess_data(data)
    
    # Save processed data
    collector.save_data(data, f"{args.symbol}_processed")
    
    # Initialize and train model
    predictor = StockPredictor(sequence_length=args.sequence_length)
    history = predictor.train(
        data,
        validation_split=args.validation_split,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    predictor.save_model(f"models/{args.symbol}_model")
    
    logger.info("Training completed successfully")

def evaluate_model(args):
    """Evaluate the trained model."""
    # Load data
    data = pd.read_csv(f"data/{args.symbol}_processed.csv", index_col=0)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(f"models/{args.symbol}_model")
    
    # Evaluate model
    metrics = evaluator.evaluate(data)
    
    # Generate plots
    evaluator.plot_predictions(
        data,
        f"results/{args.symbol}_predictions.png"
    )
    evaluator.plot_confusion_matrix(
        data,
        f"results/{args.symbol}_confusion_matrix.png"
    )
    
    logger.info("Evaluation completed successfully")

def main():
    parser = argparse.ArgumentParser(
        description='Stock Price Prediction Model'
    )
    
    parser.add_argument('--mode', type=str, required=True,
                      choices=['train', 'evaluate'],
                      help='Mode of operation')
    
    parser.add_argument('--symbol', type=str, required=True,
                      help='Stock symbol to analyze')
    
    parser.add_argument('--start-date', type=str,
                      default='2020-01-01',
                      help='Start date for data collection')
    
    parser.add_argument('--end-date', type=str,
                      default=None,
                      help='End date for data collection')
    
    parser.add_argument('--sequence-length', type=int,
                      default=60,
                      help='Sequence length for LSTM')
    
    parser.add_argument('--validation-split', type=float,
                      default=0.2,
                      help='Validation split ratio')
    
    parser.add_argument('--epochs', type=int,
                      default=50,
                      help='Number of training epochs')
    
    parser.add_argument('--batch-size', type=int,
                      default=32,
                      help='Training batch size')
    
    args = parser.parse_args()
    
    # Create necessary directories
    setup_directories()
    
    if args.mode == 'train':
        train_model(args)
    else:
        evaluate_model(args)

if __name__ == "__main__":
    main()
