import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline
from typing import List, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    def fetch_stock_data(self, 
                        symbol: str,
                        start_date: str,
                        end_date: str = None) -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            logger.info(f"Successfully fetched data for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the given price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional technical indicators
        """
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for model training.
        
        Args:
            df: Raw DataFrame with OHLCV and technical indicators
            
        Returns:
            Preprocessed DataFrame
        """
        # Calculate returns first
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Handle missing values for each feature separately
        price_cols = ['Open', 'High', 'Low', 'Close']
        df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # Handle technical indicators
        tech_cols = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 
                    'BB_middle', 'BB_upper', 'BB_lower']
        df[tech_cols] = df[tech_cols].fillna(0)  # Fill with 0 as it's the neutral value
        
        # Handle returns and volatility
        df[['Returns', 'Volatility']] = df[['Returns', 'Volatility']].fillna(0)
        
        # Create target variable (1 if price goes up, 0 if down)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop rows with any remaining NaN values
        df = df.dropna()
        
        return df

    def save_data(self, df: pd.DataFrame, filename: str):
        """
        Save the processed data to disk.
        
        Args:
            df: DataFrame to save
            filename: Name of the file to save to
        """
        try:
            df.to_csv(f"data/{filename}.csv")
            logger.info(f"Successfully saved data to data/{filename}.csv")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")

if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    
    # Fetch data for S&P 500
    sp500_data = collector.fetch_stock_data(
        symbol="^GSPC",
        start_date="2020-01-01"
    )
    
    if sp500_data is not None:
        # Calculate indicators
        sp500_data = collector.calculate_technical_indicators(sp500_data)
        
        # Preprocess data
        sp500_data = collector.preprocess_data(sp500_data)
        
        # Save to disk
        collector.save_data(sp500_data, "sp500_processed")
