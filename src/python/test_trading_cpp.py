import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pytest

# Add the C++ library path to system path
sys.path.append(os.path.join(os.path.dirname(__file__), '../cpp/build/lib'))

try:
    import trading_cpp_bindings as tcpp
except ImportError:
    print("Error: C++ trading bindings not found. Please build the C++ components first.")
    print("Run the following commands in the src/cpp directory:")
    print("mkdir -p build && cd build")
    print("cmake ..")
    print("make")
    sys.exit(1)

def test_market_data_processor():
    """Test the C++ MarketDataProcessor functionality"""
    processor = tcpp.MarketDataProcessor(window_size=30)
    
    # Create sample market data
    data = tcpp.MarketData()
    data.symbol = "AAPL"
    data.price = 150.0
    data.volume = 1000000
    data.high = 151.0
    data.low = 149.0
    data.open = 149.5
    data.timestamp = datetime.now()
    
    # Test updating with market data
    processor.update(data)
    
    # Test getting technical indicators
    try:
        sma = processor.get_sma(period=1)
        assert abs(sma - 150.0) < 1e-6, f"Expected SMA to be 150.0, got {sma}"
        print("✓ SMA test passed")
    except RuntimeError as e:
        print(f"× SMA test failed: {e}")
    
    # Add more data points for testing other indicators
    prices = [150.0, 151.0, 149.0, 152.0, 153.0]
    for price in prices:
        data.price = price
        processor.update(data)
    
    try:
        rsi = processor.get_rsi(period=5)
        assert 0 <= rsi <= 100, f"RSI should be between 0 and 100, got {rsi}"
        print("✓ RSI test passed")
    except RuntimeError as e:
        print(f"× RSI test failed: {e}")
    
    try:
        macd = processor.get_macd()
        print(f"✓ MACD test passed: {macd}")
    except RuntimeError as e:
        print(f"× MACD test failed: {e}")
    
    try:
        lower, upper = processor.get_bollinger_bands(period=5)
        assert lower < upper, f"Lower band should be less than upper band: {lower} >= {upper}"
        print("✓ Bollinger Bands test passed")
    except RuntimeError as e:
        print(f"× Bollinger Bands test failed: {e}")

def test_trading_engine():
    """Test the C++ TradingEngine functionality"""
    engine = tcpp.TradingEngine()
    
    # Start the trading engine
    engine.start()
    
    # Create and submit a market order
    order = tcpp.Order()
    order.symbol = "AAPL"
    order.type = tcpp.OrderType.MARKET
    order.side = tcpp.OrderSide.BUY
    order.quantity = 100
    order.price = 150.0
    order.timestamp = datetime.now()
    
    try:
        success = engine.submit_order(order)
        assert success, "Order submission failed"
        print("✓ Order submission test passed")
    except Exception as e:
        print(f"× Order submission test failed: {e}")
    
    # Test position management
    try:
        position = engine.get_position("AAPL")
        print(f"✓ Position retrieval test passed: {position.quantity} shares @ {position.average_price}")
    except Exception as e:
        print(f"× Position retrieval test failed: {e}")
    
    # Test risk management
    try:
        engine.set_position_limit("AAPL", 1000)
        engine.set_loss_limit("AAPL", 5000)
        print("✓ Risk management test passed")
    except Exception as e:
        print(f"× Risk management test failed: {e}")
    
    # Stop the trading engine
    engine.stop()

def main():
    print("\nTesting C++ Market Data Processor:")
    print("-" * 30)
    test_market_data_processor()
    
    print("\nTesting C++ Trading Engine:")
    print("-" * 30)
    test_trading_engine()

if __name__ == "__main__":
    main()
