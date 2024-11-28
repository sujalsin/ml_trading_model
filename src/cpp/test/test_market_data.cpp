#include "market_data.hpp"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace trading;

void test_sma() {
    MarketDataProcessor processor(5);
    
    // Add test data
    for (double price : {100.0, 110.0, 120.0, 130.0, 140.0}) {
        MarketData data;
        data.price = price;
        processor.update(data);
    }
    
    double sma = processor.get_sma(5);
    assert(std::abs(sma - 120.0) < 1e-6);
    std::cout << "SMA test passed\n";
}

void test_ema() {
    MarketDataProcessor processor(5);
    
    // Add test data
    for (double price : {100.0, 110.0, 120.0, 130.0, 140.0}) {
        MarketData data;
        data.price = price;
        processor.update(data);
    }
    
    double ema = processor.get_ema(5);
    // EMA will be weighted towards more recent values
    assert(ema > 120.0);
    std::cout << "EMA test passed\n";
}

void test_rsi() {
    MarketDataProcessor processor(5);
    
    // Add test data with upward trend
    for (double price : {100.0, 110.0, 120.0, 130.0, 140.0}) {
        MarketData data;
        data.price = price;
        processor.update(data);
    }
    
    double rsi = processor.get_rsi(5);
    assert(rsi >= 0.0 && rsi <= 100.0);
    assert(rsi > 50.0);  // Should be high due to upward trend
    std::cout << "RSI test passed\n";
}

void test_bollinger_bands() {
    MarketDataProcessor processor(5);
    
    // Add test data
    for (double price : {100.0, 110.0, 120.0, 130.0, 140.0}) {
        MarketData data;
        data.price = price;
        processor.update(data);
    }
    
    auto [lower, upper] = processor.get_bollinger_bands(5);
    assert(lower < upper);
    assert(lower < 120.0 && upper > 120.0);  // Bands should surround the mean
    std::cout << "Bollinger Bands test passed\n";
}

int main() {
    try {
        test_sma();
        test_ema();
        test_rsi();
        test_bollinger_bands();
        std::cout << "All tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
