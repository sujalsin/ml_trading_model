#pragma once

#include <string>
#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include <chrono>

namespace trading {

struct MarketData {
    std::string symbol;
    double price;
    double volume;
    double high;
    double low;
    double open;
    std::chrono::system_clock::time_point timestamp;
};

class MarketDataProcessor {
public:
    MarketDataProcessor(size_t window_size = 30);
    
    // Add new market data point and update indicators
    void update(const MarketData& data);
    
    // Get technical indicators
    double get_sma(size_t period) const;
    double get_ema(size_t period) const;
    double get_rsi(size_t period = 14) const;
    double get_macd() const;
    std::pair<double, double> get_bollinger_bands(size_t period = 20, double std_dev = 2.0) const;
    
    // Get historical data
    std::vector<MarketData> get_history() const;
    
    // Clear historical data
    void clear();

private:
    size_t window_size_;
    std::deque<MarketData> history_;
    
    // Cache for technical indicators
    mutable std::unordered_map<size_t, double> sma_cache_;
    mutable std::unordered_map<size_t, double> ema_cache_;
    
    // Helper functions
    void update_cache();
    double calculate_sma(size_t period) const;
    double calculate_ema(size_t period) const;
};

} // namespace trading
