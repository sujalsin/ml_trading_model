#include "market_data.hpp"
#include <numeric>
#include <cmath>
#include <stdexcept>

namespace trading {

MarketDataProcessor::MarketDataProcessor(size_t window_size)
    : window_size_(window_size) {}

void MarketDataProcessor::update(const MarketData& data) {
    history_.push_back(data);
    if (history_.size() > window_size_) {
        history_.pop_front();
    }
    update_cache();
}

double MarketDataProcessor::get_sma(size_t period) const {
    if (period > history_.size()) {
        throw std::runtime_error("Not enough data points for SMA calculation");
    }
    
    auto it = sma_cache_.find(period);
    if (it != sma_cache_.end()) {
        return it->second;
    }
    
    return calculate_sma(period);
}

double MarketDataProcessor::get_ema(size_t period) const {
    if (period > history_.size()) {
        throw std::runtime_error("Not enough data points for EMA calculation");
    }
    
    auto it = ema_cache_.find(period);
    if (it != ema_cache_.end()) {
        return it->second;
    }
    
    return calculate_ema(period);
}

double MarketDataProcessor::get_rsi(size_t period) const {
    if (history_.size() < period + 1) {
        throw std::runtime_error("Not enough data points for RSI calculation");
    }
    
    std::vector<double> gains;
    std::vector<double> losses;
    gains.reserve(period);
    losses.reserve(period);
    
    for (size_t i = 1; i <= period; ++i) {
        double diff = history_[history_.size() - i].price - 
                     history_[history_.size() - i - 1].price;
        if (diff > 0) {
            gains.push_back(diff);
            losses.push_back(0);
        } else {
            gains.push_back(0);
            losses.push_back(-diff);
        }
    }
    
    double avg_gain = std::accumulate(gains.begin(), gains.end(), 0.0) / period;
    double avg_loss = std::accumulate(losses.begin(), losses.end(), 0.0) / period;
    
    if (avg_loss == 0) {
        return 100;
    }
    
    double rs = avg_gain / avg_loss;
    return 100 - (100 / (1 + rs));
}

double MarketDataProcessor::get_macd() const {
    double ema_12 = get_ema(12);
    double ema_26 = get_ema(26);
    return ema_12 - ema_26;
}

std::pair<double, double> MarketDataProcessor::get_bollinger_bands(size_t period, double std_dev) const {
    if (history_.size() < period) {
        throw std::runtime_error("Not enough data points for Bollinger Bands calculation");
    }
    
    double sma = get_sma(period);
    
    // Calculate standard deviation
    double variance = 0;
    for (size_t i = 0; i < period; ++i) {
        double diff = history_[history_.size() - 1 - i].price - sma;
        variance += diff * diff;
    }
    variance /= period;
    double std = std::sqrt(variance);
    
    return {sma - std_dev * std, sma + std_dev * std};
}

std::vector<MarketData> MarketDataProcessor::get_history() const {
    return std::vector<MarketData>(history_.begin(), history_.end());
}

void MarketDataProcessor::clear() {
    history_.clear();
    sma_cache_.clear();
    ema_cache_.clear();
}

void MarketDataProcessor::update_cache() {
    sma_cache_.clear();
    ema_cache_.clear();
}

double MarketDataProcessor::calculate_sma(size_t period) const {
    double sum = 0;
    for (size_t i = 0; i < period; ++i) {
        sum += history_[history_.size() - 1 - i].price;
    }
    double sma = sum / period;
    sma_cache_[period] = sma;
    return sma;
}

double MarketDataProcessor::calculate_ema(size_t period) const {
    double multiplier = 2.0 / (period + 1);
    double ema = history_[history_.size() - period].price;
    
    for (size_t i = history_.size() - period + 1; i < history_.size(); ++i) {
        ema = (history_[i].price - ema) * multiplier + ema;
    }
    
    ema_cache_[period] = ema;
    return ema;
}

} // namespace trading
