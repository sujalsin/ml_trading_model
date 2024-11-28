#pragma once

#include "market_data.hpp"
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <functional>

namespace trading {

enum class OrderType {
    MARKET,
    LIMIT,
    STOP,
    STOP_LIMIT
};

enum class OrderSide {
    BUY,
    SELL
};

struct Order {
    std::string symbol;
    OrderType type;
    OrderSide side;
    double quantity;
    double price;
    double stop_price;  // For stop and stop-limit orders
    std::chrono::system_clock::time_point timestamp;
};

struct Position {
    std::string symbol;
    double quantity;
    double average_price;
    std::chrono::system_clock::time_point timestamp;
};

class TradingEngine {
public:
    TradingEngine();
    ~TradingEngine();

    // Start/stop the trading engine
    void start();
    void stop();

    // Order management
    bool submit_order(const Order& order);
    bool cancel_order(const std::string& order_id);
    
    // Position management
    Position get_position(const std::string& symbol) const;
    std::vector<Position> get_all_positions() const;
    
    // Market data handling
    void on_market_data(const MarketData& data);
    
    // Risk management
    void set_position_limit(const std::string& symbol, double limit);
    void set_loss_limit(const std::string& symbol, double limit);
    
    // Callback registration
    using OrderCallback = std::function<void(const Order&)>;
    void register_order_callback(OrderCallback callback);
    
    // Trading signals from ML model
    void process_ml_signal(const std::string& symbol, double signal);

private:
    // Internal state
    std::atomic<bool> running_;
    std::thread processing_thread_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
    // Order and position tracking
    std::queue<Order> order_queue_;
    std::unordered_map<std::string, Position> positions_;
    std::unordered_map<std::string, double> position_limits_;
    std::unordered_map<std::string, double> loss_limits_;
    
    // Market data processor
    MarketDataProcessor market_data_processor_;
    
    // Callbacks
    std::vector<OrderCallback> order_callbacks_;
    
    // Internal processing
    void process_orders();
    bool validate_order(const Order& order) const;
    void update_position(const std::string& symbol, double quantity, double price);
    bool check_risk_limits(const Order& order) const;
};

} // namespace trading
