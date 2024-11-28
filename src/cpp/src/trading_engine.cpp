#include "trading_engine.hpp"
#include <stdexcept>
#include <algorithm>

namespace trading {

TradingEngine::TradingEngine() : running_(false) {}

TradingEngine::~TradingEngine() {
    stop();
}

void TradingEngine::start() {
    if (running_) return;
    
    running_ = true;
    processing_thread_ = std::thread([this]() {
        process_orders();
    });
}

void TradingEngine::stop() {
    if (!running_) return;
    
    running_ = false;
    cv_.notify_all();
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

bool TradingEngine::submit_order(const Order& order) {
    if (!running_) {
        throw std::runtime_error("Trading engine is not running");
    }
    
    if (!validate_order(order)) {
        return false;
    }
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        order_queue_.push(order);
    }
    
    cv_.notify_one();
    return true;
}

bool TradingEngine::cancel_order(const std::string& order_id) {
    // In a real implementation, we would maintain an order book and cancel specific orders
    return true;
}

Position TradingEngine::get_position(const std::string& symbol) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = positions_.find(symbol);
    if (it != positions_.end()) {
        return it->second;
    }
    
    Position empty_position;
    empty_position.symbol = symbol;
    empty_position.quantity = 0;
    empty_position.average_price = 0;
    empty_position.timestamp = std::chrono::system_clock::now();
    return empty_position;
}

std::vector<Position> TradingEngine::get_all_positions() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Position> positions;
    positions.reserve(positions_.size());
    
    for (const auto& pair : positions_) {
        positions.push_back(pair.second);
    }
    
    return positions;
}

void TradingEngine::on_market_data(const MarketData& data) {
    market_data_processor_.update(data);
}

void TradingEngine::set_position_limit(const std::string& symbol, double limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    position_limits_[symbol] = limit;
}

void TradingEngine::set_loss_limit(const std::string& symbol, double limit) {
    std::lock_guard<std::mutex> lock(mutex_);
    loss_limits_[symbol] = limit;
}

void TradingEngine::process_ml_signal(const std::string& symbol, double signal) {
    // Create an order based on the ML signal
    Order order;
    order.symbol = symbol;
    order.type = OrderType::MARKET;
    order.side = signal > 0 ? OrderSide::BUY : OrderSide::SELL;
    order.quantity = std::abs(signal) * 100;  // Scale signal to number of shares
    order.timestamp = std::chrono::system_clock::now();
    
    submit_order(order);
}

void TradingEngine::process_orders() {
    while (running_) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() {
            return !running_ || !order_queue_.empty();
        });
        
        if (!running_) break;
        
        while (!order_queue_.empty()) {
            Order order = order_queue_.front();
            order_queue_.pop();
            
            // Process the order
            if (check_risk_limits(order)) {
                update_position(order.symbol, 
                              order.side == OrderSide::BUY ? order.quantity : -order.quantity,
                              order.price);
                
                // Notify callbacks
                for (const auto& callback : order_callbacks_) {
                    callback(order);
                }
            }
        }
    }
}

bool TradingEngine::validate_order(const Order& order) const {
    if (order.symbol.empty()) return false;
    if (order.quantity <= 0) return false;
    if (order.type == OrderType::LIMIT && order.price <= 0) return false;
    if ((order.type == OrderType::STOP || order.type == OrderType::STOP_LIMIT) && 
        order.stop_price <= 0) return false;
    
    return true;
}

void TradingEngine::update_position(const std::string& symbol, double quantity, double price) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& position = positions_[symbol];
    position.symbol = symbol;
    
    if (position.quantity == 0) {
        position.average_price = price;
    } else {
        double total_value = position.quantity * position.average_price + quantity * price;
        position.average_price = total_value / (position.quantity + quantity);
    }
    
    position.quantity += quantity;
    position.timestamp = std::chrono::system_clock::now();
}

bool TradingEngine::check_risk_limits(const Order& order) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Check position limits
    auto pos_limit_it = position_limits_.find(order.symbol);
    if (pos_limit_it != position_limits_.end()) {
        auto current_pos = positions_.find(order.symbol);
        double projected_position = (current_pos != positions_.end() ? current_pos->second.quantity : 0)
                                  + (order.side == OrderSide::BUY ? order.quantity : -order.quantity);
        
        if (std::abs(projected_position) > pos_limit_it->second) {
            return false;
        }
    }
    
    // Check loss limits
    auto loss_limit_it = loss_limits_.find(order.symbol);
    if (loss_limit_it != loss_limits_.end()) {
        auto current_pos = positions_.find(order.symbol);
        if (current_pos != positions_.end()) {
            double unrealized_loss = (current_pos->second.average_price - order.price) 
                                   * current_pos->second.quantity;
            if (unrealized_loss > loss_limit_it->second) {
                return false;
            }
        }
    }
    
    return true;
}

} // namespace trading
