#include "trading_engine.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <chrono>

using namespace trading;
using namespace std::chrono_literals;

void test_order_submission() {
    TradingEngine engine;
    engine.start();
    
    Order order;
    order.symbol = "AAPL";
    order.type = OrderType::MARKET;
    order.side = OrderSide::BUY;
    order.quantity = 100;
    order.price = 150.0;
    order.timestamp = std::chrono::system_clock::now();
    
    bool success = engine.submit_order(order);
    assert(success);
    
    // Give some time for order processing
    std::this_thread::sleep_for(100ms);
    
    Position pos = engine.get_position("AAPL");
    assert(pos.quantity == 100);
    assert(std::abs(pos.average_price - 150.0) < 1e-6);
    
    std::cout << "Order submission test passed\n";
    engine.stop();
}

void test_position_limits() {
    TradingEngine engine;
    engine.start();
    
    // Set position limit
    engine.set_position_limit("AAPL", 500);
    
    // Submit order within limit
    Order order1;
    order1.symbol = "AAPL";
    order1.type = OrderType::MARKET;
    order1.side = OrderSide::BUY;
    order1.quantity = 400;
    order1.price = 150.0;
    order1.timestamp = std::chrono::system_clock::now();
    
    bool success1 = engine.submit_order(order1);
    assert(success1);
    
    // Submit order exceeding limit
    Order order2;
    order2.symbol = "AAPL";
    order2.type = OrderType::MARKET;
    order2.side = OrderSide::BUY;
    order2.quantity = 200;
    order2.price = 150.0;
    order2.timestamp = std::chrono::system_clock::now();
    
    bool success2 = engine.submit_order(order2);
    assert(!success2);  // Should fail due to position limit
    
    std::cout << "Position limits test passed\n";
    engine.stop();
}

void test_loss_limits() {
    TradingEngine engine;
    engine.start();
    
    // Set loss limit
    engine.set_loss_limit("AAPL", 1000);
    
    // Submit initial position
    Order order1;
    order1.symbol = "AAPL";
    order1.type = OrderType::MARKET;
    order1.side = OrderSide::BUY;
    order1.quantity = 100;
    order1.price = 150.0;
    order1.timestamp = std::chrono::system_clock::now();
    
    bool success1 = engine.submit_order(order1);
    assert(success1);
    
    // Try to add more when price has dropped significantly
    Order order2;
    order2.symbol = "AAPL";
    order2.type = OrderType::MARKET;
    order2.side = OrderSide::BUY;
    order2.quantity = 100;
    order2.price = 140.0;  // $10 drop = $1000 unrealized loss
    order2.timestamp = std::chrono::system_clock::now();
    
    bool success2 = engine.submit_order(order2);
    assert(!success2);  // Should fail due to loss limit
    
    std::cout << "Loss limits test passed\n";
    engine.stop();
}

void test_ml_signal_processing() {
    TradingEngine engine;
    engine.start();
    
    // Process a buy signal
    engine.process_ml_signal("AAPL", 0.5);  // 50% confidence in upward movement
    
    // Give some time for order processing
    std::this_thread::sleep_for(100ms);
    
    Position pos = engine.get_position("AAPL");
    assert(pos.quantity > 0);  // Should have taken a long position
    
    // Process a sell signal
    engine.process_ml_signal("AAPL", -0.3);  // 30% confidence in downward movement
    
    std::this_thread::sleep_for(100ms);
    
    Position new_pos = engine.get_position("AAPL");
    assert(new_pos.quantity < pos.quantity);  // Should have reduced position
    
    std::cout << "ML signal processing test passed\n";
    engine.stop();
}

int main() {
    try {
        test_order_submission();
        test_position_limits();
        test_loss_limits();
        test_ml_signal_processing();
        std::cout << "All tests passed!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
