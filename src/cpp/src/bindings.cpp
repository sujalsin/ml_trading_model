#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include "market_data.hpp"
#include "trading_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(trading_cpp_bindings, m) {
    m.doc() = "Python bindings for C++ trading components";
    
    // Bind MarketData struct
    py::class_<trading::MarketData>(m, "MarketData")
        .def(py::init<>())
        .def_readwrite("symbol", &trading::MarketData::symbol)
        .def_readwrite("price", &trading::MarketData::price)
        .def_readwrite("volume", &trading::MarketData::volume)
        .def_readwrite("high", &trading::MarketData::high)
        .def_readwrite("low", &trading::MarketData::low)
        .def_readwrite("open", &trading::MarketData::open)
        .def_readwrite("timestamp", &trading::MarketData::timestamp);
    
    // Bind MarketDataProcessor class
    py::class_<trading::MarketDataProcessor>(m, "MarketDataProcessor")
        .def(py::init<size_t>(), py::arg("window_size") = 30)
        .def("update", &trading::MarketDataProcessor::update)
        .def("get_sma", &trading::MarketDataProcessor::get_sma)
        .def("get_ema", &trading::MarketDataProcessor::get_ema)
        .def("get_rsi", &trading::MarketDataProcessor::get_rsi)
        .def("get_macd", &trading::MarketDataProcessor::get_macd)
        .def("get_bollinger_bands", &trading::MarketDataProcessor::get_bollinger_bands)
        .def("get_history", &trading::MarketDataProcessor::get_history)
        .def("clear", &trading::MarketDataProcessor::clear);
    
    // Bind OrderType enum
    py::enum_<trading::OrderType>(m, "OrderType")
        .value("MARKET", trading::OrderType::MARKET)
        .value("LIMIT", trading::OrderType::LIMIT)
        .value("STOP", trading::OrderType::STOP)
        .value("STOP_LIMIT", trading::OrderType::STOP_LIMIT)
        .export_values();
    
    // Bind OrderSide enum
    py::enum_<trading::OrderSide>(m, "OrderSide")
        .value("BUY", trading::OrderSide::BUY)
        .value("SELL", trading::OrderSide::SELL)
        .export_values();
    
    // Bind Order struct
    py::class_<trading::Order>(m, "Order")
        .def(py::init<>())
        .def_readwrite("symbol", &trading::Order::symbol)
        .def_readwrite("type", &trading::Order::type)
        .def_readwrite("side", &trading::Order::side)
        .def_readwrite("quantity", &trading::Order::quantity)
        .def_readwrite("price", &trading::Order::price)
        .def_readwrite("stop_price", &trading::Order::stop_price)
        .def_readwrite("timestamp", &trading::Order::timestamp);
    
    // Bind Position struct
    py::class_<trading::Position>(m, "Position")
        .def(py::init<>())
        .def_readwrite("symbol", &trading::Position::symbol)
        .def_readwrite("quantity", &trading::Position::quantity)
        .def_readwrite("average_price", &trading::Position::average_price)
        .def_readwrite("timestamp", &trading::Position::timestamp);
    
    // Bind TradingEngine class
    py::class_<trading::TradingEngine>(m, "TradingEngine")
        .def(py::init<>())
        .def("start", &trading::TradingEngine::start)
        .def("stop", &trading::TradingEngine::stop)
        .def("submit_order", &trading::TradingEngine::submit_order)
        .def("cancel_order", &trading::TradingEngine::cancel_order)
        .def("get_position", &trading::TradingEngine::get_position)
        .def("get_all_positions", &trading::TradingEngine::get_all_positions)
        .def("on_market_data", &trading::TradingEngine::on_market_data)
        .def("set_position_limit", &trading::TradingEngine::set_position_limit)
        .def("set_loss_limit", &trading::TradingEngine::set_loss_limit)
        .def("process_ml_signal", &trading::TradingEngine::process_ml_signal);
}
