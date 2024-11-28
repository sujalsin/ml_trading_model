add_executable(test_market_data
    test_market_data.cpp
)

target_link_libraries(test_market_data
    PRIVATE
    market_data
)

add_executable(test_trading_engine
    test_trading_engine.cpp
)

target_link_libraries(test_trading_engine
    PRIVATE
    trading_engine
)

add_test(NAME test_market_data COMMAND test_market_data)
add_test(NAME test_trading_engine COMMAND test_trading_engine)
