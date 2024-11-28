cmake_minimum_required(VERSION 3.12)
project(trading_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find Python for creating bindings
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Add include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add library targets
add_library(market_data SHARED
    src/market_data.cpp
)

add_library(trading_engine SHARED
    src/trading_engine.cpp
)

target_link_libraries(trading_engine PRIVATE market_data)

# Add Python bindings
pybind11_add_module(trading_cpp_bindings
    src/bindings.cpp
)

target_link_libraries(trading_cpp_bindings PRIVATE
    market_data
    trading_engine
)

# Add tests
enable_testing()
add_subdirectory(test)
