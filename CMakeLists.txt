
cmake_minimum_required(VERSION 3.18)

project(ArmPyTorchInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

# Basic C++ tensor test
add_executable(torch_cpp_test cpp/main.cpp)
target_link_libraries(torch_cpp_test PRIVATE "${TORCH_LIBRARIES}")

# TorchScript CNN inference
add_executable(torch_cpp_inference cpp/inference.cpp)
target_link_libraries(torch_cpp_inference PRIVATE "${TORCH_LIBRARIES}")

add_executable(torch_cpp_benchmark cpp/benchmark.cpp)

target_link_libraries(
    torch_cpp_benchmark
    PRIVATE "${TORCH_LIBRARIES}"
)
