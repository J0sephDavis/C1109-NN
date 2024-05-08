cmake_minimum_required (VERSION 3.18)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) 

project(NeuralNet
  LANGUAGES CXX
  VERSION 0.0.0
)
add_executable(NeuralNet 
	main.cc
)

target_include_directories(NeuralNet PUBLIC "${PROJECT_BINARY_DIR}")

set_target_properties(NeuralNet PROPERTIES CXX_STANDARD 20)
target_compile_options(NeuralNet PUBLIC -Werror -Wextra -Wpedantic -std=c++20)
