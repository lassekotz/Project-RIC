cmake_minimum_required(VERSION 3.10)
project(RIC VERSION 0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(MotorDriver)

add_executable(motorTests runTests.c)

target_link_libraries(motorTests PUBLIC motorControl)

target_include_directories(motorTests PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MotorDriver"
                          )