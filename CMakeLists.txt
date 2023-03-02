cmake_minimum_required(VERSION 3.10)
project(RIC VERSION 0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(MotorDriver)
add_subdirectory(IMU)

add_executable(firstTests runTests.c)


target_link_libraries(firstTests PUBLIC wiringPi)
target_link_libraries(firstTests PUBLIC motorControl)
target_link_libraries(firstTests PUBLIC pidController)
target_link_libraries(firstTests PUBLIC IMUread)


target_include_directories(firstTests PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MotorDriver"
                          )


target_include_directories(firstTests PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/IMU"
                          )

