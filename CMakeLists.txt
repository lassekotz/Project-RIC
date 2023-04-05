cmake_minimum_required(VERSION 3.10)
project(RIC VERSION 0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(MotorDriver)
add_subdirectory(IMU)
add_subdirectory(Kalman)
include_directories(MotorDriver/)
link_directories(MotorDriver/)

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



add_executable(testMulti runTestsMulti.c)


target_link_libraries(testMulti PUBLIC wiringPi)
target_link_libraries(testMulti PUBLIC motorControl)
target_link_libraries(testMulti PUBLIC pidController)
target_link_libraries(testMulti PUBLIC IMUread)


target_include_directories(testMulti PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MotorDriver"
                          )


target_include_directories(testMulti PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/IMU"
                          )



#Single thread test 
add_executable(testSingle singleTest.c)


target_link_libraries(testSingle PUBLIC wiringPi)
target_link_libraries(testSingle PUBLIC motorControl)
target_link_libraries(testSingle PUBLIC pidController)
target_link_libraries(testSingle PUBLIC IMUread)


target_include_directories(testSingle PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MotorDriver"
                          )


target_include_directories(testSingle PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/IMU"
                          )
                          
#Panic stop 

add_executable(panicStop turnOff.c)
target_link_libraries(panicStop PUBLIC wiringPi)
target_link_libraries(panicStop PUBLIC motorControl)
target_include_directories(panicStop PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MotorDriver"
                          )

add_executable(sysId sysIDtest.c)
target_link_libraries(sysId PUBLIC wiringPi)
target_link_libraries(sysId PUBLIC motorControl)
target_include_directories(sysId PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MotorDriver"
                          )


#CPP program 

add_executable(ricpp cppTest.cpp)
target_link_libraries(ricpp PUBLIC wiringPi)
target_link_libraries(ricpp PUBLIC MPU6050)
target_link_libraries(ricpp PUBLIC motorControl)
target_link_libraries(ricpp PUBLIC pidController)
target_link_libraries(ricpp PUBLIC Kalmanlib)
target_include_directories(ricpp PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MotorDriver"
                          )

target_include_directories(ricpp PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/Kalman"
                          )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(ricpp PRIVATE Threads::Threads)

# Motor follower test

add_executable(motorFollow motorFollow.c)


target_link_libraries(motorFollow PUBLIC wiringPi)
target_link_libraries(motorFollow PUBLIC motorControl)
target_link_libraries(motorFollow PUBLIC pidController)
target_link_libraries(motorFollow PUBLIC m)


target_include_directories(motorFollow PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/MotorDriver"
                          )

