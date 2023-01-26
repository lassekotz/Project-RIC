cmake_minimum_required(VERSION 3.8)
project(IMU LANGUAGES C VERSION 1.0)


add_library(imu_lib SHARED
    IMU.c
)

set_target_properties(imu_lib PROPERTIES VERSION ${PROJECT_VERSION})

find_library(WIRINGPI_LIBRARIES NAMES wiringPi)