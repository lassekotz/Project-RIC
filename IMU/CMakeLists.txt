add_library(IMUread call_IMU.c)
target_link_libraries(IMUread m)

add_library(IMUlib MPU6050.cpp)
target_link_libraries(IMUlib m) 
target_link_libraries(IMUlib i2c)
