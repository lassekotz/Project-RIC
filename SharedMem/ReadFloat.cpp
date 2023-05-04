#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <MPU6050.h>
#include <wiringPi.h>


MPU6050 imu(0x68,0);

void setup(){
    wiringPiSetupGpio(); //Setup and use defult pin numbering
    
    //imu.getAngle(0,&curTheta); //Calculate first value and input to filter 
    //kalman.setAngle(curTheta); 
    

    initMotorPins(); //Initializes pins and hardware interupts for motors
    //setupFirstValue();
}

int main() {
    if( argc != 6 ){
        printf("Wrong number of arguments, you had %d \n",argc);
        exit(2);
    }
    float Kp = (float)atof(argv[1]);
    float Ki = (float)atof(argv[2]);
    float Kd = (float)atof(argv[3]);
    float Kpv = (float)atof(argv[4]);
    float Kiv = (float)atof(argv[5]);
    initRegParam( Kp , Ki, Kd, Kpv, Kiv);
    initMotRegParam( 3000.0, 30.0, 300.0);
    
    // Open the named shared memory region
    int shared_memory_fd = shm_open("currAngle", O_RDONLY, 0666);
    if (shared_memory_fd == -1) {
        std::cerr << "Failed to open shared memory object\n";
        return 1;
    }
    

    // Map the shared memory region into this process's address space
    void* shared_memory_ptr = mmap(NULL, sizeof(float), PROT_READ, MAP_SHARED, shared_memory_fd, 0);
    if (shared_memory_ptr == MAP_FAILED) {
        std::cerr << "Failed to map shared memory object\n";
        close(shared_memory_fd);
        return 1;
    }


    // Continuously read the float value from shared memory
    while (true) {


        // Read the binary data from shared memory and convert it to a float
        float shared_float = *((float*)shared_memory_ptr);

        // CALL ON IMU

        // Print the float value
        std::cout << "Shared float value: " << shared_float << std::endl;

        // Wait for a short time before reading again
        usleep(100000);
    }

    // Unmap the shared memory region and close the file descriptor
    munmap(shared_memory_ptr, sizeof(float));
    close(shared_memory_fd);

    return 0;
}
