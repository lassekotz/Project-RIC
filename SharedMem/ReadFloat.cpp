#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>



int main() {
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
