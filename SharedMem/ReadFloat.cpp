#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>

// Define a spin lock
class SpinLock {
private:
    volatile bool _lock;

public:
    SpinLock() : _lock(false) {}

    void acquire() {
        while (__sync_lock_test_and_set(&_lock, true))
            while (_lock)
                asm volatile("pause");
    }

    void release() {
        __sync_lock_release(&_lock);
    }
};

int main() {
    // Open the named shared memory region
    int shared_memory_fd = shm_open("/MySharedMemory", O_RDONLY, 0666);
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

    // Define a spin lock
    SpinLock spin_lock;

    // Continuously read the float value from shared memory
    while (true) {
        // Acquire the spin lock before accessing the shared memory
        spin_lock.acquire();

        // Read the binary data from shared memory and convert it to a float
        float shared_float = *((float*)shared_memory_ptr);

        // Release the spin lock after accessing the shared memory
        spin_lock.release();

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
