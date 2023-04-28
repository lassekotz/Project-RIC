import mmap
import struct
import time
import ctypes
import random

# Define the size of the shared memory region
SHARED_MEMORY_SIZE = 4

# Create a named shared memory region
shared_memory = mmap.mmap(-1, SHARED_MEMORY_SIZE, "MySharedMemory")

# Define a ctypes structure to pack the float value in binary format
class FloatStruct(ctypes.Structure):
    _fields_ = [("value", ctypes.c_float)]

# Create a ctypes instance of the FloatStruct
float_struct = FloatStruct()

# Define a spin lock
class SpinLock:
    def __init__(self):
        self._lock = False

    def acquire(self):
        while self._lock:
            pass
        self._lock = True

    def release(self):
        self._lock = False

spin_lock = SpinLock()

# Loop to write the float value to shared memory
while True:
    # Update the value in the ctypes struct
    float_struct.value = random.random()

    # Acquire the spin lock before accessing the shared memory
    spin_lock.acquire()

    # Pack the struct into binary format and write it to the shared memory region
    shared_memory.seek(0)
    shared_memory.write(ctypes.string_at(ctypes.byref(float_struct), ctypes.sizeof(float_struct)))

    # Release the spin lock after accessing the shared memory
    spin_lock.release()

    # Wait for a short time before updating again
    time.sleep(0.1)

