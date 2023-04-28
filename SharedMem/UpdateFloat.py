import mmap
import struct
import time
import ctypes
import random
import os
from multiprocessing import shared_memory

# Define the size of the shared memory region
SHARED_MEMORY_SIZE = 4


shared = shared_memory.SharedMemory("currAngle",create=True, size=SHARED_MEMORY_SIZE)

# Define a ctypes structure to pack the float value in binary format
class FloatStruct(ctypes.Structure):
    _fields_ = [("value", ctypes.c_float)]
    
float_struct = FloatStruct()
i = 0

while True:
    float_struct.value = i
    
    shared.buf[:4] = ctypes.string_at(ctypes.byref(float_struct), ctypes.sizeof(float_struct))
    
    i += 1
    if(i%10 == 0):
        print(i)
        
    time.sleep(1)