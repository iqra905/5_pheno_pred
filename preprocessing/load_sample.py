import gzip
import time
import psutil
import os

def measure_file_reading(file_path):
    start_time = time.time()
    
    with open(file_path, 'rt') as infile:
        lines = infile.readlines()
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_used = memory_info.rss  # Resident Set Size: total memory used by the process

    return len(lines), time_taken, memory_used

# Usage
processed_file = '/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/sample_00001_processed_3cols_5Mrows.gen'
num_lines, time_taken, memory_used = measure_file_reading(processed_file)

print(f"Number of lines read: {num_lines}")
print(f"Time taken to read the file: {time_taken} seconds")
print(f"Memory used to read the file: {memory_used / (1024 * 1024)} MB")
