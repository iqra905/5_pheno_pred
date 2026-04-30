import os
import shutil
import re

# Define the source and destination directories
source_dir = '/vol/research/ucdatasets/gwas'
destination_dir = '/vol/research/ucdatasets/gwas/error_chunks'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Define the regex pattern for matching the filenames
pattern = re.compile(r'chr\d+_\de\.gen\.gz')

# Iterate through the files in the source directory
for filename in os.listdir(source_dir):
    if pattern.match(filename):
        # Construct full file paths
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)
        
        # Move the file to the destination directory
        shutil.move(source_file, destination_file)
        print(f'Moved: {filename}')

print('File moving completed.')
