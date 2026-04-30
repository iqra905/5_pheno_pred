import os
import shutil
from multiprocessing import Pool, cpu_count

def move_file(args):
    source_folder, destination_folder, i = args
    filename = f'sample_{i:05d}.gen.gz'
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)
    
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        if os.path.exists(destination_path):
            return f'Moved and overwrote: {filename}'
        else:
            return f'Moved: {filename}'
    else:
        return f'File not found: {filename}'

def main():
    # Source and destination folder paths
    source_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_6_processed'
    destination_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data'

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Create a list of arguments for each file to be moved
    args_list = [(source_folder, destination_folder, i) for i in range(17501, 21001)]

    # Determine the number of processes to use (you can adjust this if needed)
    num_processes = min(cpu_count(), len(args_list))

    # Create a pool of worker processes
    with Pool(num_processes) as pool:
        # Map the move_file function to the arguments list
        results = pool.map(move_file, args_list)

    # Print the results
    for result in results:
        print(result)

    print('File moving completed.')

if __name__ == '__main__':
    main()