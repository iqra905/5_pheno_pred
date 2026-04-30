import os
import shutil
from multiprocessing import Pool, cpu_count

def copy_file(args):
    source_folder, destination_folder, i = args
    filename = f'sample_{i:05d}.gen.gz'
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)
    
    if os.path.exists(source_path):
        shutil.copy2(source_path, destination_path)
        print(f'Copied: {filename}')
    else:
        print(f'File not found: {filename}')

def main():
    # Source and destination folder paths
    # source_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data'
    # destination_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_10'

    source_folder = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data'
    destination_folder = '/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_10'

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Create a list of arguments for each file to be copied
    args_list = [(source_folder, destination_folder, i) for i in range(31501, 35001)]
    print(f"Number of files to copy: {len(args_list)}")

    # Determine the number of processes to use (you can adjust this if needed)
    num_processes = min(cpu_count(), len(args_list))

    # Create a pool of worker processes
    with Pool(num_processes) as pool:
        # Map the copy_file function to the arguments list
        pool.map(copy_file, args_list)

    print('File copying completed.')

if __name__ == '__main__':
    main()

