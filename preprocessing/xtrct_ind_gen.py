import pandas as pd
import os
import shutil
import gzip
import argparse
from functools import partial
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser(description='Copy sample files to corresponding disease folders.')
parser.add_argument('-source_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq', help='Path to the folder with sample_n.gen.gz files')
parser.add_argument('-base_destination_folder', type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_pruned_disease' , help='Path to folder storing unique SNPs per sample files')
args = parser.parse_args()

# List of Excel files and their corresponding destination folder names
excel_files = [
    {'file': 'brea_can.xlsx', 'folder': 'brea'},
    {'file': 'col_can.xlsx', 'folder': 'col'},
    {'file': 'pan_can.xlsx', 'folder': 'pan'},
    {'file': 'pros_can.xlsx', 'folder': 'pros'},
    {'file': 't2d.xlsx', 'folder': 't2d'}
]

def process_excel_file(excel_info, source_folder, base_destination_folder):
    excel_file = excel_info['file']
    destination_folder = excel_info['folder']
    
    # Read the Excel file
    df = pd.read_excel(os.path.join('/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno_updated', excel_file))
    print(f"Read {excel_file}...\n")

    # Get the list of sample IDs from the 'new_order' column
    sample_ids = df['new_order'].astype(str).tolist()

    # Create the destination folder if it doesn't exist
    full_destination_path = os.path.join(base_destination_folder, destination_folder)
    os.makedirs(full_destination_path, exist_ok=True)

    # Counter for copied files
    copied_files = 0

    # Iterate through the sample IDs
    for sample_id in sample_ids:
        # Construct the filename
        filename = f'sample_{sample_id.zfill(5)}.gen.gz'
        source_path = os.path.join(source_folder, filename)

        #print(f"Source path is: {source_path}\n")
        
        # Check if the file exists in the source folder
        if os.path.exists(source_path):
            # Copy the file to the destination folder
            shutil.copy2(source_path, full_destination_path)
            copied_files += 1
            print(f"Copied: {filename} to {destination_folder}")
        else:
            print(f"File not found: {filename}")

    print(f"\nTotal files copied for {excel_file}: {copied_files}")
    return copied_files

if __name__ == '__main__':
    # Source folder containing .gen.gz files
    source_folder = args.source_folder
    # Base destination folder
    base_destination_folder = args.base_destination_folder

    # Determine the number of processes to use
    num_processes = min(cpu_count(), len(excel_files))

    process_func = partial(process_excel_file, source_folder=source_folder, base_destination_folder=base_destination_folder)
    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Map the process_excel_file function to the excel_files list
        results = pool.map(process_func, excel_files)

    # Calculate and print the total number of files copied
    total_copied = sum(results)
    print(f"\nGrand total of files copied: {total_copied}")

# import pandas as pd
# import os
# import shutil
# import gzip

# # Read the Excel file
# df = pd.read_excel('/vol/research/ucdatasets/gwas/data_files/disease_pheno/pros_can.xlsx')

# # Get the list of sample IDs from the 'new_order' column
# sample_ids = df['new_order'].astype(str).tolist()

# # Source folder containing .gen.gz files
# source_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data'

# # Destination folder where files will be copied
# destination_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/pros_can'

# # Create the destination folder if it doesn't exist
# os.makedirs(destination_folder, exist_ok=True)

# # Counter for copied files
# copied_files = 0

# # Iterate through the sample IDs
# for sample_id in sample_ids:
#     # Construct the filename
#     filename = f'sample_{sample_id.zfill(5)}.gen.gz'
#     source_path = os.path.join(source_folder, filename)
    
#     # Check if the file exists in the source folder
#     if os.path.exists(source_path):
#         # Copy the file to the destination folder
#         shutil.copy2(source_path, destination_folder)
#         copied_files += 1
#         print(f"Copied: {filename}")
#     else:
#         print(f"File not found: {filename}")

# print(f"\nTotal files copied: {copied_files}")