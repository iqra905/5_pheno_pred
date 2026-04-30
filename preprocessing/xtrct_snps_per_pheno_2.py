import gzip
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial

def process_file(filename, input_folder, output_folder, bim_data):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)
    
    rows_read = 0
    rows_written = 0
    
    with gzip.open(input_path, 'rt') as in_file, gzip.open(output_path, 'wt') as out_file:
        for line in in_file:
            rows_read += 1
            fields = line.strip().split()
            chro = int(fields[0])  #  CHR is in the 1st column 
            location = int(fields[2])  # Assuming location is in the 3rd column 
            
            if (chro, location) in bim_data:
                out_file.write(line)  # Write the original line as it is
                rows_written += 1
    
    rows_dropped = rows_read - rows_written
    return filename, rows_read, rows_written, rows_dropped

def main():
    # Define paths
    input_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/pros_can'  # Folder containing .gen.gz files
    output_folder ='/vol/research/ucdatasets/gwas/gwas_mono_rm/pros_can_trimmed'  # Output folder
    bim_file = '/vol/research/ucdatasets/gwas/data_files/Archive/pros01_positions.bim'  # Path to .bim file
    summary_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/pros_can_trimmed/processing_summary.txt'  # Path for the summary output file

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read .bim file
    print("Reading .bim file...")
    bim_df = pd.read_csv(bim_file, sep=r'\s+', engine='python', header=None, 
                         names=['chr', 'snp_id', 'distance', 'location', 'ref', 'alt'])
    print("BIM file read successfully.")
    print(f"Number of SNPs in BIM file: {len(bim_df)}")
    print(f"Number of unique SNPs in BIM file: {bim_df['location'].nunique()}") 
    print(f"Number of unique chromosomes in BIM file: {bim_df['chr'].nunique()}")
    print(bim_df.head())

    # Create a set of (chr, location) tuples from .bim file for faster lookup
    bim_data = set(zip(bim_df['chr'], bim_df['location']))
    print(f"\nNumber of unique (chr, location) tuples in BIM file: {len(bim_data)}")
    print("BIM data created successfully.")
    print(type(bim_data))
    print(len(bim_data))
    #print(bim_data)

    # Get list of .gen.gz files and sort them
    gen_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.gen.gz')],
                       key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Set up multiprocessing
    num_processes = min(cpu_count(), len(gen_files))
    pool = Pool(processes=num_processes)

    # Process files in parallel
    print(f"Processing {len(gen_files)} files using {num_processes} processes...")
    process_func = partial(process_file, input_folder=input_folder, output_folder=output_folder, bim_data=bim_data)
    results = pool.map(process_func, gen_files)

    # Close the pool
    pool.close()
    pool.join()

    # Sort results to match the order of processed files
    results.sort(key=lambda x: int(x[0].split('_')[1].split('.')[0]))

    # Print results
    print("\nProcessing complete. Results:")
    for filename, rows_read, rows_written, rows_dropped in results:
        print(f"{filename}:\n Rows read: {rows_read}: Rows written: {rows_written}: Rows dropped: {rows_dropped}\n")  

    print("\nAll files processed successfully.")


    # Write results to summary file
    with open(summary_file, 'w') as f:
        f.write(f"BIM file: {bim_file}\n")
        f.write(f"Number of files processed: {len(gen_files)}\n\n")
        f.write("File-wise statistics:\n")
        
        for filename, rows_read, rows_written, rows_dropped in results:
            f.write(f"{filename}:\n Rows read: {rows_read}: Rows written: {rows_written}: Rows dropped: {rows_dropped}\n")  
            
    print(f"\nProcessing complete. Summary written to {summary_file}")

if __name__ == '__main__':
    main()

# import gzip
# import os
# import pandas as pd
# from multiprocessing import Pool, cpu_count
# from functools import partial

# def process_file(filename, input_folder, output_folder, bim_locations):
#     input_path = os.path.join(input_folder, filename)
#     output_path = os.path.join(output_folder, filename)
    
#     rows_read = 0
#     rows_written = 0
    
#     with gzip.open(input_path, 'rt') as in_file, gzip.open(output_path, 'wt') as out_file:
#         for line in in_file:
#             rows_read += 1
#             fields = line.strip().split()
#             location = int(fields[2])  # Assuming location is in the 3rd column (0-based index)
            
#             if location in bim_locations:
#                 out_file.write(line)  # Write the original line as it is
#                 rows_written += 1
    
#     rows_dropped = rows_read - rows_written
#     return filename, rows_read, rows_written, rows_dropped

# def main():
#     # Define paths
#     input_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/pros_can'  # Folder containing .gen.gz files
#     output_folder ='/vol/research/ucdatasets/gwas/gwas_mono_rm/pros_can_trimmed'  # Output folder
#     bim_file = '/vol/research/ucdatasets/gwas/data_files/Archive/pros01_positions.bim'  # Path to .bim file
#     summary_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/pros_can_trimmed/processing_summary.txt'  # Path for the summary output file


#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)

#     # Read .bim file
#     print("Reading .bim file...")
#     bim_df = pd.read_csv(bim_file, sep=r'\s+', header=None, names=['chr', 'snp_id', 'distance', 'location', 'ref', 'alt'])

#     # Create a set of locations from .bim file for faster lookup
#     bim_locations = set(bim_df['location'])

#     # Get list of .gen.gz files
#     gen_files = [f for f in os.listdir(input_folder) if f.endswith('.gen.gz')]

#     # Set up multiprocessing
#     num_processes = min(cpu_count(), len(gen_files))
#     pool = Pool(processes=num_processes)

#     # Process files in parallel
#     print(f"Processing {len(gen_files)} files using {num_processes} processes...")
#     process_func = partial(process_file, input_folder=input_folder, output_folder=output_folder, bim_locations=bim_locations)
#     results = pool.map(process_func, gen_files)

#     # Close the pool
#     pool.close()
#     pool.join()

#     # Print results
#     print("\nProcessing complete. Results:")
#     for filename, rows_read, rows_written, rows_dropped in results:
#         print(f"{filename}:")
#         print(f"  Rows read: {rows_read}")
#         print(f"  Rows written: {rows_written}")
#         print(f"  Rows dropped: {rows_dropped}")

#     print("\nAll files processed successfully.")

#      # Write results to summary file
#     with open(summary_file, 'w') as f:
#         f.write(f"BIM file: {bim_file}\n")
#         f.write(f"Number of files processed: {len(gen_files)}\n\n")
#         f.write("File-wise statistics:\n")
        
#         for filename, rows_read, rows_written, rows_dropped in results:
#             f.write(f"\n{filename}:\n Rows read: {rows_read}, Rows written: {rows_written}, Rows dropped: {rows_dropped}\n")

#     print(f"\nProcessing complete. Summary written to {summary_file}")

# if __name__ == '__main__':
#     main()