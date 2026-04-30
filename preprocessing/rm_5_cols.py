import gzip
import os
import sys
import argparse
import multiprocessing as mp
from functools import partial

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extract SNPs from gen.gz files.')
    parser.add_argument('-input_folder', type=str, required=True, help='Path to the folder with sample_n.gen.gz files')
    parser.add_argument('-output_folder', type=str, required=True, help='Output folder')
    parser.add_argument('-slice', type=str, default=':', help='Slice of files to process (e.g., "0:1000", "1000:2000", "2000:")')
    return parser.parse_args()

def process_file(input_path, output_folder):
    try:
        output_path = os.path.join(output_folder, os.path.basename(input_path))
        #Sprint(f"Processing file: {input_path}")
        
        with gzip.open(input_path, 'rt') as in_file, gzip.open(output_path, 'wt') as out_file:
            for line in in_file:
                parts = line.strip().split()
                if len(parts) >= 8:  # Ensure there are at least 8 columns
                    last_three = ' '.join(parts[-3:])
                    out_file.write(f"{last_three}\n")
                else:
                    print(f"Warning: Line with insufficient columns in {input_path}: {line.strip()}")
        
        print(f"Completed processing: {input_path}")
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}", file=sys.stderr)
        # Print more information about the file
        try:
            with gzip.open(input_path, 'rb') as f:
                first_few_lines = [f.readline().decode('utf-8', errors='replace') for _ in range(5)]
            print(f"First few lines of {input_path}:")
            for line in first_few_lines:
                print(line.strip())
            print(f"Number of columns in first line: {len(first_few_lines[0].split())}")
        except Exception as read_error:
            print(f"Unable to read file for debugging: {str(read_error)}", file=sys.stderr)
        return False

def save_first_5_columns(input_folder, output_folder):
    try:
        gen_files = [f for f in os.listdir(input_folder) if f.endswith('.gen.gz')]
        if not gen_files:
            raise FileNotFoundError("No .gen.gz files found in the input folder.")
        
        first_file = min(gen_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        first_file_path = os.path.join(input_folder, first_file)
        
        print(f"Extracting first 5 columns from: {first_file_path}")
        output_path = os.path.join(output_folder, 'first_5_columns_7M.gen.gz')
        
        with gzip.open(first_file_path, 'rt') as in_file, gzip.open(output_path, 'wt') as out_file:
            for line in in_file:
                parts = line.strip().split()
                if len(parts) >= 5:
                    first_five = ' '.join(parts[:5])
                    out_file.write(f"{first_five}\n")
                else:
                    print(f"Warning: Line with insufficient columns in {first_file_path}: {line.strip()}")
        
        print(f"Saved first 5 columns to: {output_path}")
    except Exception as e:
        print(f"Error saving first 5 columns: {str(e)}", file=sys.stderr)
        raise

def process_files(input_folder, output_folder, slice_str):
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Save first 5 columns
        save_first_5_columns(input_folder, output_folder)
        
        # Get list of .gen.gz files and sort them
        gen_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.gen.gz')],
                           key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        if not gen_files:
            raise FileNotFoundError(f"No .gen.gz files found in {input_folder}")
        
        # Parse the slice string
        start, end = map(lambda x: int(x) if x else None, slice_str.split(':'))
        subset_files = gen_files[start:end]
        
        if not subset_files:
            raise ValueError(f"No files to process in the specified slice: {slice_str}")
        
        print(f"Processing files {start or 0} to {end or len(gen_files)}")
        print(f"Number of files to process: {len(subset_files)}")
        
        # Prepare arguments for multiprocessing
        input_paths = [os.path.join(input_folder, f) for f in subset_files]
        
        # Use all available CPU cores
        num_processes = mp.cpu_count()
        
        # Process files using multiprocessing
        with mp.Pool(num_processes) as pool:
            results = pool.map(partial(process_file, output_folder=output_folder), input_paths)
        
        successful = sum(results)
        print(f"Processing complete. Successfully processed {successful} out of {len(subset_files)} files.")
    except Exception as e:
        print(f"Error in process_files: {str(e)}", file=sys.stderr)
        raise

def main():
    args = parse_arguments()
    try:
        process_files(args.input_folder, args.output_folder, args.slice)
    except Exception as e:
        print(f"An error occurred: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

# import gzip
# import os
# import pandas as pd
# import argparse
# import multiprocessing as mp
# from functools import partial
# import sys

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Extract 5M SNPs from 5-disease text file from sample files.')
#     parser.add_argument('-input_folder', type=str, required=True, help='Path to the folder with sample_n.gen.gz files')
#     parser.add_argument('-output_folder', type=str, required=True, help='Output folder')
#     parser.add_argument('-slice', type=str, default=':', help='Slice of files to process (e.g., "0:1000", "1000:2000", "2000:")')
#     return parser.parse_args()

# def save_first_5_columns(input_folder, output_folder):
#     try:
#         gen_files = [f for f in os.listdir(input_folder) if f.endswith('.gen.gz')]
#         if not gen_files:
#             raise FileNotFoundError("No .gen.gz files found in the input folder.")
        
#         first_file = min(gen_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
#         first_file_path = os.path.join(input_folder, first_file)
        
#         print(f"Extracting first 5 columns from: {first_file_path}")
#         with gzip.open(first_file_path, 'rt') as f:
#             df = pd.read_csv(f, sep=r'\s+', header=None, usecols=range(5), 
#                              names=['chr', 'rs', 'pos', 'a1', 'a2'])
        
#         output_path = os.path.join(output_folder, 'first_5_columns_5M_dup.txt')
#         df.to_csv(output_path, sep=' ', header=False, index=False)
#         print(f"Saved first 5 columns to: {output_path}")
#     except Exception as e:
#         print(f"Error saving first 5 columns: {str(e)}", file=sys.stderr)
#         raise

# def process_file(input_path, output_folder):
#     try:
#         output_path = os.path.join(output_folder, os.path.basename(input_path))
#         print(f"Processing file: {input_path}")
        
#         # Read the gzipped file
#         with gzip.open(input_path, 'rt') as f:
#             # Read only the last 3 columns
#             df = pd.read_csv(f, sep=r'\s+', header=None, usecols=[-3, -2, -1], 
#                              names=['prob1', 'prob2', 'prob3'])
        
#         # Check if the dataframe is empty
#         if df.empty:
#             raise ValueError("No data was read from the file")
        
#         # Save the last 3 columns to a new gzipped file
#         with gzip.open(output_path, 'wt') as f:
#             df.to_csv(f, sep=' ', header=False, index=False)
        
#         print(f"Completed processing: {input_path}")
#         return True
#     except Exception as e:
#         print(f"Error processing {input_path}: {str(e)}", file=sys.stderr)
#         # Print more information about the file
#         try:
#             with gzip.open(input_path, 'rt') as f:
#                 first_few_lines = [next(f) for _ in range(5)]
#             print(f"First few lines of {input_path}:")
#             for line in first_few_lines:
#                 print(line.strip())
#         except Exception as read_error:
#             print(f"Unable to read file for debugging: {str(read_error)}", file=sys.stderr)
#         return False

# def process_files(input_folder, output_folder, slice_str):
#     try:
#         # Create output folder if it doesn't exist
#         os.makedirs(output_folder, exist_ok=True)
        
#         # Save first 5 columns
#         save_first_5_columns(input_folder, output_folder)
        
#         # Get list of .gen.gz files and sort them
#         gen_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.gen.gz')],
#                            key=lambda x: int(x.split('_')[1].split('.')[0]))
        
#         if not gen_files:
#             raise FileNotFoundError(f"No .gen.gz files found in {input_folder}")
        
#         # Parse the slice string
#         start, end = map(lambda x: int(x) if x else None, slice_str.split(':'))
#         subset_files = gen_files[start:end]
        
#         if not subset_files:
#             raise ValueError(f"No files to process in the specified slice: {slice_str}")
        
#         print(f"Processing files {start or 0} to {end or len(gen_files)}")
#         print(f"Number of files to process: {len(subset_files)}")
        
#         # Prepare arguments for multiprocessing
#         input_paths = [os.path.join(input_folder, f) for f in subset_files]
        
#         # Use all available CPU cores
#         num_processes = mp.cpu_count()
        
#         # Process files using multiprocessing
#         with mp.Pool(num_processes) as pool:
#             results = pool.map(partial(process_file, output_folder=output_folder), input_paths)
        
#         successful = sum(results)
#         print(f"Processing complete. Successfully processed {successful} out of {len(subset_files)} files.")
#     except Exception as e:
#         print(f"Error in process_files: {str(e)}", file=sys.stderr)
#         raise

# if __name__ == '__main__':
#     args = parse_arguments()
#     try:
#         process_files(args.input_folder, args.output_folder, args.slice)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}", file=sys.stderr)
#         sys.exit(1)

# import gzip
# import os
# import pandas as pd
# import argparse
# import multiprocessing as mp
# from functools import partial
# import sys

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Extract 5M SNPs from 5-disease text file from sample files.')
#     parser.add_argument('-input_folder', type=str, required=True, help='Path to the folder with sample_n.gen.gz files')
#     parser.add_argument('-output_folder', type=str, required=True, help='Output folder')
#     parser.add_argument('-slice', type=str, default=':', help='Slice of files to process (e.g., "0:1000", "1000:2000", "2000:")')
#     return parser.parse_args()
# def process_file(input_path, output_folder):
#     try:
#         output_path = os.path.join(output_folder, os.path.basename(input_path))
#         print(f"Processing file: {input_path}")
        
#         # Read the gzipped file
#         with gzip.open(input_path, 'rt') as f:
#             # Read the first line to determine the number of columns
#             first_line = f.readline().strip()
#             num_columns = len(first_line.split())
            
#             if num_columns < 3:
#                 raise ValueError(f"File has only {num_columns} columns, expected at least 3")
            
#             # Reset file pointer to the beginning
#             f.seek(0)
            
#             # Read only the last 3 columns
#             df = pd.read_csv(f, sep=r'\s+', header=None, usecols=[-3, -2, -1], 
#                              names=['col1', 'col2', 'col3'])  # Add names to avoid warnings
        
#         # Check if the dataframe is empty
#         if df.empty:
#             raise ValueError("No data was read from the file")
        
#         # Save the last 3 columns to a new gzipped file
#         with gzip.open(output_path, 'wt') as f:
#             df.to_csv(f, sep=' ', header=False, index=False)
        
#         print(f"Completed processing: {input_path}")
#         return True
#     except Exception as e:
#         print(f"Error processing {input_path}: {str(e)}", file=sys.stderr)
#         # Print more information about the file
#         try:
#             with gzip.open(input_path, 'rt') as f:
#                 first_few_lines = [next(f) for _ in range(5)]
#             print(f"First few lines of {input_path}:")
#             for line in first_few_lines:
#                 print(line.strip())
#         except Exception as read_error:
#             print(f"Unable to read file for debugging: {str(read_error)}", file=sys.stderr)
#         return False
# # def process_file(input_path, output_folder):
# #     try:
# #         output_path = os.path.join(output_folder, os.path.basename(input_path))
# #         print(f"Processing file: {input_path}")
        
# #         # Read the gzipped file
# #         with gzip.open(input_path, 'rt') as f:
# #             # Read only the last 3 columns
# #             df = pd.read_csv(f, sep=r'\s+', header=None)
# #             print(df)
# #             df1 = pd.read_csv(f, sep=r'\s+', header=None, usecols=[-3, -2, -1])
# #             print(df1)
        
# #         # Save the last 3 columns to a new gzipped file
# #         with gzip.open(output_path, 'wt') as f:
# #             df.to_csv(f, sep=' ', header=False, index=False)
        
# #         print(f"Completed processing: {input_path}")
# #         return True
# #     except Exception as e:
# #         print(f"Error processing {input_path}: {str(e)}", file=sys.stderr)
# #         return False

# def save_first_5_columns(input_folder, output_folder):
#     try:
#         gen_files = [f for f in os.listdir(input_folder) if f.endswith('.gen.gz')]
#         if not gen_files:
#             raise FileNotFoundError("No .gen.gz files found in the input folder.")
        
#         first_file = min(gen_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
#         first_file_path = os.path.join(input_folder, first_file)
        
#         print(f"Extracting first 5 columns from: {first_file_path}")
#         with gzip.open(first_file_path, 'rt') as f:
#             df = pd.read_csv(f, sep=r'\s+', header=None, usecols=range(5))
        
#         output_path = os.path.join(output_folder, 'first_5_columns_5M_dup.txt')
#         df.to_csv(output_path, sep=' ', header=False, index=False)
#         print(f"Saved first 5 columns to: {output_path}")
#     except Exception as e:
#         print(f"Error saving first 5 columns: {str(e)}", file=sys.stderr)
#         raise

# # def process_files(input_folder, output_folder, slice_str):
# #     try:
# #         # Create output folder if it doesn't exist
# #         os.makedirs(output_folder, exist_ok=True)
        
# #         # Save first 5 columns
# #         save_first_5_columns(input_folder, output_folder)
        
# #         # Get list of .gen.gz files and sort them
# #         gen_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.gen.gz')],
# #                            key=lambda x: int(x.split('_')[1].split('.')[0]))
        
# #         # Parse the slice string
# #         start, end = map(lambda x: int(x) if x else None, slice_str.split(':'))
# #         subset_files = gen_files[start:end]
        
# #         print(f"Processing files {start or 0} to {end or len(gen_files)}")
# #         print(f"Number of files to process: {len(subset_files)}")
        
# #         # Prepare arguments for multiprocessing
# #         input_paths = [os.path.join(input_folder, f) for f in subset_files]
        
# #         # Use all available CPU cores
# #         num_processes = mp.cpu_count()
        
# #         # Process files using multiprocessing
# #         with mp.Pool(num_processes) as pool:
# #             results = pool.map(partial(process_file, output_folder=output_folder), input_paths)
        
# #         successful = sum(results)
# #         print(f"Processing complete. Successfully processed {successful} out of {len(subset_files)} files.")
# #     except Exception as e:
# #         print(f"Error in process_files: {str(e)}", file=sys.stderr)
# #         raise
# def process_files(input_folder, output_folder, slice_str):
#     try:
#         # Create output folder if it doesn't exist
#         os.makedirs(output_folder, exist_ok=True)
        
#         # Save first 5 columns
#         save_first_5_columns(input_folder, output_folder)
        
#         # Get list of .gen.gz files and sort them
#         gen_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.gen.gz')],
#                            key=lambda x: int(x.split('_')[1].split('.')[0]))
        
#         if not gen_files:
#             raise FileNotFoundError(f"No .gen.gz files found in {input_folder}")
        
#         # Parse the slice string
#         start, end = map(lambda x: int(x) if x else None, slice_str.split(':'))
#         subset_files = gen_files[start:end]
        
#         if not subset_files:
#             raise ValueError(f"No files to process in the specified slice: {slice_str}")
        
#         print(f"Processing files {start or 0} to {end or len(gen_files)}")
#         print(f"Number of files to process: {len(subset_files)}")
        
#         # Prepare arguments for multiprocessing
#         input_paths = [os.path.join(input_folder, f) for f in subset_files]
        
#         # Use all available CPU cores
#         num_processes = mp.cpu_count()
        
#         # Process files using multiprocessing
#         with mp.Pool(num_processes) as pool:
#             results = pool.map(partial(process_file, output_folder=output_folder), input_paths)
        
#         successful = sum(results)
#         print(f"Processing complete. Successfully processed {successful} out of {len(subset_files)} files.")
#     except Exception as e:
#         print(f"Error in process_files: {str(e)}", file=sys.stderr)
#         raise
# if __name__ == '__main__':
#     args = parse_arguments()
#     try:
#         process_files(args.input_folder, args.output_folder, args.slice)
#     except Exception as e:
#         print(f"An error occurred: {str(e)}", file=sys.stderr)
#         sys.exit(1)

###################################################################
        
# import gzip
# import os
# import pandas as pd
# from tqdm import tqdm
# import argparse
# import multiprocessing as mp


# parser = argparse.ArgumentParser(description='Extract 5M SNPs from 5-disease text file from sample files.')
# parser.add_argument('-input_folder', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_dup', help='Path to the folder with sample_n.gen.gz files')
# parser.add_argument('-output_folder', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_dup_3_cols', help='Output folder')
# parser.add_argument('-slice', type=str, default='0:1000', help='Slice of files to process (e.g., "0:10000", "10000:20000", "20000:")')
# args = parser.parse_args()

# def process_file(args):
#     input_path, output_path = args
    
#     # Read the gzipped file
#     print(f"Reading input file: {input_path}\n")
#     with gzip.open(input_path, 'rt') as f:
#         df = pd.read_csv(f, sep=r'\s+', header=None)
    
#     # Extract the last 3 columns
#     last_3 = df.iloc[:, -3:]
    
#     # Save the last 3 columns to a new gzipped file
#     with gzip.open(output_path, 'wt') as f:
#         last_3.to_csv(f, sep=' ', header=False, index=False)

# def process_files(input_folder, output_folder, slice_str):
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     # Get list of .gen.gz files and sort them
#     gen_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.gen.gz')],
#                        key=lambda x: int(x.split('_')[1].split('.')[0]))
    
#         # Store the first 5 columns from the first file
#     first_file = gen_files[0]
#     with gzip.open(os.path.join(input_folder, first_file), 'rt') as f:
#         df = pd.read_csv(f, sep=r'\s+', header=None)
#     first_5_columns = df.iloc[:, :5]
#     first_5_columns.to_csv(os.path.join(output_folder, 'first_5_columns_5M_dup.txt'), sep=' ', header=False, index=False)
    
#     # Parse the slice string
#     slice_parts = slice_str.split(':')
#     start = int(slice_parts[0]) if slice_parts[0] else None
#     end = int(slice_parts[1]) if len(slice_parts) > 1 and slice_parts[1] else None

#     # Select subset of files based on the slice
#     subset_files = gen_files[start:end]

#     print(f"Processing files {start or 0} to {end or 'end'}")
#     print(f"Number of files to process: {len(subset_files)}")

#     # Prepare arguments for multiprocessing
#     args = [(os.path.join(input_folder, f), os.path.join(output_folder, f)) for f in subset_files]
    
#     # Use all available CPU cores
#     num_processes = mp.cpu_count()
    
#     # Process files using multiprocessing
#     with mp.Pool(num_processes) as pool:
#         pool.map(process_file, args)
#     print("Processing complete.")

# if __name__ == '__main__':
#     # Usage
#     input_folder = args.input_folder
#     output_folder = args.output_folder
#     slice_str = args.slice

#     process_files(input_folder, output_folder, slice_str)