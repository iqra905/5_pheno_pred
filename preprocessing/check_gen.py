import time
import os
import gzip
import re

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])  
    except ValueError:
        chr_num = float('inf')
    
    return (chr_num, file_name)

def check_file_content(input_folder, output_file):
    start_time = time.time()

    files = os.listdir(input_folder)
    gen_files = [f for f in files if f.endswith('.gen.gz')]
    sorted_files = sorted(gen_files, key=custom_sort)
    
    # Regular expression pattern
    pattern = re.compile(r'^(\S+\s+){4}\S+(\s+(-?\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)*$')
    
    try:
        with open(output_file, 'w') as out_file:
            for gen_file in sorted_files: 
                print(f"\nProcessing File: {gen_file}")
                unexpected_lines_count = 0
                original_lines_count = 0
                updated_lines_count = 0
                
                with gzip.open(os.path.join(input_folder, gen_file), 'rt') as file, \
                     gzip.open(os.path.join(input_folder, f"updated_{gen_file}"), 'wt') as updated_file:
                    
                    for line_num, line in enumerate(file, 1):
                        original_lines_count += 1
                        if not pattern.match(line.strip()):
                            unexpected_lines_count += 1
                            print(f"Unexpected character found in line {line_num}:")
                            print(line.strip())
                            columns = line.strip().split()
                            if len(columns) > 5:
                                print("First 5 columns (should be strings):", columns[:5])
                                print("First 5 numeric columns:", columns[5:10])
                            
                            # Write the unexpected line to the output file
                            out_file.write(f"File: {gen_file}, Line: {line_num}\n")
                            out_file.write(line)
                            out_file.write("\n\n")
                            
                            # Remove unexpected characters and write to updated file
                            updated_line = re.sub(r'[^\S\t\n\r]+', ' ', line).strip() + '\n'
                            updated_file.write(updated_line)
                            updated_lines_count += 1
                        else:
                            updated_file.write(line)
                            updated_lines_count += 1
                
                if unexpected_lines_count == 0:
                    print("No unexpected characters found.")
                else:
                    print(f"Found {unexpected_lines_count} lines with unexpected characters.")
                
                print(f"Shape of original data in {gen_file}: {original_lines_count} lines")
                print(f"Shape of updated data in updated_{gen_file}: {updated_lines_count} lines")
    
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print(f"\nExecution time: {elapsed_time:.2f} seconds")
    print(f"Unexpected lines have been saved to {output_file}")

if __name__ == "__main__":
    input_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/error_chunks/updated'
    output_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/error_chunks/updated/unexpected_lines_all_18_1a.txt'
    check_file_content(input_folder, output_file)

# import time
# import os
# import gzip
# import re

# def custom_sort(file_name):
#     parts = file_name.split('_')
#     try:
#         chr_num = int(parts[0][3:])  
#     except ValueError:
#         chr_num = float('inf')
    
#     return (chr_num, file_name)

# def check_file_content(input_folder, output_file):
#     start_time = time.time()

#     files = os.listdir(input_folder)
#     gen_files = [f for f in files if f.endswith('.gen.gz')]
#     sorted_files = sorted(gen_files, key=custom_sort)
    
#     # Regular expression pattern
#     # First 5 columns: any non-space characters
#     # Remaining columns: float numbers (allowing scientific notation)
#     pattern = re.compile(r'^(\S+\s+){4}\S+(\s+(-?\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)*$')
    
#     try:
#         with open(output_file, 'w') as out_file:
#             for gen_file in sorted_files: 
#                 print(f"\nProcessing File: {gen_file}")
#                 with gzip.open(os.path.join(input_folder, gen_file), 'rt') as file:
#                     unexpected_chars = False
#                     for line_num, line in enumerate(file, 1):
#                         if not pattern.match(line.strip()):
#                             print(f"Unexpected character found in line {line_num}:")
#                             print(line.strip())
#                             columns = line.strip().split()
#                             if len(columns) > 5:
#                                 print("First 5 columns (should be strings):", columns[:5])
#                                 print("First 5 numeric columns:", columns[5:10])
#                             unexpected_chars = True
                            
#                             # Write the unexpected line to the output file
#                             out_file.write(f"File: {gen_file}, Line: {line_num}\n")
#                             out_file.write(line)
#                             out_file.write("\n\n")
                            
#                             break  # Stop after finding the first unexpected line
                    
#                     if not unexpected_chars:
#                         print("No unexpected characters found.")
                    
#                     # Move back to the start of the file to count lines and columns
#                     file.seek(0)
#                     num_lines = sum(1 for _ in file)
                    
#                     # Move back to the start again to check the first line for column count
#                     file.seek(0)
#                     first_line = next(file)
#                     num_columns = len(first_line.split())
                    
#                 print(f"Shape of data in {gen_file}: {num_lines} lines x {num_columns} columns")
    
#     except FileNotFoundError:
#         print("File not found. Please provide a valid file path.")
    
#     end_time = time.time() 
#     elapsed_time = end_time - start_time
#     print(f"\nExecution time: {elapsed_time} seconds")
#     print(f"Unexpected lines have been saved to {output_file}")

# if __name__ == "__main__":
#     input_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/error_chunks'
#     output_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/error_chunks/unexpected_lines_18_1a.txt'
#     check_file_content(input_folder, output_file)


# import time
# import os
# import gzip
# import re

# def custom_sort(file_name):
#     parts = file_name.split('_')
#     try:
#         chr_num = int(parts[0][3:])  
#     except ValueError:
#         chr_num = float('inf')
    
#     return (chr_num, file_name)

# def check_file_content(input_folder):
#     start_time = time.time()

#     files = os.listdir(input_folder)
#     gen_files = [f for f in files if f.endswith('.gen.gz')]
#     sorted_files = sorted(gen_files, key=custom_sort)
    
#     # Regular expression pattern
#     # First 5 columns: any non-space characters
#     # Remaining columns: float numbers (allowing scientific notation)
#     pattern = re.compile(r'^(\S+\s+){4}\S+(\s+(-?\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)*$')
    
#     try:
#         for gen_file in sorted_files: 
#             print(f"\nProcessing File: {gen_file}")
#             with gzip.open(os.path.join(input_folder, gen_file), 'rt') as file:
#                 unexpected_chars = False
#                 for line_num, line in enumerate(file, 1):
#                     if not pattern.match(line.strip()):
#                         print(f"Unexpected character found in line {line_num}:")
#                         print(line.strip())
#                         columns = line.strip().split()
#                         if len(columns) > 5:
#                             print("First 5 columns (should be strings):", columns[:5])
#                             print("First 5 numeric columns:", columns[5:10])
#                         unexpected_chars = True
#                         break  # Stop after finding the first unexpected line
                
#                 if not unexpected_chars:
#                     print("No unexpected characters found.")
                
#                 # Move back to the start of the file to count lines and columns
#                 file.seek(0)
#                 num_lines = sum(1 for _ in file)
                
#                 # Move back to the start again to check the first line for column count
#                 file.seek(0)
#                 first_line = next(file)
#                 num_columns = len(first_line.split())
                
#             print(f"Shape of data in {gen_file}: {num_lines} lines x {num_columns} columns")
    
#     except FileNotFoundError:
#         print("File not found. Please provide a valid file path.")
    
#     end_time = time.time() 
#     elapsed_time = end_time - start_time
#     print(f"\nExecution time: {elapsed_time} seconds")

# if __name__ == "__main__":
#     input_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/error_chunks'
#     check_file_content(input_folder)