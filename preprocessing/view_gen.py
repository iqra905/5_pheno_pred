import time
import os
import gzip

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])  
    except ValueError:
        chr_num = float('inf')
    
    return (chr_num, file_name)

def read_files(input_folder):
    start_time = time.time()

    files = os.listdir(input_folder)
    gen_files = [f for f in files if f.endswith('.gen.gz')]
    sorted_files = sorted(gen_files, key=custom_sort)
    num_snps = 0
    num_columns = 0
    try:
        for gen_file in sorted_files: 
            print(f"Processing File:\n {gen_file}")
            with gzip.open(os.path.join(input_folder, gen_file), 'rt') as file:
                num_lines = 0
                for line in file:
                    parts = line.split()
                    #print(parts[0:5])
                    num_lines += 1
                    if num_lines % 50000 == 0:
                        print(f"Processed {num_lines} lines in {gen_file}")
                num_columns = len(line.split())
            num_snps += num_lines
            print("Shape of data in", gen_file, ":", num_lines, "lines x", num_columns, "columns")
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    print(f"Total number of SNPs is: \n {num_snps}")
    end_time = time.time() 
    elapsed_time = end_time - start_time
    print("Execution time:", elapsed_time, "seconds")

if __name__ == "__main__":
    #input_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M_filtered/t2d/0.1_pre'
    #input_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_gen_data/pan'
    input_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/gen_data_5M/merged/chr_wise'
    
    read_files(input_folder)

