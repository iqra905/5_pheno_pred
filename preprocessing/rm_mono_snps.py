import os
import gzip
import multiprocessing
from pathlib import Path
from datetime import datetime

def custom_sort(file_name):
    parts = file_name.split('_')
    try:
        chr_num = int(parts[0][3:])
    except ValueError:
        chr_num = float('inf')
    return (chr_num, file_name)

def process_genotype_file(file_path, output_folder,summary_file):
    base_name = os.path.basename(file_path)
    print(f"Processing file: {base_name}")
    output_path = os.path.join(output_folder, base_name)    
    skipped_count = 0  
    written_count = 0 
    total_lines = 0   

    try:
        with gzip.open(file_path, 'rt') as f_in, gzip.open(output_path, 'at') as f_out:
            for line in f_in:
                total_lines += 1
                columns = line.strip().split()
                num_individuals = (len(columns) - 5) // 3
                genotypes = [columns[5 + i*3: 5 + (i+1)*3] for i in range(num_individuals)]
                
                # Check if at least 5% of the individuals have different genotype info
                first_genotype = genotypes[0]
                different_count = sum(1 for genotype in genotypes[1:] if genotype != first_genotype)
                if different_count / num_individuals <= 0.05:
                    skipped_count += 1
                    continue  
                f_out.write(line)
                written_count += 1  # Increment written count

        with open(summary_file, 'a') as f:
            f.write(f"{base_name}: Total SNPs {total_lines}, Dropped {skipped_count}, Written {written_count}\n")
            f.flush()

        print(f"Processed file: {base_name}")
        print(f"Total SNPs: {total_lines}, Dropped: {skipped_count}, Written: {written_count}")
    
    except FileNotFoundError as e:
        print(f"File not found: {file_path} - {e}")
    except Exception as e:
        print(f"Error processing file: {file_path} - {e}")

def main(input_folder, output_folder, start_file_index, end_file_index, output_file):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    #output_file = output_folder / 'skipped_lines_summary.txt'

    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        print(f"Error: Input folder '{input_folder}' not found.")
        return
    except PermissionError:
        print(f"Error: Permission denied for accessing '{input_folder}'.")
        return

    gen_files = [f for f in files if f.endswith('.gen.gz')]
    sorted_files = sorted(gen_files, key=custom_sort)
    total_files = len(sorted_files)
    print(f"Found {total_files} genotype files.")

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    full_paths = [input_folder / file for file in sorted_files]
    files_to_process = full_paths[start_file_index:end_file_index]

    with open(output_file, 'a') as f:
        #f.write(f"Processing started at: {datetime.now()}\n")
        f.write(f"Processing {len(files_to_process)} files (from file {start_file_index+1} to file {end_file_index})\n")
        f.write(f"File names under process are: {', '.join([os.path.basename(file) for file in files_to_process])}\n\n")

    with multiprocessing.Pool() as pool:
        pool.starmap(process_genotype_file, [(file, output_folder, output_file) for file in files_to_process])

    total_snps = 0
    total_skipped = 0
    total_written = 0

    with open(output_file, 'r') as f:
        for line in f:
            if "Total SNPs" in line:
                parts = line.split(',')
                total_snps += int(parts[0].split()[-1])
                total_skipped += int(parts[1].split()[-1])
                total_written += int(parts[2].split()[-1])

    with open(output_file, 'a') as f:
        f.write(f"\nTotal SNPs across all files: {total_snps}\n")
        f.write(f"Total SNPs skipped: {total_skipped}\n")
        f.write(f"Total SNPs written: {total_written}\n")
        #f.write(f"Processing completed at: {datetime.now()}\n\n")

    print(f"Processing completed. Summary appended to {output_file}")      

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process genotype files.")
    parser.add_argument("-input_folder", type=str, default='/vol/vssp/SF_ucdatasets/gwas', help="Path to the input folder containing .gen.gz files")
    parser.add_argument("-output_folder", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm', help="Path to the output folder to save filtered files")
    parser.add_argument("-output_file", type=str, default='/vol/vssp/SF_ucdatasets/gwas/gwas_mono_rm/skipped_lines_summary_1.txt', help="Path to the output folder to save filtered files")

    # parser.add_argument('-input_folder', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen', help='Path to the folder with .gen.gz files')
    # parser.add_argument('-output_folder', type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen/processed', help="Path to the output folder to save filtered files")
    # parser.add_argument("-output_file", type=str, default='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_gen/processed/skipped_lines_summary_1.txt', help="Path to the output folder to save filtered files")

    parser.add_argument("-start_index", type=int, default=0, help="Index of the first file to process (0-based)")
    parser.add_argument("-end_index", type=int, default=2, help="Index of the last file to process (0-based)")
   

    args = parser.parse_args()
    
    main(args.input_folder, args.output_folder, args.start_index, args.end_index, args.output_file)
   