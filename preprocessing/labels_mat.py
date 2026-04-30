#**************** 1, -1 mapping of controls and cases ***************#
import os
import numpy as np
import pandas as pd
from scipy import io
import re
import glob
import argparse

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def create_labels_mat(excel_file, sample_folder, output_file, label_column):
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
    # Extract sample IDs and labels
    sample_ids = df['new_order'].tolist()
    labels = df[label_column].tolist()
    
    # Convert 0 to -1 in labels
    labels = [-1 if label == 0 else label for label in labels]
    
    # Create a dictionary mapping sample IDs to labels
    label_dict = dict(zip(sample_ids, labels))
    
    # Get the list of sample files in the correct order
    sample_files = sorted(glob.glob(os.path.join(sample_folder, 'sample_*.mat')), key=natural_sort_key)
    #print(sample_files)
    
    # Extract sample numbers from filenames
    sample_numbers = [int(re.search(r'sample_(\d+)', os.path.basename(f)).group(1)) for f in sample_files]
    
    # Create the labels array in the correct order
    ordered_labels = np.array([label_dict.get(sample_id, np.nan) for sample_id in sample_numbers])
    
    # Save the labels to a .mat file
    io.savemat(output_file, {'labels': ordered_labels})
    
    print(f"Labels saved to {output_file}")
    print(f"Number of labels: {len(ordered_labels)}")
    print(f"First few labels: {ordered_labels[:5]}")
    print(f"Last few labels: {ordered_labels[-5:]}")
    print(f"Number of -1 labels (control): {np.sum(ordered_labels == -1)}")
    print(f"Number of 1 labels (case): {np.sum(ordered_labels == 1)}")

def main():
    parser = argparse.ArgumentParser(description='Create labels matrix for GWAS analysis')
    parser.add_argument('-excel_file', required=True, help='Path to the Excel file containing phenotype data')
    parser.add_argument('-sample_folder', required=True, help='Path to the folder containing sample .mat files')
    parser.add_argument('-output_file', required=True, help='Path for the output labels .mat file')
    parser.add_argument('-label_column', default='t2dm', help='Column name in Excel file containing labels (default: t2dm)')

    args = parser.parse_args()

     # Validate inputs
    if not os.path.exists(args.excel_file):
        print(f"Error: Excel file not found: {args.excel_file}")
        return
    
    if not os.path.exists(args.sample_folder):
        print(f"Error: Sample folder not found: {args.sample_folder}")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    create_labels_mat(args.excel_file, args.sample_folder, args.output_file, args.label_column)

if __name__ == "__main__":
    main()

    
#**************** 0, 1 mapping of controls and cases ***************#
# import os
# import numpy as np
# import breadas as pd
# from scipy import io
# import re
# import glob

# def natural_sort_key(s):
#     return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

# def create_labels_mat(excel_file, sample_folder, output_file):
#     # Read the Excel file
#     df = pd.read_excel(excel_file)
    
#     # Extract sample IDs and labels
#     sample_ids = df['new_order'].tolist()
#     labels = df['pan01'].tolist()
    
#     # Create a dictionary mapping sample IDs to labels
#     label_dict = dict(zip(sample_ids, labels))
    
#     # Get the list of sample files in the correct order
#     sample_files = sorted(glob.glob(os.path.join(sample_folder, 'sample_*.mat')), key=natural_sort_key)
    
#     # Extract sample numbers from filenames
#     sample_numbers = [int(re.search(r'sample_(\d+)', os.path.basename(f)).group(1)) for f in sample_files]
    
#     # Create the labels array in the correct order
#     ordered_labels = np.array([label_dict.get(sample_id, np.nan) for sample_id in sample_numbers])
    
#     # Save the labels to a .mat file
#     io.savemat(output_file, {'labels': ordered_labels})
    
#     print(f"Labels saved to {output_file}")
#     print(f"Number of labels: {len(ordered_labels)}")
#     print(f"First few labels: {ordered_labels[:5]}")
#     print(f"Last few labels: {ordered_labels[-5:]}")

# def main():
#     excel_file = '/vol/research/ucdatasets/gwas/data_files/disease_pheno/pan_can.xlsx'
#     sample_folder = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pan_can_seq'
#     output_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_sampled_data/pan_labels.mat'

#     create_labels_mat(excel_file, sample_folder, output_file)

# if __name__ == "__main__":
#     main()
