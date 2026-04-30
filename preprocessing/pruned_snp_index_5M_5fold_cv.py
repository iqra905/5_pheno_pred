import pandas as pd
import glob
import os

def main():
    # Path to the CSV files
    file_pattern = "/vol/research/ucdatasets/gwas/gwas_mono_rm/data_new_study_stratified_kfold/gen_data_5M_filtered_plink_files/brea/brea_fold_*_0.05_LDpruned.csv"
    
    # Get the list of files
    files = sorted(glob.glob(file_pattern))
    
    if not files:
        print(f"No files found matching the pattern: {file_pattern}")
        return
    
    print(f"Found {len(files)} files: {files}")
    
    # Columns to keep in the final output
    columns_to_keep = ["CHR", "SNP", "BP", "A1", "A2", "Global_SNP_Index"]
    
    # Initialize sets for union and intersection
    union_rows = set()
    intersection_rows = None
    
    # Process each file
    for i, file in enumerate(files):
        print(f"Processing file {i+1}/{len(files)}: {file}")
        
        # Read the file
        df = pd.read_csv(file)
        
        # Keep only the columns we need
        df = df[columns_to_keep]
        
        # Convert rows to tuples for set operations
        current_rows = set(tuple(row) for row in df.values)
        
        # Update union
        union_rows.update(current_rows)
        
        # Update intersection
        if intersection_rows is None:
            intersection_rows = current_rows
        else:
            intersection_rows.intersection_update(current_rows)
    
    # Convert sets back to DataFrames
    union_df = pd.DataFrame(list(union_rows), columns=columns_to_keep)
    print(f"Total unique rows (union): {len(union_df)}")
    
    intersection_df = pd.DataFrame(list(intersection_rows) if intersection_rows else [], columns=columns_to_keep)
    print(f"Rows common to all files (intersection): {len(intersection_df)}")
    
    # Sort both DataFrames by Global_SNP_Index
    union_df = union_df.sort_values(by="Global_SNP_Index")
    intersection_df = intersection_df.sort_values(by="Global_SNP_Index")
    
    # Define output file paths
    output_dir = os.path.dirname(files[0])
    union_output = os.path.join(output_dir, "brea_5fold_0.05_LDpruned_union.csv")
    intersection_output = os.path.join(output_dir, "brea_5fold_0.05_LDpruned_intersection.csv")
    
    # Write results to CSV files
    print(f"Writing union file to: {union_output}")
    union_df.to_csv(union_output, index=False)
    
    print(f"Writing intersection file to: {intersection_output}")
    intersection_df.to_csv(intersection_output, index=False)
    
    print("Process completed successfully!")

if __name__ == "__main__":
    main()