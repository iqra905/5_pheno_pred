import pandas as pd
import argparse
import os
import sys

def handle_duplicates(input_file, unq_output=None, dup_output=None, columns=None, keep='first', delimiter=',', encoding='utf-8', verbose=True):
    try:
        # Auto-generate output filenames if not provided
        if unq_output is None:
            base_name = os.path.splitext(input_file)[0]
            unq_output = f"{base_name}_unq.csv"
        
        if dup_output is None:
            base_name = os.path.splitext(input_file)[0]
            dup_output = f"{base_name}_dup.csv"
        
        # Read the CSV file
        if verbose:
            print(f"Reading file: {input_file}")
            
        df = pd.read_csv(input_file, delimiter=delimiter, encoding=encoding, low_memory=False)
        
        # Get original row count
        original_count = len(df)
        
        if verbose:
            print(f"Original file has {original_count:,} rows and {len(df.columns)} columns")
            
        # Validate columns if specified
        if columns:
            invalid_columns = [col for col in columns if col not in df.columns]
            if invalid_columns:
                print(f"Warning: These columns do not exist in the CSV: {', '.join(invalid_columns)}")
                print(f"Available columns: {', '.join(df.columns)}")
                columns = [col for col in columns if col in df.columns]
                if not columns:
                    print("Error: No valid columns specified for deduplication")
                    return -1, -1
                print(f"Proceeding with valid columns: {', '.join(columns)}")
        
        # Identify duplicate rows
        # The duplicated method returns True for the second and subsequent occurrences of each duplicated row
        dup_mask = df.duplicated(subset=columns, keep=False)
        
        # Extract unique and duplicate rows
        unique_df = df[~dup_mask].copy()
        duplicate_df = df[dup_mask].copy()

        # If keep is 'first', keep only the first occurrence in unique_df
        if keep == 'first':
            unique_df = df.drop_duplicates(subset=columns, keep='first')
        # If keep is 'last', keep only the last occurrence in unique_df
        elif keep == 'last':
            unique_df = df.drop_duplicates(subset=columns, keep='last')
            
        # Count rows in each dataframe
        unique_count = len(unique_df)
        duplicate_count = len(duplicate_df)
        
        # Write to output files
        unique_df.to_csv(unq_output, index=False, encoding=encoding)
        duplicate_df.to_csv(dup_output, index=False, encoding=encoding)
        
        if verbose:
            print(f"Found {duplicate_count:,} duplicate rows ({(duplicate_count/original_count)*100:.2f}% of total)")
            print(f"Unique rows saved to: {unq_output}")
            print(f"Duplicate rows saved to: {dup_output}")
            print(f"Unique file has {unique_count:,} rows")
            print(f"Duplicate file has {duplicate_count:,} rows")
        
        return unique_count, duplicate_count
    
    except pd.errors.EmptyDataError:
        print(f"Error: The file {input_file} is empty")
        return -1, -1
    except pd.errors.ParserError:
        print(f"Error: Could not parse {input_file} as a CSV file. Check delimiter and encoding.")
        return -1, -1
    except MemoryError:
        print(f"Error: Not enough memory to process the file. The file may be too large.")
        print("Consider using a different tool or splitting the file first.")
        return -1, -1
    except Exception as e:
        print(f"Error: {str(e)}")
        return -1, -1

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Separate a CSV file into unique and duplicate rows')
    
    parser.add_argument('-input_file', default='/vol/research/ucdatasets/gwas/data_files/5D_snp_info_files/extracted_5M/pros_0.05.csv', help='Path to input CSV file')
    parser.add_argument('-unq', help='Path to output CSV file for unique rows (default: input_file_unq.csv)')
    parser.add_argument('-dup', help='Path to output CSV file for duplicate rows (default: input_file_dup.csv)')
    parser.add_argument('-columns', nargs='+', help='Columns to consider when identifying duplicates (default: all columns)')
    parser.add_argument('-keep', choices=['first', 'last'], default='first', 
                        help='Which occurrence to keep in the unique file (default: first)')
    parser.add_argument('-delimiter', default=',', help='CSV delimiter character (default: ,)')
    parser.add_argument('-encoding', default='utf-8', help='File encoding (default: utf-8)')
    parser.add_argument('-quiet', action='store_true', help='Suppress informational output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)
    
    # Check if output directories exist and create if necessary
    for output_file in [args.unq, args.dup]:
        if output_file:
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except OSError as e:
                    print(f"Error creating output directory: {str(e)}")
                    sys.exit(1)
    
    # Handle duplicates
    unique_count, duplicate_count = handle_duplicates(
        args.input_file, 
        unq_output=args.unq,
        dup_output=args.dup,
        columns=args.columns, 
        keep=args.keep,
        delimiter=args.delimiter,
        encoding=args.encoding,
        verbose=not args.quiet
    )
    
    if unique_count < 0 or duplicate_count < 0:
        sys.exit(1)

if __name__ == "__main__":
    main()