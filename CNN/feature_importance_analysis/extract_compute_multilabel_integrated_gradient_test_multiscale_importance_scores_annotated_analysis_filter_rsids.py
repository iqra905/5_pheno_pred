import pandas as pd
import argparse
import sys


def filter_snp_data(input_file, output_file):
    """
    Filter SNP data by removing rows where snp_id contains ':'
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Display initial row count
        print(f"Initial number of rows: {len(df)}")
        
        # Check if snp_id column exists
        if 'snp_id' not in df.columns:
            print(f"Error: 'snp_id' column not found in the file.")
            print(f"Available columns: {', '.join(df.columns)}")
            return False
        
        # Filter out rows where snp_id contains ":"
        df_filtered = df[~df['snp_id'].astype(str).str.contains(':', na=False)]
        
        # Display filtered row count
        print(f"Rows after filtering: {len(df_filtered)}")
        print(f"Rows removed: {len(df) - len(df_filtered)}")
        
        # Save the filtered data to a new CSV file
        df_filtered.to_csv(output_file, index=False)
        
        print(f"\nFiltered data saved to: {output_file}")
        return True
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def main():
    """Main function to handle command-line arguments and run the filter"""
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Filter SNP data by removing rows with ":" in snp_id column')
    parser.add_argument("-input_file", default= None, help='Path to input CSV file')
    parser.add_argument("-output_file", default= None, help='Path to output CSV file')
    
    args = parser.parse_args()
    
    # Run the filter function
    success = filter_snp_data(args.input_file, args.output_file)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()