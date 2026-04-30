import pandas as pd
import argparse
from pathlib import Path

def read_bim_file(bim_file):
    columns = ['chr', 'snp_id', 'distance', 'bp_position', 'ref_allele', 'alt_allele']
    
    try:
        df = pd.read_csv(
            bim_file,
            sep=r'\s+',  
            header=None,
            names=columns,
            engine='python'
        )
        
        if df.shape[1] != 6:
            raise ValueError(f"Expected 6 columns, but found {df.shape[1]} columns")
        
        return df
    
    except Exception as e:
        raise Exception(f"Error reading .bim file: {str(e)}")

def read_topology_file(bim_file):
    try:
        df = pd.read_csv(
            bim_file,
            sep='\t',
            dtype={
                'Unnamed: 0': str,
                'chr': str,
                'layer0_node': str,
                'layer0_name': str,
                'layer1_node': str,
                'layer1_name': str,
                'layer2_node': str,
                'layer2_name': str
            }
        )
        return df
    except Exception as e:
        raise Exception(f"Error reading topology file: {str(e)}")

def update_missing_snp_ids(df):
    df_updated = df.copy()
    
    # Create chr:bp_position format for missing SNP IDs
    mask = df_updated['snp_id'] == '.'
    df_updated.loc[mask, 'snp_id'] = df_updated.loc[mask].apply(
        lambda row: f"{row['chr']}:{row['bp_position']}", axis=1
    )
    num_replacements = mask.sum()
    return df_updated, num_replacements
    
def find_duplicate_positions(df):
    duplicates = df.groupby(['chr','bp_position']).size().reset_index(name='count')
    
    # Filter for positions that appear more than once
    duplicate_positions = duplicates[duplicates['count'] > 1]
    
    # Get all rows that match these duplicate positions
    duplicate_rows = pd.merge(
        df,
        duplicate_positions[['chr','bp_position']],
        on=['chr','bp_position']
    )
    
    # Sort by chromosome and position for better readability
    duplicate_rows = duplicate_rows.sort_values(['chr','bp_position'])
    
    return duplicate_rows

def write_duplicate_positions(df, output_file):
    # Get duplicate positions
    duplicates = find_duplicate_positions(df)
    duplicates.to_csv(output_file, sep='\t', index=False)
    
    # Calculate summary statistics
    num_total_variants = len(df)
    num_duplicate_positions = len(duplicates)
    
    # Display summary
    print("\n=== Duplicate Positions Summary ===")
    print(f"Total variants in file: {num_total_variants}")
    print(f"Number of duplicate variants (chr:position): {num_duplicate_positions}")
    #print(f"Number of unique positions with duplicates: {num_unique_duplicate_positions}")
    #print(f"\nDuplicate positions have been written to: {output_file}")
    
    # Display first few duplicate positions as example
    #print("\nFirst few duplicate positions:")
    #print(duplicates.head().to_string())

def update_topology_file(topology_df, bim_df):
    snp_ids = bim_df['snp_id'].reset_index(drop=True)
    
    # If topology file has different number of rows, adjust snp_ids accordingly
    if len(topology_df) != len(snp_ids):
        print(f"SNP-ID columns differ in length as {len(topology_df)} : {len(snp_ids)}")
    if len(topology_df) > len(snp_ids):
        # Pad with NaN if topology file is longer
        snp_ids = pd.Series([*snp_ids, *[pd.NA] * (len(topology_df) - len(snp_ids))])
    elif len(topology_df) < len(snp_ids):
        # Truncate if topology file is shorter
        snp_ids = snp_ids[:len(topology_df)]
    
    # Add snp_id column to topology DataFrame
    topology_df['layer0_name'] = snp_ids
    
    return topology_df

def main():
    parser = argparse.ArgumentParser(description='Find duplicate positions in a .bim file')
    parser.add_argument('--bim_file', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_gen_data/pruned_data_5D/t2d_merged_pruned.bim', help='Path to the .bim file')
    parser.add_argument('--topology_file', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_gen_data/pruned_data_5D/topology/T2D_topology_final.txt', help='Path to the topology.txt file')
    parser.add_argument('--output', type=str, default='/vol/research/ucdatasets/gwas/gwas_mono_rm/disease_wise_gen_data/pruned_data_5D/duplicate_positions_t2d.txt',
                      help='Path to save the duplicate positions (default: duplicate_positions.txt)')
    
    args = parser.parse_args()
    
    # Check if files exist
    for file_path in [args.bim_file, args.topology_file]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"The file {file_path} does not exist")
    
    # Process bim file
    print(f"Reading .bim file: {args.bim_file}")
    bim_df = read_bim_file(args.bim_file)
    
    # Update missing SNP IDs in bim file
    bim_df_updated, num_snp_replacements = update_missing_snp_ids(bim_df)
    print(f"\n=== SNP ID Updates ===")
    print(f"Number of missing SNP IDs replaced: {num_snp_replacements}")
    
    # Save updated bim file
    input_path = Path(args.bim_file)
    updated_bim_file = input_path.parent / f"{input_path.stem}_updated{input_path.suffix}"
    bim_df_updated.to_csv(updated_bim_file, sep='\t', index=False, header=False)
    print(f"Updated .bim file saved as: {updated_bim_file}")
    
    # Process topology file
    print(f"\nReading topology file: {args.topology_file}")
    topology_df = read_topology_file(args.topology_file)
    
    # Update topology file based on bim file
    topology_updated = update_topology_file(topology_df, bim_df_updated)
    # Print info about the number of rows
    print(f"\nTopology file rows: {len(topology_df)}")
    print(f"BIM file rows: {len(bim_df)}")
    
    # Save updated topology file as CSV
    topology_path = Path(args.topology_file)
    updated_topology_file = topology_path.parent / f"{topology_path.stem}_updated.csv"
    topology_updated.to_csv(updated_topology_file, index=False)
    print(f"Updated topology file saved as: {updated_topology_file}")
    
    # Write duplicate positions to file
    write_duplicate_positions(bim_df_updated, args.output)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)