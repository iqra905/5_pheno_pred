import pandas as pd
import numpy as np

def process_files(gen_file, txt_file, output_file, chunksize=100000):
    # Read txt file
    txt_df = pd.read_csv(txt_file, sep='\t')
    txt_df['Position'] = txt_df['Position'].astype(np.int64)
    
    # Create dictionary for matching
    valid_positions = {(int(chr_), int(pos)): True 
                      for chr_, pos in zip(txt_df['Chromosome'], txt_df['Position'])}
    
    print(f"Number of unique positions in txt file: {len(valid_positions)}")
    
    all_filtered_chunks = []
    
    for chunk in pd.read_csv(gen_file, header=None, sep=r'\s+',
                            names=['chr', 'snp_id', 'position', 'ref', 'alt'],
                            chunksize=chunksize):
        
        # Convert to integers for matching
        chunk['chr'] = chunk['chr'].astype(np.int64)
        chunk['position'] = chunk['position'].astype(np.int64)
        
        # Filter rows that match positions in txt file
        mask_position = [valid_positions.get((c, p), False) 
                        for c, p in zip(chunk['chr'], chunk['position'])]
        filtered_chunk = chunk[mask_position].copy()
        
        if len(filtered_chunk) > 0:
            all_filtered_chunks.append(filtered_chunk)
    
    # Combine all filtered chunks
    final_df = pd.concat(all_filtered_chunks, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=['chr', 'position'])
    
    # Update SNP IDs where needed
    mask_snp = final_df['snp_id'] == '.'
    final_df.loc[mask_snp, 'snp_id'] = (
        final_df.loc[mask_snp, 'chr'].astype(str) + ':' +
        final_df.loc[mask_snp, 'position'].astype(str) + ':' +
        final_df.loc[mask_snp, 'alt'] + ':' +
        final_df.loc[mask_snp, 'ref']
    )
    
    # Save final result
    final_df.to_csv(output_file, sep=' ', header=False, index=False)
    
    print(f"\nTotal rows in txt file: {len(txt_df)}")
    print(f"Unique positions in txt file: {len(valid_positions)}")
    print(f"Total rows in output file: {len(final_df)}")

# Usage
gen_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M.gen'
txt_file = '/vol/research/ucdatasets/gwas/data_files/5d_gwas_05maf_0001hwe_08info_VL2.txt'
output_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated.gen'
process_files(gen_file, txt_file, output_file)
