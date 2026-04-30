#!/usr/bin/env python3

# Input and output file paths
input_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen'
output_file = '/vol/research/fmodal_mmmed/Codes/Datasets/NewDataset/max_basepair_per_chromosome.txt'

# Dictionary to store max basepair position and corresponding row for each chromosome
max_positions = {}

print(f"Processing file: {input_file}")
print("Reading rows and finding max basepair positions per chromosome...")

# Read the file line by line for memory efficiency
with open(input_file, 'r') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        
        # Split the space-delimited columns
        parts = line.split()
        
        # Parse columns: chromosome, rsid, basepair, ref_allele, alt_allele
        chromosome = parts[0]
        basepair = int(parts[2])
        
        # Update if this is the first entry for this chromosome or has a larger basepair value
        if chromosome not in max_positions or basepair > max_positions[chromosome][0]:
            max_positions[chromosome] = (basepair, parts)
        
        # Progress indicator every 500k rows
        if line_num % 500000 == 0:
            print(f"  Processed {line_num:,} rows...")

print(f"Completed processing {line_num:,} total rows")

# Determine column widths by finding max length in each column
col_widths = [len("chr"), len("rsid"), len("basepair"), len("A1"), len("A2")]

for chrom in max_positions.keys():
    basepair, parts = max_positions[chrom]
    for i, val in enumerate(parts):
        col_widths[i] = max(col_widths[i], len(val))

# Write results to output file with aligned columns
print(f"\nWriting results to: {output_file}")
with open(output_file, 'w') as f:
    # Write header with proper alignment
    header = f"{'chr':<{col_widths[0]}}  {'rsid':<{col_widths[1]}}  {'basepair':<{col_widths[2]}}  {'A1':<{col_widths[3]}}  {'A2':<{col_widths[4]}}"
    f.write(header + '\n')
    
    # Write data rows with proper alignment, sorted by chromosome number
    for chrom in sorted(max_positions.keys(), key=lambda x: int(x)):
        basepair, parts = max_positions[chrom]
        row = f"{parts[0]:<{col_widths[0]}}  {parts[1]:<{col_widths[1]}}  {parts[2]:<{col_widths[2]}}  {parts[3]:<{col_widths[3]}}  {parts[4]:<{col_widths[4]}}"
        f.write(row + '\n')
        print(f"  Chromosome {chrom}: max basepair = {basepair:,}")

print(f"\nSuccess! Found max positions for {len(max_positions)} chromosomes")


# #!/usr/bin/env python3

# # Input and output file paths
# input_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen'
# output_file = '/vol/research/fmodal_mmmed/Codes/Datasets/NewDataset/max_basepair_per_chromosome.txt'

# # Dictionary to store max basepair position and corresponding row for each chromosome
# max_positions = {}

# print(f"Processing file: {input_file}")
# print("Reading rows and finding max basepair positions per chromosome...")

# # Read the file line by line for memory efficiency
# with open(input_file, 'r') as f:
#     for line_num, line in enumerate(f, 1):
#         line = line.strip()
#         if not line:
#             continue
        
#         # Split the space-delimited columns
#         parts = line.split()
        
#         # Parse columns: chromosome, rsid, basepair, ref_allele, alt_allele
#         chromosome = parts[0]
#         basepair = int(parts[2])
        
#         # Update if this is the first entry for this chromosome or has a larger basepair value
#         if chromosome not in max_positions or basepair > max_positions[chromosome][0]:
#             max_positions[chromosome] = (basepair, line)
        
#         # Progress indicator every 500k rows
#         if line_num % 500000 == 0:
#             print(f"  Processed {line_num:,} rows...")

# print(f"Completed processing {line_num:,} total rows")

# # Write results to output file, sorted by chromosome number
# print(f"\nWriting results to: {output_file}")
# with open(output_file, 'w') as f:
#     for chrom in sorted(max_positions.keys(), key=lambda x: int(x)):
#         basepair, row = max_positions[chrom]
#         f.write(row + '\n')
#         print(f"  Chromosome {chrom}: max basepair = {basepair:,}")

# print(f"\nSuccess! Found max positions for {len(max_positions)} chromosomes")

# #!/usr/bin/env python3

# # Input and output file paths
# input_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_5M_updated_unq.gen'
# output_file = '/vol/research/fmodal_mmmed/Codes/Datasets/NewDataset/max_basepair_per_chromosome1.txt'

# # Dictionary to store max basepair position and corresponding row for each chromosome
# max_positions = {}

# print(f"Processing file: {input_file}")
# print("Reading rows and finding max basepair positions per chromosome...")

# # Read the file line by line for memory efficiency
# with open(input_file, 'r') as f:
#     for line_num, line in enumerate(f, 1):
#         line = line.strip()
#         if not line:
#             continue
        
#         # Split the space-delimited columns
#         parts = line.split()
        
#         # Parse columns: chromosome, rsid, basepair, ref_allele, alt_allele
#         chromosome = parts[0]
#         basepair = int(parts[2])
        
#         # Update if this is the first entry for this chromosome or has a larger basepair value
#         if chromosome not in max_positions or basepair > max_positions[chromosome][0]:
#             max_positions[chromosome] = (basepair, parts)
        
#         # Progress indicator every 500k rows
#         if line_num % 500000 == 0:
#             print(f"  Processed {line_num:,} rows...")

# print(f"Completed processing {line_num:,} total rows")

# # Write results to output file with header and tab-delimited format
# print(f"\nWriting results to: {output_file}")
# with open(output_file, 'w') as f:
#     # Write header
#     f.write("chromosome\trsid\tbasepair\tref_allele\talt_allele\n")
    
#     # Write data rows, sorted by chromosome number
#     for chrom in sorted(max_positions.keys(), key=lambda x: int(x)):
#         basepair, parts = max_positions[chrom]
#         # Join columns with tabs
#         f.write('\t'.join(parts) + '\n')
#         print(f"  Chromosome {chrom}: max basepair = {basepair:,}")

# print(f"\nSuccess! Found max positions for {len(max_positions)} chromosomes")