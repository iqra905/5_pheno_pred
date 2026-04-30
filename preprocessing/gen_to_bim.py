import csv

input_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_t2d.gen'
output_file = '/vol/research/ucdatasets/gwas/gwas_mono_rm/meta_data/first_5_columns_t2d.bim'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile, delimiter=' ')
    writer = csv.writer(outfile, delimiter=' ')
    
    for row in reader:     
        new_row = row[:2] + ['0'] + row[2:]
        writer.writerow(new_row)

print(f"Conversion complete. Output saved to {output_file}")