import gzip

def process_gen_file(input_file, output_file, row_limit=5000000):
    # Open the input .gen.gz file
    with gzip.open(input_file, 'rt') as infile:
        # Prepare to write to the output .gen.gz file
        with gzip.open(output_file, 'wt') as outfile:
            # Initialize a counter to keep track of the number of processed rows
            count = 0
            
            for line in infile:
                if count >= row_limit:
                    break
                
                # Split the line by spaces
                columns = line.strip().split(' ')
                
                # Join the processed columns back into a space-separated string
                #processed_line = ' '.join(columns)
                
                # Retain only the last 3 columns
                processed_columns = columns[-3:]
                
                # Join the processed columns back into a space-separated string
                processed_line = ' '.join(processed_columns)
                
                # Write the processed line to the output file
                outfile.write(processed_line + '\n')
                
                count += 1

# Usage
input_file = '/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/sample_00001.gen.gz'
output_file ='/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/sample_00001_processed_3cols_5Mrows.gen.gz'
process_gen_file(input_file, output_file)
