import pandas as pd

# Read the file
input_file = '/vol/research/fmodal_mmmed/Codes/plink_linux_x86_64_20240818/37k.sample.txt'
output_file = '/vol/research/fmodal_mmmed/Codes/plink_linux_x86_64_20240818/37k.sample'

# Read the file into a pandas DataFrame
# We're using read_csv with sep=None and engine='python' to automatically detect the delimiter
df = pd.read_csv(input_file, sep=None, engine='python')

# If you need to perform any operations on the data, do it here
# For example, if you need to modify certain columns or rows

# Save the DataFrame back to a file
# We're using to_csv with sep='\t' to save as a tab-separated file, which is common for .sample files
df.to_csv(output_file, sep='\t', index=False)

print(f"File has been updated and saved as {output_file}")