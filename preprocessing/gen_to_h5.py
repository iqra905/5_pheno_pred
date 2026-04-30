import gzip
import h5py
import numpy as np

def extract_gen_data(gen_file):
    with gzip.open(gen_file, 'rt') as f:
        # Assuming the data is stored as space-separated values
        data = np.genfromtxt(f)
    return data

def remove_columns(data):
    return data[:, 5:]

def convert_to_hdf5(data, output_file, dataset_name='data'):
    with h5py.File(output_file, 'a') as hf:
        hf.create_dataset(dataset_name, data=data)

def main():
    gen_file = '/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/sample_1.gen.gz'
    output_file = '/vol/research/fmodal_mmmed/Codes/GenNet_MLP/data_ind/sample_1.h5'

    # Extract data from the .gen.gz file
    print("Extracting data from the .gen.gz file...")
    data = extract_gen_data(gen_file)
    print("Data extraction complete.")

    # # Remove the first five columns from the data
    # print("Removing the first five columns from the data...")
    # data_trimmed = remove_columns(data)
    # print("Columns removed.")

    # # Transpose the trimmed data
    # print("Transposing the trimmed data...")
    # data_transposed = data_trimmed.T
    # print("Data transposed.")

    # Convert the data to HDF5 format
    print("Converting data to HDF5 format...")
    #convert_to_hdf5(data_trimmed, output_file, dataset_name='trimmed_data')
    #convert_to_hdf5(data_transposed, output_file, dataset_name='transposed_data')
    convert_to_hdf5(data, output_file, dataset_name='transposed_data')

    print("Conversion to HDF5 complete.")
    print("HDF5 file saved as:", output_file)

if __name__ == "__main__":
    main()

