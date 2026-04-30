import numpy as np
import torch

# Path to your .npy file
file_path = "/mnt/fast/datasets/ucdatasets/gwas/gwas_mono_rm/sampled_data_5M_unq_npy/sample_00001.npy"
#file_path = "/mnt/fast/datasets/ucdatasets/gwas/ukbb/samples_chr_wise_uint8/chr1/sample_4542299_4542299_chr1.npy"

# Load with NumPy
array = np.load(file_path)

# Convert to PyTorch tensor
tensor = torch.from_numpy(array)

# Print information
print("Tensor shape:", tensor.shape)
print("Tensor dtype:", tensor.dtype)
print("Tensor contents:")
print(tensor[:7])