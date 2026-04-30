import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the data with pre-calculated residuals
excel_file = '/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/brea_can_res.xlsx'
data = pd.read_excel(excel_file)

# Specify the column name for the residuals
residual_col = 'breacancer_res'  # Adjust this to match your column name

# Extract the residuals
residuals = data[residual_col]

# Create histogram
fig, ax = plt.subplots(figsize=(15, 8))

n_bins = 300  
counts, bins, patches = ax.hist(residuals, bins=n_bins, edgecolor='black')

plt.title('Histogram of Residual Phenotypes (Breast Cancer)')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')

# Add vertical line at mean
plt.axvline(residuals.mean(), color='red', linestyle='dashed', linewidth=2)
plt.text(residuals.mean(), plt.ylim()[1], 'Mean', horizontalalignment='center', verticalalignment='bottom', color='red')

# Display statistics
plt.text(0.05, 0.95, f'Mean: {residuals.mean():.5f}\nStd Dev: {residuals.std():.5f}\n'
                     f'Min: {residuals.min():.5f}\nMax: {residuals.max():.5f}', 
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Add count labels to bins with more than 1% of total samples
total_samples = len(residuals)
threshold = total_samples * 0.01
for i, (count, patch) in enumerate(zip(counts, patches)):
    if count > threshold:
        bin_center = (bins[i] + bins[i+1]) / 2
        ax.text(bin_center, count, f'{int(count)}', ha='center', va='bottom', fontweight='bold')

# Save the plot
plt.tight_layout()
plt.savefig('/vol/vssp/SF_ucdatasets/gwas/data_files/disease_pheno/residual_histogram_brea.png', dpi=300)
plt.close()

print("Histogram has been saved as 'residual_phenotypes_histogram_detailed_200.png'")

# Display basic statistics
print("\nResidual Phenotypes Statistics:")
print(f"Mean: {residuals.mean():.5f}")
print(f"Standard Deviation: {residuals.std():.5f}")
print(f"Minimum: {residuals.min():.5f}")
print(f"Maximum: {residuals.max():.5f}")
print(f"Range: {residuals.max() - residuals.min():.5f}")

# Additional statistics
print(f"\nMedian: {residuals.median():.5f}")
print(f"25th Percentile: {residuals.quantile(0.25):.5f}")
print(f"75th Percentile: {residuals.quantile(0.75):.5f}")
print(f"Skewness: {residuals.skew():.5f}")
print(f"Kurtosis: {residuals.kurtosis():.5f}")

# Check for normality
_, p_value = stats.normaltest(residuals)
print(f"\nNormality test p-value: {p_value:.5f}")
print("If p-value < 0.05, the distribution is likely not normal.")

# Print summary of bin counts
print("\nBin Summary:")
non_empty_bins = sum(counts > 0)
print(f"Total number of bins: {n_bins}")
print(f"Number of bins with samples: {non_empty_bins}")
print(f"Number of empty bins: {n_bins - non_empty_bins}")

# Print counts for all non-empty bins
print("\nDetailed Bin Information:")
for i, (count, bin_start, bin_end) in enumerate(zip(counts, bins[:-1], bins[1:])):
    if count > 0:
        print(f"Bin {i+1}: [{bin_start:.5f}, {bin_end:.5f}) - {int(count)} samples")

# Calculate and print the number of unique values
unique_values = residuals.nunique()
print(f"\nNumber of unique residual values: {unique_values}")

# Print value counts for top 20 most common residual values
print("\nTop 20 most common residual values:")
value_counts = residuals.value_counts().head(20)
for value, count in value_counts.items():
    print(f"Value: {value:.5f} - Count: {count}")