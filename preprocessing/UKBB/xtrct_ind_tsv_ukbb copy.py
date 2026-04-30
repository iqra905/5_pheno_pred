import pandas as pd

# Read the TSV file
df = pd.read_csv('/vol/research/ucdatasets/gwas/data_files/ukbb/ukb_cancers_t2d_ukb676869_13102025.tsv', sep='\t')

# List of disease columns
disease_columns = ['T2D', 'BC', 'CRC', 'PrC', 'PanC']

# Count samples with NA values in disease columns
na_counts = df[disease_columns].isna().sum()
samples_with_any_na = df[disease_columns].isna().any(axis=1).sum()

print("=" * 60)
print("INITIAL DATA SUMMARY")
print("=" * 60)
print(f"Total samples in original file: {len(df)}")
print(f"\nSamples with NA values in each disease column:")
for col in disease_columns:
    print(f"  {col}: {na_counts[col]}")
print(f"\nTotal samples with at least one NA value: {samples_with_any_na}")

# Remove rows with NA in any of the disease columns
df_cleaned = df.dropna(subset=disease_columns)

print(f"\nSamples after removing NAs: {len(df_cleaned)}")
print(f"Samples removed: {len(df) - len(df_cleaned)}")

# Calculate statistics
print("\n" + "=" * 60)
print("DISEASE STATISTICS (After NA removal)")
print("=" * 60)

for disease in disease_columns:
    cases = (df_cleaned[disease] == 1).sum()
    controls = (df_cleaned[disease] == 0).sum()
    ratio = cases / controls if controls > 0 else float('inf')
    
    print(f"\n{disease}:")
    print(f"  Cases (disease = 1): {cases}")
    print(f"  Controls (disease = 0): {controls}")
    print(f"  Total: {cases + controls}")
    print(f"  Case:Control Ratio: 1:{controls/cases:.2f}" if cases > 0 else "  Case:Control Ratio: N/A")

# Calculate common controls (controls for ALL diseases)
common_controls_mask = df_cleaned[disease_columns].eq(0).all(axis=1)
common_controls_count = common_controls_mask.sum()

print("\n" + "=" * 60)
print("COMMON CONTROLS")
print("=" * 60)
print(f"Controls common across all diseases (all = 0): {common_controls_count}")

# Save the cleaned file
output_path = '/vol/research/ucdatasets/gwas/data_files/ukbb/ukb_cancers_t2d_ukb676869_13102025_cleaned.tsv'
df_cleaned.to_csv(output_path, sep='\t', index=False)

print("\n" + "=" * 60)
print("FILE SAVED")
print("=" * 60)
print(f"Cleaned file saved to: {output_path}")
print(f"Total rows in cleaned file: {len(df_cleaned)}")
print(f"Total columns: {len(df_cleaned.columns)}")