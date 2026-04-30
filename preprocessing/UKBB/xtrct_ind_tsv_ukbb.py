import pandas as pd

# Read the TSV file
df = pd.read_csv('/vol/research/ucdatasets/gwas/ukbb/ukbb_disease_wise_matched_1_to_1/ukb_cancers_t2d_ukb676869_13102025_cleaned_matched_T2D.tsv', sep='\t')

#df = pd.read_csv('/vol/research/ucdatasets/gwas/ukbb/ukb_cancers_t2d_ukb676869_13102025.tsv', sep='\t')

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
    
    print(f"\n{disease}:")
    print(f"  Cases (disease = 1): {cases}")
    print(f"  Controls (disease = 0): {controls}")
    print(f"  Total: {cases + controls}")
    if cases > 0:
        print(f"  Case:Control Ratio: 1:{controls/cases:.2f}")
    else:
        print("  Case:Control Ratio: N/A")

# Calculate common controls (controls for ALL diseases)
common_controls_mask = df_cleaned[disease_columns].eq(0).all(axis=1)
common_controls_count = common_controls_mask.sum()

print("\n" + "=" * 60)
print("COMMON CONTROLS")
print("=" * 60)
print(f"Controls common across all diseases (all = 0): {common_controls_count}")

# ---------------------------------------------------------------
# NEW SECTION: COMBINATIONS OF DISEASES
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("DISEASE COMBINATION COUNTS")
print("=" * 60)

# Compute combination counts
combo_counts = df_cleaned[disease_columns].value_counts().reset_index(name='count')

# Print combinations in readable format
for _, row in combo_counts.iterrows():
    combo = row[disease_columns].tolist()
    count = row['count']
    
    # Convert binary pattern into list of present diseases
    diseases_present = [disease_columns[i] for i, val in enumerate(combo) if val == 1]
    
    if diseases_present:
        description = ", ".join(diseases_present)
    else:
        description = "None (all 0)"
    
    print(f"{description:40s}  -> {count} samples")

# Save the cleaned file
output_path = '/vol/research/ucdatasets/gwas/ukbb/ukbb_disease_wise_matched_1_to_1/ukb_cancers_t2d_ukb676869_13102025_cleaned_matched_T2D_1.tsv'
#output_path = '/vol/research/ucdatasets/gwas/ukbb/ukb_cancers_t2d_ukb676869_13102025_cleaned_1.tsv'

df_cleaned.to_csv(output_path, sep='\t', index=False)

print("\n" + "=" * 60)
print("FILE SAVED")
print("=" * 60)
print(f"Cleaned file saved to: {output_path}")
print(f"Total rows in cleaned file: {len(df_cleaned)}")
print(f"Total columns: {len(df_cleaned.columns)}")
