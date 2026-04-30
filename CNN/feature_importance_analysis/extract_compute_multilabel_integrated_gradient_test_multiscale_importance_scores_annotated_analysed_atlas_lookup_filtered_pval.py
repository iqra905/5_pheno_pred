#!/usr/bin/env python3
"""
Dual Manhattan Plot Generator with GWAS Atlas Validation - HYBRID COLOR VERSION
--------------------------------------------------------------------------------
Creates mirror Manhattan plots with HYBRID COLORING in bottom panel:
  - Top: SNP importance scores from ML model (alternating colors by chromosome)
    * GWAS-significant SNPs: Red diamonds with labels (shows published validation)
    * Top importance SNPs: Yellow labels (may overlap with GWAS-significant)
  - Bottom: GWAS Atlas p-values for ALL associations (mirrored) - SQRT TRANSFORMED
    * 5 main diseases: Disease-specific colors (Yellow=T2DM, Orange=Breast, Cyan=Pancreatic, Green=Prostate, Magenta=Colorectal)
    * Other categories: Alternating blue/gray by chromosome
    * Significant (p < 5e-8): Disease-colored diamonds (or chromosome-colored for "Other")
    * Legend includes total associations count (X associations, Y unique SNPs)
  
Also filters and saves genome-wide significant associations (p < 5e-8)
Creates BOTH a dual Manhattan plot AND a standalone bottom panel plot

FEATURES:
  - Bottom panel uses sqrt(-log10(p)) transformation to spread out compressed values
  - HYBRID COLOR SCHEME: Disease-specific for 5 main diseases, chromosome-based for others
  - Diamond markers indicate GWAS significance (p<5e-8), colored appropriately
  - Trait normalization to 5 disease categories
  - One row per SNP-disease combination (minimum p-value)
  - Complete visualization of all GWAS associations
  - Standalone bottom panel plot (full height) for separate viewing/publication
  - **Standalone plot shows only 5 main diseases** (excludes "Other" for visual clarity)
  - Legend displays detailed per-disease statistics (total assoc, total SNPs, sig assoc, sig SNPs)

    
Options:
    -highlight_gwas_sig 1        # Show red diamonds for GWAS-significant SNPs in top panel (default)
    -highlight_top_n 10          # Label top 10 importance SNPs with yellow text (default)
    -label_gwas_sig_top_n 10     # Label top 10 GWAS-significant SNPs in top panel (default)
    -label_top_significant_n 10  # Label top 10 significant SNPs in bottom panel (default)
    -label_top_significant_n 0   # Label ALL significant SNPs in bottom panel
    -label_top_significant_n -1  # Disable labeling in bottom panel
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Genome-wide significance threshold
GWAS_SIGNIFICANCE = 5e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dual Manhattan plots with normalized y-axis and filter significant GWAS associations"
    )
    
    # parser.add_argument("-importance", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_disease_wise/t2dm_all_snps_importance_loss_based_disease_wise_test_set_annotated.csv', help="CSV file with importance scores (from Script 1)")
    # parser.add_argument("-lookup", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_disease_wise/analysis_20251218_023025/filtered_snps/t2dm_all_snps_importance_loss_based_disease_wise_test_set_std_filtered_altas_lookup_cancer_diabetes.csv', help="CSV file with GWAS Atlas associations (from Script 2)")
    # parser.add_argument("-output", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_disease_wise/analysis_20251218_023025/filtered_snps/', help="Output directory for plots and filtered data")
        
    parser.add_argument("-importance", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_disease_wise_no_cov/ig_50/t2dm_all_snps_importance_integrated_gradients_disease_wise_test_set_annotated.csv', help="CSV file with importance scores (from Script 1)")
    parser.add_argument("-lookup", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_disease_wise_no_cov/ig_50/analysis_20260102_122403/filtered_snps/All-Diseases_all_snps_importance_integrated_gradients_test_set_std_filtered_altas_lookup_cancer_diabetes_deduplicated.csv', help="CSV file with GWAS Atlas associations (from Script 2)")
    parser.add_argument("-output", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_disease_wise_no_cov/ig_50/analysis_20260102_122403/filtered_snps/', help="Output directory for plots and filtered data")
     
    
    # parser.add_argument("-importance", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_overall_no_cov/ig_50/overall_all_snps_importance_integrated_gradients_overall_test_set_annotated.csv', help="CSV file with importance scores (from Script 1)")
    # parser.add_argument("-lookup", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_overall_no_cov/ig_50/analysis_20260102_122349/filtered_snps/overall_all_snps_importance_integrated_gradients_overall_test_set_std_filtered_altas_lookup_cancer_diabetes.csv', help="CSV file with GWAS Atlas associations (from Script 2)")
    # parser.add_argument("-output", type=str, default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_overall_no_cov/ig_50/analysis_20260102_122349/filtered_snps/', help="Output directory for plots and filtered data")


    # Optional filtering
    parser.add_argument("-min_importance", type=float, default=None, help="Minimum importance score to display (default: show all)")
    parser.add_argument("-max_pvalue", type=float, default=None, help="[DEPRECATED - no longer used] All associations are shown; significant ones (p<5e-8) are highlighted in yellow")
    
    # Visualization options
    parser.add_argument("-point_size", type=float, default=5, help="Size of points in Manhattan plot (default: 3)")
    parser.add_argument("-highlight_top_n", type=int, default=0, help="Number of top importance SNPs to label in top panel with yellow labels (default: 10, 0 to disable)")
    parser.add_argument("-highlight_gwas_sig", type=int, default=1, choices=[0, 1], help="Highlight GWAS-significant SNPs as red diamonds in BOTH panels (default: 1=yes, 0=no)")
    parser.add_argument("-label_gwas_sig_top_n", type=int, default=1, help="Number of top GWAS-significant SNPs to label in top panel (default: 10, 0=all, -1=none)")
    parser.add_argument("-label_significant", type=int, default=1, choices=[0, 1], help="Label genome-wide significant SNPs in bottom panel (default: 1=yes, 0=no)")
    parser.add_argument("-label_top_significant_n", type=int, default=1, help="Number of top significant SNPs to label in bottom panel (default: 10, 0=all significant, -1=none)")
    parser.add_argument("-plot_format", type=str, default="pdf", choices=["png", "pdf", "svg"], help="Plot format (default: pdf)")
    parser.add_argument("-dpi", type=int, default=300, help="DPI for PNG output (default: 300)")
    parser.add_argument("-figsize", type=str, default="16,10", help="Figure size as 'width,height' in inches (default: 16,10)")
    
    return parser.parse_args()


def natural_sort_key(text):
    """Natural sorting for chromosomes (1, 2, ..., 22, X, Y, MT)"""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', str(text))]


def detect_entity_name(lookup_filename):
    """
    Detect entity/disease name from lookup filename.
    Splits filename on underscore and uses first part (index 0).
    """
    basename = Path(lookup_filename).stem  # Remove extension
    
    # Split on underscore and take first part
    first_part = basename.split('_')[0].lower()
    
    # Disease name mapping
    disease_map = {
        'breast': 'Breast Cancer',
        'breacancer': 'Breast Cancer',
        't2dm': 'Type 2 Diabetes',
        't2d': 'Type 2 Diabetes',
        'diabetes': 'Type 2 Diabetes',
        'panca': 'Pancreatic Cancer',
        'pancreatic': 'Pancreatic Cancer',
        'pros': 'Prostate Cancer',
        'pros01': 'Prostate Cancer',
        'prostate': 'Prostate Cancer',
        'crc': 'Colorectal Cancer',
        'colorectal': 'Colorectal Cancer',
        'overall': 'Overall',
        'multi': 'Multi-Label',
        'multilabel': 'Multi-Label'
    }
    
    # Check if first part matches any disease pattern
    if first_part in disease_map:
        return disease_map[first_part]
    else:
        # If not in map, capitalize the first part
        return first_part.capitalize() if first_part else 'SNP'


def prepare_chromosome_data(df):
    """
    Prepare data for Manhattan plot:
    - Sort chromosomes naturally
    - Create cumulative positions for x-axis
    - Calculate chromosome midpoints for labels
    """
    # Clean chromosome names
    df['chromosome'] = df['chromosome'].astype(str).str.replace('chr', '', case=False)
    
    # Sort by chromosome and position
    chr_order = sorted(df['chromosome'].unique(), key=natural_sort_key)
    df['chr_order'] = pd.Categorical(df['chromosome'], categories=chr_order, ordered=True)
    df = df.sort_values(['chr_order', 'bp']).reset_index(drop=True)
    
    # Calculate cumulative positions
    df['cumulative_pos'] = 0.0  # Initialize as float to avoid dtype warning
    cumulative_offset = 0
    chr_midpoints = {}
    chr_boundaries = {}
    
    for chrom in chr_order:
        chr_data = df[df['chromosome'] == chrom]
        if len(chr_data) == 0:
            continue
            
        chr_length = chr_data['bp'].max() - chr_data['bp'].min()
        chr_start = cumulative_offset
        chr_end = cumulative_offset + chr_length
        
        # Update cumulative positions
        df.loc[df['chromosome'] == chrom, 'cumulative_pos'] = (
            df.loc[df['chromosome'] == chrom, 'bp'] - chr_data['bp'].min() + cumulative_offset
        )
        
        # Store midpoint for chromosome label
        chr_midpoints[chrom] = (chr_start + chr_end) / 2
        chr_boundaries[chrom] = (chr_start, chr_end)
        
        # Add gap between chromosomes
        cumulative_offset = chr_end + chr_length * 0.02
    
    return df, chr_order, chr_midpoints, chr_boundaries


def normalize_trait_name(trait_name):
    """
    Normalize trait names to one of 5 canonical disease categories.
    
    Returns:
        Normalized trait name (one of the 5 canonical categories) or 'Other'
    """
    trait_lower = str(trait_name).lower()
    
    # Type 2 Diabetes - match various diabetes-related terms
    if any(keyword in trait_lower for keyword in [
        'diabetes', 'diabetic', 'dm2', 't2d', 't2dm', 'type 2', 'type ii',
        'e11', 'e10', 'glucose', 'glyc', 'hba1c', 'insulin'
    ]):
        return 'Type 2 Diabetes'
    
    # Colorectal Cancer
    elif any(keyword in trait_lower for keyword in [
        'colorectal', 'colon', 'rectal', 'rectum', 'crc', 'bowel',
        'c18', 'c19', 'c20', 'd12'  # ICD codes
    ]):
        return 'Colorectal Cancer'
    
    # Breast Cancer
    elif any(keyword in trait_lower for keyword in [
        'breast', 'mammary', 'c50', 'd05'  # ICD codes
    ]):
        return 'Breast Cancer'
    
    # Prostate Cancer
    elif any(keyword in trait_lower for keyword in [
        'prostate', 'prostatic', 'c61', 'd07.5'  # ICD codes
    ]):
        return 'Prostate Cancer'
    
    # Pancreatic Cancer
    elif any(keyword in trait_lower for keyword in [
        'pancrea', 'c25', 'd13.6'  # ICD codes
    ]):
        return 'Pancreatic Cancer'
    
    # If it's a cancer but not one of the above
    elif any(keyword in trait_lower for keyword in [
        'cancer', 'carcinoma', 'neoplasm', 'tumor', 'tumour', 'malign'
    ]):
        return 'Other Cancer'
    
    else:
        return 'Other'


def aggregate_multiple_associations(df_lookup):
    """
    Handle SNPs with multiple GWAS associations.
    
    Strategy:
    1. Normalize trait names to canonical categories
    2. Keep all unique normalized traits
    3. For same normalized trait: keep only the one with minimum p-value
    
    Args:
        df_lookup: DataFrame with GWAS associations (may have multiple rows per SNP)
    
    Returns:
        DataFrame with deduplicated associations (one row per SNP-normalized_trait combination)
    """
    print("  • Normalizing trait names to 5 canonical categories...")
    
    # Add normalized trait column
    df_lookup['normalized_trait'] = df_lookup['trait'].apply(normalize_trait_name)
    
    # Show normalization stats
    print(f"    Original unique traits: {df_lookup['trait'].nunique()}")
    print(f"    Normalized to: {df_lookup['normalized_trait'].nunique()} categories")
    print(f"    Distribution:")
    for trait, count in df_lookup['normalized_trait'].value_counts().head(10).items():
        print(f"      - {trait}: {count:,}")
    
    print("  • Deduplication strategy: Keep all unique normalized traits, deduplicate by min p-value")
    
    # Group by rsID and NORMALIZED trait, keep the most significant (min p-value) for each combination
    df_dedup = df_lookup.sort_values('pval').groupby(['rsid', 'normalized_trait'], as_index=False).first()
    
    # Add a column indicating how many associations this SNP originally had
    snp_counts = df_lookup.groupby('rsid').size().reset_index(name='n_total_associations')
    df_dedup = df_dedup.merge(snp_counts, on='rsid', how='left')
    
    # Add how many unique NORMALIZED traits per SNP
    trait_counts = df_lookup.groupby('rsid')['normalized_trait'].nunique().reset_index(name='n_unique_normalized_traits')
    df_dedup = df_dedup.merge(trait_counts, on='rsid', how='left')
    
    return df_dedup


def filter_significant_associations(df_gwas, pval_threshold=5e-8):
    """
    Filter GWAS associations below p-value threshold.
    Returns ONLY rows where p-value < threshold (not all rows for those SNPs).
    """
    # Get only rows where p-value is below threshold
    df_significant = df_gwas[df_gwas['pval'] < pval_threshold].copy()
    
    # Sort by p-value
    df_significant = df_significant.sort_values('pval')
    
    return df_significant


def plot_dual_manhattan(df_importance, df_gwas_agg, chr_order, chr_midpoints, 
                        chr_boundaries, output_path, args, lookup_basename='', entity_name='SNP',
                        df_significant=None):
    """
    Create dual Manhattan plot with HYBRID COLORING in bottom panel:
    - Top panel: Importance scores with threshold line (alternating blue/gray by chromosome)
      * GWAS-significant SNPs: Red diamonds with labels (shows published validation)
      * Top importance SNPs: Yellow label boxes (may overlap with red diamonds)
    - Bottom panel: sqrt[-log10(p-values)] from GWAS (mirrored) - SQRT TRANSFORMED
      * All associations shown (no p-value filtering)
      * 5 main diseases: Disease-specific colors (Yellow=T2DM, Orange=Breast, Cyan=Pancreatic, Green=Prostate, Magenta=Colorectal)
      * Other categories: Alternating blue/gray by chromosome (same as top panel)
      * Significant (p < 5e-8): Disease-colored diamonds (or chromosome-colored for "Other")
      * SQRT transformation spreads out compressed values for better visualization
      * Legend includes total associations count (X associations, Y unique SNPs)
      
    Hybrid approach: Disease colors highlight the 5 main diseases, chromosome colors for others.
    """
    figsize = tuple(map(float, args.figsize.split(',')))
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    
    # Create gridspec for better control
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.05,
                         left=0.08, right=0.98, top=0.95, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Color palette - alternating colors for chromosomes (top panel and "Other" categories)
    colors_main = ['#0C4C8A', '#7F7F7F']  # Dark Blue and Gray alternating
    
    # Disease-specific color palette for bottom panel - ONLY 5 main diseases
    # Other categories will use chromosome-based alternating colors
    disease_colors = {
        'Type 2 Diabetes': '#FFD700',      # Yellow (Gold)
        'Breast Cancer': '#FF8C00',        # Orange (Dark Orange)
        'Pancreatic Cancer': '#00FFFF',    # Cyan
        'Prostate Cancer': '#32CD32',      # Green (Lime Green)
        'Colorectal Cancer': '#FF00FF'     # Magenta
    }
    
    # List of diseases that should use disease-specific colors
    main_diseases = list(disease_colors.keys())
    
    # Calculate threshold for importance scores
    # Use standard deviation method: mean + 5*std (default)
    mean_importance = df_importance['Importance_Score'].mean()
    std_importance = df_importance['Importance_Score'].std()
    importance_threshold = mean_importance + 5.0 * std_importance
    
    # ==============================================================
    # TOP PANEL: Importance Scores
    # ==============================================================
    print("  • Plotting importance scores (top panel)...")
    
    for i, chrom in enumerate(chr_order):
        df_chr = df_importance[df_importance['chromosome'] == chrom]
        if len(df_chr) == 0:
            continue
        
        color = colors_main[i % 2]
        
        ax1.scatter(df_chr['cumulative_pos'], 
                   df_chr['Importance_Score'],
                   c=color, 
                   s=args.point_size, 
                   alpha=0.7,
                   edgecolors='none',
                   rasterized=True)
    
    # Count SNPs above threshold
    n_above_threshold = (df_importance['Importance_Score'] > importance_threshold).sum()
    
    # Add threshold line with count
    ax1.axhline(y=importance_threshold, color='red', 
                linestyle='--', linewidth=1.5, alpha=0.8,
                label=f'Threshold (mean + 5×std): {importance_threshold:.2e}\n    SNPs above threshold: {n_above_threshold:,}',
                zorder=10)
    
    # Highlight GWAS-significant SNPs with RED DIAMONDS (if requested) - MATCHING bottom panel style
    if args.highlight_gwas_sig and df_significant is not None and len(df_significant) > 0:
        # Get unique rsIDs of GWAS-significant SNPs
        sig_rsids = set(df_significant['rsid'].unique())
        
        # Find these SNPs in the importance data
        df_importance['is_gwas_sig'] = df_importance['rsid'].isin(sig_rsids)
        df_gwas_sig_in_importance = df_importance[df_importance['is_gwas_sig']].copy()
        
        if len(df_gwas_sig_in_importance) > 0:
            # Plot as RED DIAMONDS - matching bottom panel
            ax1.scatter(df_gwas_sig_in_importance['cumulative_pos'], 
                       df_gwas_sig_in_importance['Importance_Score'],
                       c='red', 
                       s=args.point_size * 3, 
                       marker='D',  # Diamond shape to match bottom panel
                       edgecolors='darkred',
                       linewidths=0.8,
                       alpha=0.9,
                       zorder=80,  # Below top importance labels (z=100) but above regular points
                       label=f'GWAS-significant (p<5×10^-8): {len(df_gwas_sig_in_importance):,} SNPs')
            
            print(f"    - Highlighted {len(df_gwas_sig_in_importance)} GWAS-significant SNPs as red diamonds in top panel")
            
            # Add labels to GWAS-significant SNPs in top panel (if requested)
            if args.label_gwas_sig_top_n != -1:
                # Merge with significant data to get p-values
                df_gwas_sig_labeled = df_gwas_sig_in_importance.merge(
                    df_significant[['rsid', 'pval']].drop_duplicates('rsid'),
                    on='rsid',
                    how='left'
                )
                
                # Sort by p-value (most significant first) if we have p-values
                if 'pval' in df_gwas_sig_labeled.columns:
                    df_gwas_sig_labeled = df_gwas_sig_labeled.sort_values('pval')
                
                # Determine how many to label
                if args.label_gwas_sig_top_n == 0:
                    # Label all GWAS-significant SNPs
                    df_to_label = df_gwas_sig_labeled
                    if len(df_to_label) > 50:
                        print(f"  ⚠ Labeling all {len(df_to_label)} GWAS-significant SNPs may cause clutter")
                else:
                    # Label only top N most significant
                    df_to_label = df_gwas_sig_labeled.head(args.label_gwas_sig_top_n)
                    if len(df_gwas_sig_labeled) > args.label_gwas_sig_top_n:
                        print(f"  • Labeling top {args.label_gwas_sig_top_n} GWAS-significant SNPs in top panel (out of {len(df_gwas_sig_labeled)} total)")
                
                # Determine ID column
                id_col = 'rsid' if 'rsid' in df_to_label.columns else 'snp_id'
                
                # Add labels with p-value info
                for idx, row in df_to_label.iterrows():
                    if 'pval' in row and pd.notna(row['pval']):
                        label_text = f"{row[id_col]}, p={row['pval']:.2e}"
                    else:
                        label_text = f"{row[id_col]}\n(GWAS sig)"
                    
                    ax1.annotate(label_text, 
                                (row['cumulative_pos'], row['Importance_Score']),
                                fontsize=7, 
                                alpha=0.8,
                                fontweight='bold',
                                color='darkred',
                                xytext=(5, -15),  # Offset downward to avoid overlap with yellow labels
                                textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor='pink',  # Different color from yellow to distinguish
                                         alpha=0.4, 
                                         edgecolor='darkred',
                                         linewidth=0.5),
                                zorder=90)
        else:
            print(f"    - No GWAS-significant SNPs found in importance data to highlight")
    
    # Highlight top N importance SNPs with YELLOW LABELS (may overlap with red diamonds)
    if args.highlight_top_n > 0:
        id_col = None
        if 'snp_id' in df_importance.columns:
            id_col = 'snp_id'
        elif 'rsid' in df_importance.columns:
            id_col = 'rsid'
        
        if id_col:
            top_snps = df_importance.nlargest(args.highlight_top_n, 'Importance_Score')
            
            # Note: We don't plot additional red diamond markers here anymore
            # Red diamonds are now exclusively used for GWAS-significant SNPs
            # We only add yellow labels for top importance scores
            
            # Add yellow labels for top importance SNPs
            for idx, row in top_snps.iterrows():
                ax1.annotate(row[id_col], 
                            (row['cumulative_pos'], row['Importance_Score']),
                            fontsize=7, 
                            alpha=0.8,
                            fontweight='bold',
                            color='black',
                            xytext=(5, 5), 
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor='yellow', 
                                     alpha=0.5,
                                     edgecolor='orange',
                                     linewidth=0.5),
                            zorder=100)  # Highest z-order so always visible
    
    # Styling for top panel
    ax1.set_ylabel('Importance Score', fontsize=13, fontweight='bold')
    ax1.set_title(f'{entity_name} - Manhattan Plot of SNP Importance', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    
    # Legend in top right corner
    ax1.legend(loc='upper right', frameon=True, fontsize=9, 
              framealpha=0.9, edgecolor='black', fancybox=False)
    
    # Use scientific notation for y-axis
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Add subtle chromosome separators
    for chrom in chr_order[:-1]:
        if chrom in chr_boundaries:
            boundary = chr_boundaries[chrom][1]
            ax1.axvline(x=boundary, color='gray', linestyle='-', linewidth=0.3, alpha=0.3, zorder=0)
    
    # ==============================================================
    # BOTTOM PANEL: GWAS P-values (MIRRORED) - WITH SQRT TRANSFORMATION
    # ==============================================================
    print("  • Plotting GWAS p-values (bottom panel) with sqrt transformation...")
    print("  • Disease color palette (5 main diseases):")
    for disease, color in disease_colors.items():
        disease_count = len(df_gwas_agg[df_gwas_agg['normalized_trait'] == disease])
        if disease_count > 0:
            print(f"    - {disease}: {color} (n={disease_count})")
    
    # Check if there are "Other" categories
    df_others = df_gwas_agg[~df_gwas_agg['normalized_trait'].isin(main_diseases)]
    if len(df_others) > 0:
        other_traits = df_others['normalized_trait'].unique()
        print(f"  • Other categories (chromosome-based coloring):")
        for trait in other_traits:
            trait_count = len(df_gwas_agg[df_gwas_agg['normalized_trait'] == trait])
            print(f"    - {trait}: alternating blue/gray (n={trait_count})")
    
    # *** MODIFICATION: Apply sqrt transformation to spread out values ***
    df_gwas_agg['-log10_pval'] = -np.log10(df_gwas_agg['pval'] + 1e-300)
    df_gwas_agg['sqrt_log10_pval'] = np.sqrt(df_gwas_agg['-log10_pval'])
    
    # Identify significant SNPs
    df_sig = df_gwas_agg[df_gwas_agg['pval'] < GWAS_SIGNIFICANCE].copy()
    df_nonsig = df_gwas_agg[df_gwas_agg['pval'] >= GWAS_SIGNIFICANCE].copy()
    
    print(f"    - Total associations: {len(df_gwas_agg):,}")
    print(f"    - Significant (p<5e-8): {len(df_sig):,}")
    print(f"    - Non-significant: {len(df_nonsig):,}")
    print(f"    - Using sqrt transformation to reduce compression")
    print(f"    - Coloring: 5 main diseases by disease color, others by chromosome")
    
    # Plot non-significant SNPs - HYBRID APPROACH
    # 1. First plot the 5 main diseases with disease-specific colors
    for disease, color in disease_colors.items():
        df_disease_nonsig = df_nonsig[df_nonsig['normalized_trait'] == disease]
        if len(df_disease_nonsig) == 0:
            continue
        
        # Plot as NEGATIVE values to create mirror effect
        ax2.scatter(df_disease_nonsig['cumulative_pos'], 
                   -df_disease_nonsig['sqrt_log10_pval'],  # Negative for mirror, sqrt transformed
                   c=color, 
                   s=args.point_size * 2, 
                   alpha=0.6,  # Slightly transparent for overlapping points
                   edgecolors='none',
                   rasterized=True)
        
        # Calculate detailed statistics for this disease
        df_disease_all = df_gwas_agg[df_gwas_agg['normalized_trait'] == disease]
        df_disease_sig = df_sig[df_sig['normalized_trait'] == disease] if len(df_sig) > 0 else pd.DataFrame()
        
        n_total_assoc = len(df_disease_all)
        n_total_snps = df_disease_all['rsid'].nunique()
        n_sig_assoc = len(df_disease_sig)
        n_sig_snps = df_disease_sig['rsid'].nunique() if len(df_disease_sig) > 0 else 0
        
        # Create detailed legend label: disease: total assoc, total SNPs, sig assoc, sig SNPs
        #legend_label = f'{disease}: {n_total_assoc} assoc, {n_total_snps} SNPs, {n_sig_assoc} sig. assoc, {n_sig_snps} sig. SNPs'
        legend_label = f'{disease}: {n_total_snps} SNPs, {n_sig_snps} sig. SNPs'

        
        ax2.scatter([], [], c=color, s=args.point_size * 2, alpha=0.6,
                   edgecolors='none', label=legend_label)
    
    # 2. Then plot "Other" categories using chromosome-based alternating colors
    df_others_nonsig = df_nonsig[~df_nonsig['normalized_trait'].isin(main_diseases)]
    if len(df_others_nonsig) > 0:
        for i, chrom in enumerate(chr_order):
            df_chr_others_nonsig = df_others_nonsig[df_others_nonsig['chromosome'] == chrom]
            if len(df_chr_others_nonsig) == 0:
                continue
            
            color = colors_main[i % 2]
            
            # Plot as NEGATIVE values to create mirror effect
            ax2.scatter(df_chr_others_nonsig['cumulative_pos'], 
                       -df_chr_others_nonsig['sqrt_log10_pval'],  # Negative for mirror, sqrt transformed
                       c=color, 
                       s=args.point_size * 2, 
                       alpha=0.6,
                       edgecolors='none',
                       rasterized=True)
        
        # Add a detailed legend entry for "Other" categories with statistics
        df_others_all = df_gwas_agg[~df_gwas_agg['normalized_trait'].isin(main_diseases)]
        df_others_sig = df_sig[~df_sig['normalized_trait'].isin(main_diseases)] if len(df_sig) > 0 else pd.DataFrame()
        
        n_total_assoc_others = len(df_others_all)
        n_total_snps_others = df_others_all['rsid'].nunique()
        n_sig_assoc_others = len(df_others_sig)
        n_sig_snps_others = df_others_sig['rsid'].nunique() if len(df_others_sig) > 0 else 0
        
        other_traits = df_others_nonsig['normalized_trait'].unique()
        #other_label = f'Other ({", ".join(other_traits)}): {n_total_assoc_others} assoc, {n_total_snps_others} SNPs, {n_sig_assoc_others} sig. assoc, {n_sig_snps_others} sig. SNPs'
        other_label = f'Other ({", ".join(other_traits)}): {n_sig_snps_others} sig. SNPs'

        ax2.scatter([], [], c='gray', s=args.point_size * 2, alpha=0.6,
                   label=other_label)
    
    # Plot significant SNPs in DIAMONDS - HYBRID COLORING
    if len(df_sig) > 0:
        # Calculate unique SNP counts
        n_unique_snps_total = df_gwas_agg['rsid'].nunique()
        n_unique_snps_sig = df_sig['rsid'].nunique()
        
        # 1. Plot significant SNPs for the 5 main diseases with disease-specific colors
        for disease, color in disease_colors.items():
            df_disease_sig = df_sig[df_sig['normalized_trait'] == disease]
            if len(df_disease_sig) == 0:
                continue
            
            # Plot as DISEASE-COLORED DIAMONDS
            ax2.scatter(df_disease_sig['cumulative_pos'], 
                       -df_disease_sig['sqrt_log10_pval'],  # sqrt transformed
                       c=color, 
                       s=args.point_size * 3,  # Larger for visibility
                       marker='D',  # Diamond shape
                       edgecolors='black',  # Black edge for better contrast
                       linewidths=0.8,
                       alpha=0.9,
                       zorder=100)
        
        # 2. Plot significant SNPs for "Other" categories using chromosome-based colors
        df_others_sig = df_sig[~df_sig['normalized_trait'].isin(main_diseases)]
        if len(df_others_sig) > 0:
            for i, chrom in enumerate(chr_order):
                df_chr_others_sig = df_others_sig[df_others_sig['chromosome'] == chrom]
                if len(df_chr_others_sig) == 0:
                    continue
                
                color = colors_main[i % 2]
                
                # Plot as CHROMOSOME-COLORED DIAMONDS for "Other" categories
                ax2.scatter(df_chr_others_sig['cumulative_pos'], 
                           -df_chr_others_sig['sqrt_log10_pval'],  # sqrt transformed
                           c=color, 
                           s=args.point_size * 3,  # Larger for visibility
                           marker='D',  # Diamond shape
                           edgecolors='black',  # Black edge for better contrast
                           linewidths=0.8,
                           alpha=0.9,
                           zorder=100)
        
        # Add a single legend entry for all significant associations
        ax2.scatter([], [], c='black', s=args.point_size * 3, marker='D',
                   edgecolors='black', linewidths=0.8,
                   label=f'Significant (p<5×10^-8): {len(df_sig):,} associations ({n_unique_snps_sig:,} unique SNPs)')
        
        # Add labels to significant SNPs with rsID and p-value
        if args.label_significant and args.label_top_significant_n != -1:
            # Determine how many to label
            if args.label_top_significant_n == 0:
                # Label all significant SNPs (but warn if too many)
                df_to_label = df_sig
                if len(df_to_label) > 50:
                    print(f"  ⚠ Labeling all {len(df_to_label)} significant associations may cause clutter")
            else:
                # Label only top N most significant (lowest p-values)
                df_to_label = df_sig.nsmallest(args.label_top_significant_n, 'pval')
                if len(df_sig) > args.label_top_significant_n:
                    print(f"  • Labeling top {args.label_top_significant_n} most significant SNPs (out of {len(df_sig)} total)")
            
            # Add labels with disease-colored backgrounds for main diseases, gray for others
            for idx, row in df_to_label.iterrows():
                label_text = f"{row['rsid']}, p={row['pval']:.2e}"
                
                # Get disease color for this association
                disease = row['normalized_trait']
                if disease in disease_colors:
                    # Use disease-specific color for main diseases
                    bg_color = disease_colors[disease]
                else:
                    # Use gray for "Other" categories
                    bg_color = '#7f7f7f'
                
                ax2.annotate(label_text, 
                            (row['cumulative_pos'], -row['sqrt_log10_pval']),  # sqrt transformed
                            fontsize=7,
                            alpha=0.9,
                            fontweight='bold',
                            color='white',  # White text for better contrast on colored backgrounds
                            xytext=(5, 5),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor=bg_color, 
                                     alpha=0.7,
                                     edgecolor='black',
                                     linewidth=0.5))
        elif args.label_significant and args.label_top_significant_n == -1:
            print(f"  • Labeling disabled (label_top_significant_n=-1)")
    else:
        # No significant SNPs - add legend entries with full statistics
        # Add entries for each of the 5 main diseases
        for disease, color in disease_colors.items():
            df_disease = df_gwas_agg[df_gwas_agg['normalized_trait'] == disease]
            if len(df_disease) > 0:
                n_total_assoc = len(df_disease)
                n_total_snps = df_disease['rsid'].nunique()
                # No significant associations in this case
                #legend_label = f'{disease}: {n_total_assoc} assoc, {n_total_snps} SNPs, 0 sig. assoc, 0 sig. SNPs'
                legend_label = f'{disease}: {n_total_snps} SNPs, 0 sig. SNPs'

                ax2.scatter([], [], c=color, s=args.point_size * 2, alpha=0.6,
                           label=legend_label)
        
        # Add entry for "Other" categories if they exist with detailed statistics
        df_others = df_gwas_agg[~df_gwas_agg['normalized_trait'].isin(main_diseases)]
        if len(df_others) > 0:
            n_total_assoc_others = len(df_others)
            n_total_snps_others = df_others['rsid'].nunique()
            
            other_traits = df_others['normalized_trait'].unique()
            #other_label = f'Other ({", ".join(other_traits)}): {n_total_assoc_others} assoc, {n_total_snps_others} SNPs, 0 sig. assoc, 0 sig. SNPs'
            other_label = f'Other ({", ".join(other_traits)}): 0 sig. SNPs'

            ax2.scatter([], [], c='gray', s=args.point_size * 2, alpha=0.6,
                       label=other_label)
        
        n_unique_snps_total = df_gwas_agg['rsid'].nunique()
        ax2.scatter([], [], c='black', s=args.point_size * 3, marker='D',
                   label=f'No significant associations (p<5×10^-8)')
    
    # *** MODIFICATION: Calculate sqrt-transformed threshold ***
    threshold_y = -np.sqrt(-np.log10(GWAS_SIGNIFICANCE))
    
    ax2.axhline(y=threshold_y, color='red', 
                linestyle='--', linewidth=1.5, alpha=0.8,
                label=f'Genome-wide threshold (p=5×10^-8)', zorder=10)
    
    # Add total associations count to legend
    n_unique_snps_total = df_gwas_agg['rsid'].nunique()
    ax2.scatter([], [], alpha=0, s=0,
               label=f'Total: {len(df_gwas_agg):,} associations ({n_unique_snps_total:,} unique SNPs)')
    
    # Styling for bottom panel
    # *** MODIFICATION: Updated y-axis label to indicate sqrt transformation ***
    ax2.set_ylabel('√[-log₁₀(p-value)] [Mirrored]', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Chromosome', fontsize=13, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend for bottom panel - upper left corner INSIDE the plot
    ax2.legend(loc='upper left', frameon=True, fontsize=8,
              framealpha=0.95, edgecolor='black', fancybox=False)
    
    # Use scientific notation for y-axis
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Invert y-axis to create true mirror effect
    ax2.invert_yaxis()
    
    # Add subtle chromosome separators
    for chrom in chr_order[:-1]:
        if chrom in chr_boundaries:
            boundary = chr_boundaries[chrom][1]
            ax2.axvline(x=boundary, color='gray', linestyle='-', linewidth=0.3, alpha=0.3, zorder=0)
    
    # ==============================================================
    # X-axis: Chromosome labels (shared)
    # ==============================================================
    ax2.set_xticks([chr_midpoints[chrom] for chrom in chr_order if chrom in chr_midpoints])
    ax2.set_xticklabels(chr_order, fontsize=10)
    ax2.set_xlim(0, df_importance['cumulative_pos'].max())
    
    # Add light background shading for alternating chromosomes
    for i, chrom in enumerate(chr_order):
        if chrom not in chr_boundaries:
            continue
        start, end = chr_boundaries[chrom]
        if i % 2 == 0:
            ax1.axvspan(start, end, facecolor='gray', alpha=0.03, zorder=0)
            ax2.axvspan(start, end, facecolor='gray', alpha=0.03, zorder=0)
    
    # Save
    save_kwargs = {'dpi': args.dpi, 'bbox_inches': 'tight'} if args.plot_format == 'png' else {'bbox_inches': 'tight'}
    plt.savefig(output_path, format=args.plot_format, **save_kwargs)
    plt.close()
    
    print(f"  ✓ Saved dual Manhattan plot with sqrt-transformed bottom panel: {output_path}")


def plot_bottom_panel_standalone(df_gwas_agg, chr_order, chr_midpoints, chr_boundaries, 
                                  output_path, args, disease_colors, main_diseases, colors_main,
                                  df_significant=None, entity_name='SNP'):
    """
    Create standalone bottom panel Manhattan plot (GWAS p-values only).
    Uses the same styling as the dual plot bottom panel.
    Uses full height for better readability.
    Legend includes total associations count (X associations, Y unique SNPs).
    Only shows the 5 main diseases (excludes "Other" categories for visual clarity).
    """
    print("\n  • Creating standalone bottom panel plot (5 main diseases only)...")
    
    # Filter to show only the 5 main diseases (exclude "Other" categories)
    df_gwas_agg = df_gwas_agg[df_gwas_agg['normalized_trait'].isin(main_diseases)].copy()
    if df_significant is not None:
        df_significant = df_significant[df_significant['normalized_trait'].isin(main_diseases)].copy()
    
    print(f"    - Filtered to {len(df_gwas_agg):,} associations from 5 main diseases")
    
    figsize = tuple(map(float, args.figsize.split(',')))
    # Use full height for standalone plot (same as dual plot total height)
    fig = plt.figure(figsize=(figsize[0], figsize[1]), constrained_layout=False)
    
    # Create single axis
    ax = fig.add_subplot(111)
    
    # Calculate sqrt transformation
    df_gwas_agg['-log10_pval'] = -np.log10(df_gwas_agg['pval'] + 1e-300)
    df_gwas_agg['sqrt_log10_pval'] = np.sqrt(df_gwas_agg['-log10_pval'])
    
    # Identify significant SNPs
    df_sig = df_gwas_agg[df_gwas_agg['pval'] < GWAS_SIGNIFICANCE].copy()
    df_nonsig = df_gwas_agg[df_gwas_agg['pval'] >= GWAS_SIGNIFICANCE].copy()
    
    # Plot non-significant SNPs - HYBRID APPROACH
    # 1. First plot the 5 main diseases with disease-specific colors
    for disease, color in disease_colors.items():
        df_disease_nonsig = df_nonsig[df_nonsig['normalized_trait'] == disease]
        if len(df_disease_nonsig) == 0:
            continue
        
        ax.scatter(df_disease_nonsig['cumulative_pos'], 
                   -df_disease_nonsig['sqrt_log10_pval'],
                   c=color, 
                   s=args.point_size * 2, 
                   alpha=0.6,
                   edgecolors='none',
                   rasterized=True)
        
        # Calculate detailed statistics for this disease
        df_disease_all = df_gwas_agg[df_gwas_agg['normalized_trait'] == disease]
        df_disease_sig = df_sig[df_sig['normalized_trait'] == disease] if len(df_sig) > 0 else pd.DataFrame()
        
        n_total_assoc = len(df_disease_all)
        n_total_snps = df_disease_all['rsid'].nunique()
        n_sig_assoc = len(df_disease_sig)
        n_sig_snps = df_disease_sig['rsid'].nunique() if len(df_disease_sig) > 0 else 0
        
        # Create detailed legend label: disease: total assoc, total SNPs, sig assoc, sig SNPs
        #legend_label = f'{disease}: {n_total_assoc} assoc, {n_total_snps} SNPs, {n_sig_assoc} sig. assoc, {n_sig_snps} sig. SNPs'
        legend_label = f'{disease}: {n_total_snps} SNPs, {n_sig_snps} sig. SNPs'

        
        ax.scatter([], [], c=color, s=args.point_size * 2, alpha=0.6,
                   edgecolors='none', label=legend_label)
    
    # Plot significant SNPs in DIAMONDS - colored by disease
    if len(df_sig) > 0:
        n_unique_snps_sig = df_sig['rsid'].nunique()
        
        # Plot significant SNPs for the 5 main diseases with disease-specific colors
        for disease, color in disease_colors.items():
            df_disease_sig = df_sig[df_sig['normalized_trait'] == disease]
            if len(df_disease_sig) == 0:
                continue
            
            ax.scatter(df_disease_sig['cumulative_pos'], 
                       -df_disease_sig['sqrt_log10_pval'],
                       c=color, 
                       s=args.point_size * 3,
                       marker='D',
                       edgecolors='black',
                       linewidths=0.8,
                       alpha=0.9,
                       zorder=100)
        
        # Add legend entry for significant associations
        ax.scatter([], [], c='black', s=args.point_size * 3, marker='D',
                   edgecolors='black', linewidths=0.8,
                   label=f'Significant (p<5×10^-8): {len(df_sig):,} associations ({n_unique_snps_sig:,} unique SNPs)')
        
        # Add labels if requested
        if args.label_significant and args.label_top_significant_n != -1:
            if args.label_top_significant_n == 0:
                df_to_label = df_sig
            else:
                df_to_label = df_sig.nsmallest(args.label_top_significant_n, 'pval')
            
            for idx, row in df_to_label.iterrows():
                label_text = f"{row['rsid']}, p={row['pval']:.2e}"
                disease = row['normalized_trait']
                bg_color = disease_colors.get(disease, '#7f7f7f')
                
                ax.annotate(label_text, 
                            (row['cumulative_pos'], -row['sqrt_log10_pval']),
                            fontsize=7,
                            alpha=0.9,
                            fontweight='bold',
                            color='white',
                            xytext=(5, 5),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3',
                                     facecolor=bg_color, 
                                     alpha=0.7,
                                     edgecolor='black',
                                     linewidth=0.5))
    else:
        # No significant SNPs - add legend entries with full statistics
        for disease, color in disease_colors.items():
            df_disease = df_gwas_agg[df_gwas_agg['normalized_trait'] == disease]
            if len(df_disease) > 0:
                n_total_assoc = len(df_disease)
                n_total_snps = df_disease['rsid'].nunique()
                # No significant associations in this case
                #legend_label = f'{disease}: {n_total_assoc} assoc, {n_total_snps} SNPs, 0 sig. assoc, 0 sig. SNPs'
                legend_label = f'{disease}: {n_total_snps} SNPs, 0 sig. SNPs'

                ax.scatter([], [], c=color, s=args.point_size * 2, alpha=0.6,
                           label=legend_label)
        
        ax.scatter([], [], c='black', s=args.point_size * 3, marker='D',
                   label=f'No significant associations (p<5×10^-8)')
    
    # Add threshold line
    threshold_y = -np.sqrt(-np.log10(GWAS_SIGNIFICANCE))
    ax.axhline(y=threshold_y, color='red', 
                linestyle='--', linewidth=1.5, alpha=0.8,
                label=f'Genome-wide threshold (p=5×10^-8)', zorder=10)
    
    # Add total associations count to legend
    n_unique_snps_total = df_gwas_agg['rsid'].nunique()
    ax.scatter([], [], alpha=0, s=0,
               label=f'Total: {len(df_gwas_agg):,} associations ({n_unique_snps_total:,} unique SNPs)')
    
    # Styling
    ax.set_ylabel('√[-log₁₀(p-value)] [Mirrored]', fontsize=13, fontweight='bold')
    ax.set_xlabel('Chromosome', fontsize=13, fontweight='bold')
    ax.set_title(f'{entity_name} - GWAS Association Manhattan Plot', 
                  fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend in upper left corner
    ax.legend(loc='upper left', frameon=True, fontsize=8,
              framealpha=0.95, edgecolor='black', fancybox=False)
    
    # Invert y-axis for mirror effect
    ax.invert_yaxis()
    
    # X-axis: Chromosome labels
    ax.set_xticks([chr_midpoints[chrom] for chrom in chr_order if chrom in chr_midpoints])
    ax.set_xticklabels(chr_order, fontsize=10)
    ax.set_xlim(0, df_gwas_agg['cumulative_pos'].max())
    
    # Add chromosome separators and shading
    for i, chrom in enumerate(chr_order):
        if chrom not in chr_boundaries:
            continue
        start, end = chr_boundaries[chrom]
        if i % 2 == 0:
            ax.axvspan(start, end, facecolor='gray', alpha=0.03, zorder=0)
        if i < len(chr_order) - 1:
            ax.axvline(x=end, color='gray', linestyle='-', linewidth=0.3, alpha=0.3, zorder=0)
    
    # Save
    save_kwargs = {'dpi': args.dpi, 'bbox_inches': 'tight'} if args.plot_format == 'png' else {'bbox_inches': 'tight'}
    plt.savefig(output_path, format=args.plot_format, **save_kwargs)
    plt.close()
    
    print(f"  ✓ Saved standalone bottom panel plot: {output_path}")


def create_summary_statistics(df_importance, df_lookup, df_gwas_agg, df_significant):
    """Generate summary statistics for the analysis"""
    
    stats = {
        'Total SNPs in importance data': len(df_importance),
        'Total GWAS associations found (raw)': len(df_lookup),
        'Unique SNP-normalized_trait combinations (after deduplication)': len(df_gwas_agg),
        'Unique SNPs with GWAS associations': df_gwas_agg['rsid'].nunique() if len(df_gwas_agg) > 0 else 0,
        'Average normalized traits per SNP': df_gwas_agg.groupby('rsid')['normalized_trait'].nunique().mean() if len(df_gwas_agg) > 0 else 0,
        'Genome-wide significant associations (p<5e-8)': len(df_significant),
        'Unique SNPs with genome-wide significance': df_significant['rsid'].nunique() if len(df_significant) > 0 else 0,
    }
    
    return stats


def main():
    args = parse_args()
    
    print("="*80)
    print("DUAL MANHATTAN PLOT GENERATOR - HYBRID COLOR VERSION")
    print("="*80)
    print("FEATURES: sqrt(-log10(p)) transformation + Hybrid coloring scheme")
    print("  - Top panel: Red diamonds = GWAS-significant (p<5e-8)")
    print("  - Bottom panel: Disease colors for 5 main diseases, chromosome colors for others")
    print("  - Diamond markers colored appropriately for significant associations")
    print("="*80)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==============================================================
    # LOAD DATA
    # ==============================================================
    print("\n[1] Loading data...")
    
    # Load importance scores
    df_importance = pd.read_csv(args.importance)
    print(f"  ✓ Loaded importance scores: {len(df_importance):,} SNPs")
    print(f"    Columns: {list(df_importance.columns)}")
    
    # Verify required columns
    required_cols = ['chromosome', 'bp', 'Importance_Score']
    missing_cols = [col for col in required_cols if col not in df_importance.columns]
    if missing_cols:
        raise ValueError(f"Importance file missing required columns: {missing_cols}")
    
    # Filter by minimum importance if specified
    if args.min_importance is not None:
        df_importance = df_importance[df_importance['Importance_Score'] >= args.min_importance]
        print(f"  ✓ Filtered to {len(df_importance):,} SNPs with importance >= {args.min_importance}")
    
    # Load GWAS lookup results
    df_lookup = pd.read_csv(args.lookup)
    print(f"  ✓ Loaded GWAS associations: {len(df_lookup):,} associations")
    print(f"    Unique SNPs: {df_lookup['rsid'].nunique():,}")
    
    # Verify minimum required columns
    if 'rsid' not in df_lookup.columns or 'pval' not in df_lookup.columns:
        raise ValueError("Lookup file must contain 'rsid' and 'pval' columns")
    
    # Check if chromosome/bp are in GWAS file
    has_chr_bp = 'chromosome' in df_lookup.columns and 'bp' in df_lookup.columns
    if has_chr_bp:
        print(f"    ✓ GWAS file contains chromosome and bp columns")
    else:
        print(f"    ⚠ GWAS file missing chromosome/bp - will use from importance data for overlapping SNPs")
    
    # ==============================================================
    # DEDUPLICATE ASSOCIATIONS (CRITICAL: DO THIS FIRST!)
    # ==============================================================
    print(f"\n[2] Deduplicating multiple associations per SNP...")
    
    # Check for missing chromosome/bp data (only if these columns exist)
    if 'chromosome' in df_lookup.columns and 'bp' in df_lookup.columns:
        missing_chr = df_lookup['chromosome'].isna().sum()
        missing_bp = df_lookup['bp'].isna().sum()
        if missing_chr > 0 or missing_bp > 0:
            print(f"  ⚠ Warning: Found {missing_chr} associations missing chromosome and {missing_bp} missing bp")
            print(f"    Dropping these associations as they cannot be plotted...")
            df_lookup = df_lookup.dropna(subset=['chromosome', 'bp'])
            print(f"    Remaining: {len(df_lookup):,} associations")
    
    df_gwas_agg = aggregate_multiple_associations(df_lookup)
    print(f"  ✓ Original associations: {len(df_lookup):,}")
    print(f"  ✓ After deduplication: {len(df_gwas_agg):,} (removed {len(df_lookup) - len(df_gwas_agg):,} duplicate trait-SNP pairs)")
    
    # Note: We keep ALL associations for visualization (no p-value filtering for display)
    # Significant ones will be highlighted in yellow in the plot
    print(f"  ✓ Total associations for visualization: {len(df_gwas_agg):,}")
    
    # ==============================================================
    # FILTER SIGNIFICANT ASSOCIATIONS (FROM DEDUPLICATED DATA!)
    # ==============================================================
    print(f"\n[3] Filtering genome-wide significant associations (p < {GWAS_SIGNIFICANCE})...")
    
    # Filter from deduplicated data
    df_significant = filter_significant_associations(df_gwas_agg, GWAS_SIGNIFICANCE)
    
    # Get base name from lookup file for output naming
    lookup_basename = Path(args.lookup).stem
    
    if len(df_significant) > 0:
        print(f"  ✓ Found {len(df_significant):,} significant associations (p < {GWAS_SIGNIFICANCE})")
        print(f"    Affecting {df_significant['rsid'].nunique():,} unique SNPs")
        print(f"    Across {df_significant['normalized_trait'].nunique():,} normalized trait categories")
        
        # Show distribution
        print(f"    Distribution:")
        for trait, count in df_significant['normalized_trait'].value_counts().items():
            print(f"      - {trait}: {count:,}")
        
        # Save all significant associations (one row per rsID-trait with p-value)
        sig_output = output_dir / f"{lookup_basename}_significant_associations_p5e-8.csv"
        df_significant.to_csv(sig_output, index=False)
        print(f"  ✓ Saved: {sig_output}")
        
        # Create a summary of significant SNPs (one row per rsID-normalized_trait combination)
        cols_to_keep = ['rsid', 'normalized_trait', 'trait', 'pval', 'category']
        sig_summary = df_significant[[col for col in cols_to_keep if col in df_significant.columns]].copy()
        
        # Add count of how many normalized traits each SNP is associated with
        trait_counts = sig_summary.groupby('rsid')['normalized_trait'].count().reset_index(name='n_significant_traits')
        sig_summary = sig_summary.merge(trait_counts, on='rsid', how='left')
        
        # Add importance scores and genomic position if available
        # Use left join to keep all significant SNPs even if they don't have importance scores
        if 'rsid' in df_importance.columns or 'snp_id' in df_importance.columns:
            rsid_col = 'rsid' if 'rsid' in df_importance.columns else 'snp_id'
            importance_subset = df_importance[[rsid_col, 'Importance_Score', 'chromosome', 'bp']].rename(columns={rsid_col: 'rsid'})
            sig_summary = sig_summary.merge(
                importance_subset,
                on='rsid',
                how='left'  # Keep all significant SNPs even without importance scores
            )
        
        # Add chromosome/bp from GWAS data if not already present
        if 'chromosome' not in sig_summary.columns and 'chromosome' in df_significant.columns:
            sig_summary = sig_summary.merge(
                df_significant[['rsid', 'chromosome', 'bp']],
                on='rsid',
                how='left'
            )
        
        # Sort by p-value
        sig_summary = sig_summary.sort_values('pval')
        
        # Reorder columns for better readability
        cols_order = ['rsid', 'chromosome', 'bp', 'normalized_trait', 'trait', 'pval', 'category', 'n_significant_traits']
        if 'Importance_Score' in sig_summary.columns:
            cols_order.insert(4, 'Importance_Score')
        sig_summary = sig_summary[[col for col in cols_order if col in sig_summary.columns]]
        
        sig_summary_output = output_dir / f"{lookup_basename}_significant_snps_summary.csv"
        sig_summary.to_csv(sig_summary_output, index=False)
        print(f"  ✓ Saved summary: {sig_summary_output}")
        print(f"    Format: One row per unique rsID-normalized_trait association with its p-value")
    else:
        print("  ⚠ No genome-wide significant associations found")
        df_significant = pd.DataFrame()
    
    # ==============================================================
    # MERGE IMPORTANCE AND GWAS DATA
    # ==============================================================
    print("\n[4] Merging importance scores with GWAS data...")
    
    # Ensure we have rsid column in importance data
    if 'snp_id' in df_importance.columns and 'rsid' not in df_importance.columns:
        df_importance['rsid'] = df_importance['snp_id']
    elif 'rsid' not in df_importance.columns:
        print("  ⚠ Warning: No rsid/snp_id column in importance data")
        df_importance['rsid'] = 'unknown'
    
    # Determine which columns to merge from importance data
    # ALWAYS merge chromosome and bp from importance data to ensure we have them
    merge_cols = ['rsid', 'chromosome', 'bp', 'Importance_Score']
    if 'snp_id' in df_importance.columns and 'rsid' != 'snp_id':
        merge_cols.append('snp_id')
    
    # If GWAS already has chromosome/bp, we need to handle the merge differently
    has_chr_in_gwas = 'chromosome' in df_gwas_agg.columns
    has_bp_in_gwas = 'bp' in df_gwas_agg.columns
    
    # LEFT JOIN: Keep all GWAS SNPs, add importance scores + chromosome/bp where available
    df_gwas_agg = df_gwas_agg.merge(
        df_importance[merge_cols],
        on='rsid',
        how='left',
        suffixes=('_gwas', '_importance')
    )
    
    # Handle chromosome column
    if 'chromosome_importance' in df_gwas_agg.columns:
        if has_chr_in_gwas:
            # Use GWAS chromosome if available, otherwise use importance
            df_gwas_agg['chromosome'] = df_gwas_agg['chromosome_gwas'].fillna(df_gwas_agg['chromosome_importance'])
            df_gwas_agg = df_gwas_agg.drop(columns=['chromosome_gwas', 'chromosome_importance'])
        else:
            # Only have importance chromosome
            df_gwas_agg['chromosome'] = df_gwas_agg['chromosome_importance']
            df_gwas_agg = df_gwas_agg.drop(columns=['chromosome_importance'])
    
    # Handle bp column
    if 'bp_importance' in df_gwas_agg.columns:
        if has_bp_in_gwas:
            # Use GWAS bp if available, otherwise use importance
            df_gwas_agg['bp'] = df_gwas_agg['bp_gwas'].fillna(df_gwas_agg['bp_importance'])
            df_gwas_agg = df_gwas_agg.drop(columns=['bp_gwas', 'bp_importance'])
        else:
            # Only have importance bp
            df_gwas_agg['bp'] = df_gwas_agg['bp_importance']
            df_gwas_agg = df_gwas_agg.drop(columns=['bp_importance'])
    
    # Handle Importance_Score column (might have suffix if GWAS had chromosome/bp)
    if 'Importance_Score_importance' in df_gwas_agg.columns:
        # Merge added suffix because GWAS had chromosome/bp
        df_gwas_agg['Importance_Score'] = df_gwas_agg['Importance_Score_importance']
        df_gwas_agg = df_gwas_agg.drop(columns=['Importance_Score_importance'])
    # If it doesn't exist at all, that's fine - means no overlap with importance data
    
    # Drop SNPs that still don't have chromosome/bp (can't be plotted)
    before_drop = len(df_gwas_agg)
    df_gwas_agg = df_gwas_agg.dropna(subset=['chromosome', 'bp'])
    after_drop = len(df_gwas_agg)
    
    if before_drop > after_drop:
        print(f"  ⚠ Dropped {before_drop - after_drop:,} associations without chromosome/bp information")
    
    # Count overlaps (safely check if column exists)
    if 'Importance_Score' in df_gwas_agg.columns:
        has_importance = df_gwas_agg['Importance_Score'].notna().sum()
        no_importance = df_gwas_agg['Importance_Score'].isna().sum()
    else:
        has_importance = 0
        no_importance = len(df_gwas_agg)
    
    print(f"  ✓ Total GWAS associations with valid positions: {len(df_gwas_agg):,}")
    print(f"    - With importance scores: {has_importance:,}")
    print(f"    - Without importance scores: {no_importance:,}")
    
    if len(df_gwas_agg) == 0:
        print("\n  ERROR: No GWAS associations after merge!")
        print("  This means no GWAS SNPs have chromosome/bp information.")
        print("  Check that:")
        print("    1. GWAS file has chromosome/bp columns, OR")
        print("    2. GWAS rsIDs overlap with importance file rsIDs")
        return
    
    # ==============================================================
    # PREPARE CHROMOSOME DATA FOR MANHATTAN PLOT
    # ==============================================================
    print("\n[5] Preparing Manhattan plot data...")
    
    # Check data quality
    print(f"  • Importance data: {len(df_importance)} SNPs across {df_importance['chromosome'].nunique()} chromosomes")
    print(f"  • GWAS data (before cleaning): {len(df_gwas_agg)} SNPs across {df_gwas_agg['chromosome'].nunique()} chromosomes")
    
    # Show sample chromosome names for debugging
    gwas_chrs_before = sorted(df_gwas_agg['chromosome'].unique(), key=lambda x: (len(str(x)), str(x)))[:5]
    print(f"    Sample GWAS chromosomes (before): {gwas_chrs_before}")
    
    # Prepare chromosome data for importance (calculates cumulative positions)
    df_importance, chr_order, chr_midpoints, chr_boundaries = prepare_chromosome_data(df_importance)
    
    print(f"    Importance chromosomes (after cleaning): {chr_order[:5]}...") if len(chr_order) > 5 else print(f"    Importance chromosomes: {chr_order}")
    
    # CRITICAL: Clean chromosome names in GWAS data to match importance data
    # The prepare_chromosome_data function cleans chromosomes for importance data,
    # but we need to do the same for GWAS data
    print(f"  • Cleaning chromosome names in GWAS data to match importance data...")
    
    # Force to string and clean (handles both numeric and string inputs)
    df_gwas_agg['chromosome'] = df_gwas_agg['chromosome'].astype(str).str.strip().str.replace('chr', '', case=False)
    
    gwas_chrs_after = sorted(df_gwas_agg['chromosome'].unique(), key=lambda x: (len(str(x)), str(x)))[:5]
    print(f"    Sample GWAS chromosomes (after): {gwas_chrs_after}")
    print(f"    GWAS chromosome dtype: {df_gwas_agg['chromosome'].dtype}")
    print(f"    chr_order dtype: {type(chr_order[0]) if chr_order else 'N/A'}")
    
    # Double-check: count how many GWAS chromosomes are in chr_order
    gwas_chr_set = set(df_gwas_agg['chromosome'].unique())
    chr_order_set = set(chr_order)
    overlap = gwas_chr_set & chr_order_set
    print(f"    Chromosomes in both: {len(overlap)} / {len(gwas_chr_set)}")
    
    # CRITICAL FIX: Calculate cumulative positions for ALL GWAS SNPs independently
    # This ensures we plot all associations, not just those in importance data
    print(f"  • Calculating cumulative positions for all GWAS SNPs...")
    print(f"    chr_order type: {type(chr_order[0]) if chr_order else 'empty'}, sample: {chr_order[:3]}")
    print(f"    GWAS chr type: {df_gwas_agg['chromosome'].dtype}, sample: {list(df_gwas_agg['chromosome'].unique())[:3]}")
    
    # Use the same chromosome boundaries from importance data
    df_gwas_agg['cumulative_pos'] = 0.0
    
    positioned_count = 0
    skipped_chromosomes = []
    matched_chromosomes = []
    
    for chrom in chr_order:
        # Get GWAS SNPs for this chromosome
        gwas_chr_mask = df_gwas_agg['chromosome'] == chrom
        gwas_chr_count = gwas_chr_mask.sum()
        
        if gwas_chr_count == 0:
            continue
        else:
            matched_chromosomes.append(f"{chrom}({gwas_chr_count})")
        
        # Get importance data for this chromosome to establish the coordinate system
        imp_chr_data = df_importance[df_importance['chromosome'] == chrom]
        
        if len(imp_chr_data) == 0:
            skipped_chromosomes.append(chrom)
            print(f"    ⚠ Chromosome {chrom}: has {gwas_chr_count} GWAS SNPs but no importance data - cannot position these SNPs")
            continue
        
        # Use the min bp from importance data as the baseline for this chromosome
        chr_min_bp = imp_chr_data['bp'].min()
        
        # Get the cumulative offset for this chromosome from chr_boundaries
        chr_start = chr_boundaries[chrom][0]
        
        # Calculate cumulative positions for GWAS SNPs
        df_gwas_agg.loc[gwas_chr_mask, 'cumulative_pos'] = (
            df_gwas_agg.loc[gwas_chr_mask, 'bp'] - chr_min_bp + chr_start
        )
        
        positioned_count += gwas_chr_count
    
    # Show what was matched
    if matched_chromosomes:
        print(f"    Matched chromosomes: {', '.join(matched_chromosomes[:10])}")
        if len(matched_chromosomes) > 10:
            print(f"    ... and {len(matched_chromosomes) - 10} more")
    else:
        print(f"    ⚠ NO CHROMOSOMES MATCHED!")
        print(f"    chr_order: {chr_order}")
        print(f"    GWAS unique chromosomes: {sorted(df_gwas_agg['chromosome'].unique())}")
    
    # Remove any SNPs that couldn't be positioned
    unpositioned = (df_gwas_agg['cumulative_pos'] == 0) & (df_gwas_agg['bp'] > 0)
    if unpositioned.any():
        print(f"  ⚠ Warning: {unpositioned.sum()} GWAS SNPs couldn't be positioned. Dropping them.")
        if skipped_chromosomes:
            print(f"    Affected chromosomes: {', '.join(skipped_chromosomes)}")
        df_gwas_agg = df_gwas_agg[~unpositioned]
    
    # Sort by chromosome order for plotting
    df_gwas_agg['chr_order'] = pd.Categorical(df_gwas_agg['chromosome'], categories=chr_order, ordered=True)
    df_gwas_agg = df_gwas_agg.sort_values(['chr_order', 'cumulative_pos']).reset_index(drop=True)
    
    print(f"  ✓ Chromosomes found: {', '.join(chr_order)}")
    print(f"  ✓ Final GWAS data for plotting: {len(df_gwas_agg):,} associations")
    
    if positioned_count > 0:
        print(f"    - Successfully positioned: {positioned_count:,} SNPs")
    if skipped_chromosomes:
        print(f"    - Skipped chromosomes (no importance data): {', '.join(skipped_chromosomes)}")
    
    if len(df_gwas_agg) == 0:
        print("\n  ERROR: No GWAS associations can be plotted!")
        print("  Possible reasons:")
        print("    1. GWAS file doesn't have chromosome/bp columns")
        print("    2. No overlap between GWAS rsIDs and importance rsIDs")
        print("    3. All GWAS SNPs filtered out due to missing data")
        return
    
    # Show breakdown
    gwas_rsids = set(df_gwas_agg['rsid'].dropna())
    importance_rsids = set(df_importance['rsid'].dropna())
    overlap = gwas_rsids & importance_rsids
    print(f"  ✓ SNP overlap summary:")
    print(f"    - GWAS-only SNPs: {len(gwas_rsids - overlap):,}")
    print(f"    - Overlapping SNPs: {len(overlap):,}")
    print(f"    - Importance-only SNPs: {len(importance_rsids - overlap):,}")
    
    # ==============================================================
    # CREATE DUAL MANHATTAN PLOT WITH NORMALIZED BOTTOM PANEL
    # ==============================================================
    print("\n[6] Creating dual Manhattan plot with hybrid-colored bottom panel...")
    print("  • Visualization approach:")
    print("    - Top panel: All SNPs with importance scores (alternating colors by chromosome)")
    if args.highlight_gwas_sig:
        print("      * GWAS-significant SNPs (p<5e-8): Red diamonds with pink labels")
    if args.highlight_top_n > 0:
        print("      * Top N importance SNPs: Yellow labels")
    print("    - Bottom panel: ALL GWAS associations (SQRT TRANSFORMED)")
    print("      * 5 main diseases: Disease-specific colors (Yellow=T2DM, Orange=Breast, Cyan=Pancreatic, Green=Prostate, Magenta=Colorectal)")
    print("      * Other categories: Alternating blue/gray by chromosome")
    print("      * Significant (p < 5e-8): Disease-colored diamonds (or chromosome-colored for 'Other')")
    print("      * Y-axis: sqrt[-log10(p)] to reduce compression")
    print("    - Hybrid coloring provides biological insight for main diseases")
    
    # Detect entity name
    entity_name = detect_entity_name(args.lookup)
    print(f"  • Detected entity name: {entity_name}")
    
    plot_output = output_dir / f"{lookup_basename}_dual_manhattan_plot_normalized.{args.plot_format}"
    plot_dual_manhattan(df_importance, df_gwas_agg, chr_order, chr_midpoints, 
                       chr_boundaries, plot_output, args, lookup_basename, entity_name,
                       df_significant=df_significant)
    
    # ==============================================================
    # CREATE STANDALONE BOTTOM PANEL PLOT
    # ==============================================================
    print("\n[6b] Creating standalone bottom panel plot...")
    
    # Define color palettes (same as in dual plot)
    colors_main = ['#0C4C8A', '#7F7F7F']  # Dark Blue and Gray alternating
    disease_colors = {
        'Type 2 Diabetes': '#FFD700',      # Yellow (Gold)
        'Breast Cancer': '#FF8C00',        # Orange (Dark Orange)
        'Pancreatic Cancer': '#00FFFF',    # Cyan
        'Prostate Cancer': '#32CD32',      # Green (Lime Green)
        'Colorectal Cancer': '#FF00FF'     # Magenta
    }
    main_diseases = list(disease_colors.keys())
    
    bottom_plot_output = output_dir / f"{lookup_basename}_bottom_panel_only.{args.plot_format}"
    plot_bottom_panel_standalone(df_gwas_agg, chr_order, chr_midpoints, chr_boundaries,
                                 bottom_plot_output, args, disease_colors, main_diseases, 
                                 colors_main, df_significant=df_significant, 
                                 entity_name=entity_name)
    
    # ==============================================================
    # GENERATE SUMMARY STATISTICS
    # ==============================================================
    print("\n[7] Generating summary statistics...")
    
    stats = create_summary_statistics(df_importance, df_lookup, df_gwas_agg, df_significant)
    
    # Save statistics
    stats_output = output_dir / "analysis_summary_normalized.txt"
    with open(stats_output, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DUAL MANHATTAN PLOT ANALYSIS SUMMARY - HYBRID COLOR VERSION\n")
        f.write("="*80 + "\n")
        f.write("FEATURES: sqrt(-log10(p)) transformation + Hybrid coloring\n")
        f.write("  - Bottom panel: Disease colors for 5 main diseases, chromosome colors for others\n")
        f.write("  - Disease-colored diamonds for main diseases, chromosome-colored for others\n")
        f.write("="*80 + "\n\n")
        
        f.write("INPUT FILES:\n")
        f.write(f"  Importance scores: {args.importance}\n")
        f.write(f"  GWAS lookup: {args.lookup}\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write(f"  Manhattan plot: {lookup_basename}_dual_manhattan_plot_normalized.{args.plot_format}\n")
        f.write(f"  Significant associations: {lookup_basename}_significant_associations_p5e-8.csv\n")
        f.write(f"  Significant SNPs summary: {lookup_basename}_significant_snps_summary.csv\n")
        f.write(f"  Analysis summary: analysis_summary_normalized.txt\n\n")
        
        f.write("STATISTICS:\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.2f}\n")
            else:
                f.write(f"  {key}: {value:,}\n")
        
        f.write("\n" + "-"*80 + "\n")
        
        # Add distribution by normalized trait
        if len(df_gwas_agg) > 0:
            f.write("\nDISTRIBUTION BY NORMALIZED TRAIT CATEGORY:\n")
            trait_dist = df_gwas_agg['normalized_trait'].value_counts()
            for trait, count in trait_dist.items():
                f.write(f"  {trait}: {count:,} associations\n")
        
        if len(df_significant) > 0:
            f.write("\nSIGNIFICANT ASSOCIATIONS BY NORMALIZED TRAIT:\n")
            sig_trait_dist = df_significant['normalized_trait'].value_counts()
            for trait, count in sig_trait_dist.items():
                f.write(f"  {trait}: {count:,} significant associations\n")
        
        f.write("\n" + "="*80 + "\n")
        
        if len(df_significant) > 0:
            f.write("\nTOP 20 GENOME-WIDE SIGNIFICANT ASSOCIATIONS:\n")
            f.write("-"*80 + "\n")
            top_sig = df_significant.nsmallest(20, 'pval')
            for idx, row in top_sig.iterrows():
                f.write(f"\n{row['rsid']} (p={row['pval']:.2e}):\n")
                f.write(f"  Normalized Trait: {row['normalized_trait']}\n")
                f.write(f"  Original Trait: {row['trait']}\n")
                f.write(f"  Category: {row['category']}\n")
                if 'Importance_Score' in row:
                    f.write(f"  Importance Score: {row['Importance_Score']:.6f}\n")
    
    print(f"  ✓ Saved summary: {stats_output}")
    
    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY - HYBRID COLOR VERSION")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nKey findings:")
    print(f"  • {stats['Unique SNPs with GWAS associations']:,} unique SNPs have published associations")
    print(f"  • {stats['Unique SNP-normalized_trait combinations (after deduplication)']:,} unique SNP-trait combinations")
    print(f"  • {stats['Unique SNPs with genome-wide significance']:,} SNPs reach genome-wide significance (p < 5e-8)")
    print(f"  • {stats['Genome-wide significant associations (p<5e-8)']:,} significant SNP-trait associations")
    
    if len(df_gwas_agg) > 0:
        print(f"\n  Trait distribution (normalized):")
        trait_dist = df_gwas_agg['normalized_trait'].value_counts().head(7)
        for trait, count in trait_dist.items():
            print(f"    - {trait}: {count:,}")
    
    print(f"\n  ✓ Manhattan plots saved to:")
    print(f"    Dual plot: {output_dir / f'{lookup_basename}_dual_manhattan_plot_normalized.{args.plot_format}'}")
    print(f"    Bottom panel only: {output_dir / f'{lookup_basename}_bottom_panel_only.{args.plot_format}'}")
    
    if len(df_significant) > 0:
        print(f"\n  ✓ Significant associations (p < 5e-8) saved to:")
        print(f"    {output_dir / f'{lookup_basename}_significant_associations_p5e-8.csv'}")
        print(f"    {output_dir / f'{lookup_basename}_significant_snps_summary.csv'}")
        print(f"    Format: One row per rsID-normalized_trait (e.g., rs123 appears once per disease category)")
    
    print("\n" + "="*80)
    print("VISUALIZATION FEATURES:")
    print("  • Bottom panel: √[-log₁₀(p-value)] for better spread")
    print("  • Hybrid color scheme:")
    print("    - 5 main diseases: Disease-specific colors (Yellow=T2DM, Orange=Breast, Cyan=Pancreatic, Green=Prostate, Magenta=Colorectal)")
    print("    - Other categories: Alternating blue/gray by chromosome")
    print("  • Disease-colored diamonds: Significant SNPs (p<5e-8) for main diseases")
    print("  • Chromosome-colored diamonds: Significant SNPs for 'Other' categories")
    print("  • Pink labels (top): GWAS-validated SNPs with p-values")
    print("  • Yellow labels (top): Top importance scores")
    print("  • Disease/chromosome-colored labels (bottom): Significant GWAS hits")
    print("  • Standalone bottom panel plot: Full height, available for separate viewing/publication")
    print("="*80)


if __name__ == "__main__":
    main()