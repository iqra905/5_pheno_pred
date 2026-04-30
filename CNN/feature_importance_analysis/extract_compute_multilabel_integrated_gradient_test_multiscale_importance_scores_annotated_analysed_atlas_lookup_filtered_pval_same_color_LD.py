#!/usr/bin/env python3
"""
Dual Manhattan Plot Generator with GWAS-Significant SNP Clumping
------------------------------------------------------------------
Enhanced version that applies LD clumping ONLY to genome-wide significant 
GWAS associations (p < 5e-8) in the bottom panel.

KEY DIFFERENCE FROM PREVIOUS VERSION:
- Top panel: Shows ALL importance scores (no clumping)
- Bottom panel: Clumps only GWAS-significant SNPs (p < 5e-8)
- All non-significant GWAS SNPs remain unchanged

This allows you to:
1. See your complete ML model predictions (top)
2. Identify independent GWAS-validated signals (bottom, clumped)
3. Focus on truly independent genome-wide significant associations

Options:
    -enable_gwas_clumping 1      # Clump GWAS-significant SNPs only (default: 0)
    -clump_window_kb 500         # Clumping window for GWAS sig SNPs (default: 500)
    -clump_min_distance_kb 250   # Min distance between GWAS sig peaks (default: 250)
    -show_ld_colors 1            # Color GWAS SNPs by distance to sig peaks (default: 0)
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

GWAS_SIGNIFICANCE = 5e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dual Manhattan plots with GWAS-significant SNP clumping"
    )
    
    parser.add_argument("-importance", type=str, 
                       default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_disease_wise/t2dm_all_snps_importance_loss_based_disease_wise_test_set_annotated.csv',
                       help="CSV file with importance scores")
    parser.add_argument("-lookup", type=str,
                       default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_disease_wise/analysis_20251218_023025/filtered_snps/t2dm_all_snps_importance_loss_based_disease_wise_test_set_std_filtered_altas_lookup_cancer_diabetes.csv',
                       help="CSV file with GWAS Atlas associations")
    parser.add_argument("-output", type=str,
                       default='/vol/research/fmodal_mmmed/Codes/5_disease_experiments/CNN/results/5d_multilabel/multiscale/hardcoded/parallel/transformer/feat_imp_threshold/feat_imp/feat_imp_disease_wise/analysis_20251218_023025/filtered_snps/',
                       help="Output directory")
    
    # GWAS-specific clumping parameters
    parser.add_argument("-enable_gwas_clumping", type=int, default=1, choices=[0, 1],
                       help="Enable clumping for GWAS-significant SNPs ONLY (default: 0=disabled)")
    parser.add_argument("-clump_window_kb", type=float, default=100,
                       help="Clumping window for GWAS-significant SNPs in kb (default: 500)")
    parser.add_argument("-clump_min_distance_kb", type=float, default=250,
                       help="Minimum distance between GWAS-significant peaks in kb (default: 250)")
    
    # LD Visualization for GWAS panel
    parser.add_argument("-show_ld_colors", type=int, default=1, choices=[0, 1],
                       help="Color ALL GWAS SNPs by distance to significant peaks (default: 0=disabled)")
    parser.add_argument("-ld_color_window_kb", type=float, default=500,
                       help="Window for LD coloring in GWAS panel (default: 500)")
    
    # Regional plots for GWAS-significant loci
    parser.add_argument("-create_regional_plots", type=int, default=0, choices=[0, 1],
                       help="Create regional plots for top GWAS-significant loci (default: 0)")
    parser.add_argument("-n_regional_plots", type=int, default=5,
                       help="Number of top GWAS-significant loci for regional plots (default: 5)")
    parser.add_argument("-regional_window_kb", type=float, default=1000,
                       help="Window size for regional plots in kb (default: 1000)")
    
    # Original parameters
    parser.add_argument("-min_importance", type=float, default=None,
                       help="Minimum importance score to display")
    parser.add_argument("-point_size", type=float, default=5,
                       help="Size of points in Manhattan plot (default: 5)")
    parser.add_argument("-highlight_top_n", type=int, default=0,
                       help="Number of top importance SNPs to label (default: 0)")
    parser.add_argument("-highlight_gwas_sig", type=int, default=1, choices=[0, 1],
                       help="Highlight GWAS-significant SNPs in top panel (default: 1)")
    parser.add_argument("-label_gwas_sig_top_n", type=int, default=1,
                       help="Number of GWAS-significant SNPs to label in top panel (default: 1)")
    parser.add_argument("-label_significant", type=int, default=1, choices=[0, 1],
                       help="Label significant SNPs in bottom panel (default: 1)")
    parser.add_argument("-label_top_significant_n", type=int, default=1,
                       help="Number of significant SNPs to label in bottom panel (default: 1)")
    parser.add_argument("-plot_format", type=str, default="pdf", choices=["png", "pdf", "svg"],
                       help="Plot format (default: pdf)")
    parser.add_argument("-dpi", type=int, default=300,
                       help="DPI for PNG output (default: 300)")
    parser.add_argument("-figsize", type=str, default="16,10",
                       help="Figure size as 'width,height' in inches (default: 16,10)")
    
    return parser.parse_args()


def natural_sort_key(text):
    """Natural sorting for chromosomes"""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', str(text))]


def detect_entity_name(lookup_filename):
    """Detect entity/disease name from lookup filename"""
    basename = Path(lookup_filename).stem
    first_part = basename.split('_')[0].lower()
    
    disease_map = {
        'breast': 'Breast Cancer', 'breacancer': 'Breast Cancer',
        't2dm': 'Type 2 Diabetes', 't2d': 'Type 2 Diabetes', 'diabetes': 'Type 2 Diabetes',
        'panca': 'Pancreatic Cancer', 'pancreatic': 'Pancreatic Cancer',
        'pros': 'Prostate Cancer', 'pros01': 'Prostate Cancer', 'prostate': 'Prostate Cancer',
        'crc': 'Colorectal Cancer', 'colorectal': 'Colorectal Cancer',
        'overall': 'Overall', 'multi': 'Multi-Label', 'multilabel': 'Multi-Label'
    }
    
    return disease_map.get(first_part, first_part.capitalize() if first_part else 'SNP')


def prepare_chromosome_data(df):
    """Prepare data for Manhattan plot with cumulative positions"""
    df['chromosome'] = df['chromosome'].astype(str).str.replace('chr', '', case=False)
    
    chr_order = sorted(df['chromosome'].unique(), key=natural_sort_key)
    df['chr_order'] = pd.Categorical(df['chromosome'], categories=chr_order, ordered=True)
    df = df.sort_values(['chr_order', 'bp']).reset_index(drop=True)
    
    df['cumulative_pos'] = 0.0
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
        
        df.loc[df['chromosome'] == chrom, 'cumulative_pos'] = (
            df.loc[df['chromosome'] == chrom, 'bp'] - chr_data['bp'].min() + cumulative_offset
        )
        
        chr_midpoints[chrom] = (chr_start + chr_end) / 2
        chr_boundaries[chrom] = (chr_start, chr_end)
        
        cumulative_offset = chr_end + chr_length * 0.02
    
    return df, chr_order, chr_midpoints, chr_boundaries


def clump_gwas_significant_snps(df_significant, window_kb=500, min_distance_kb=250):
    """
    Clump ONLY genome-wide significant GWAS SNPs (p < 5e-8).
    
    This function operates ONLY on significant associations, leaving all
    non-significant SNPs untouched.
    
    Parameters:
    -----------
    df_significant : DataFrame
        ONLY the GWAS-significant SNPs (p < 5e-8)
    window_kb : float
        Clumping window in kb (default: 500)
    min_distance_kb : float
        Minimum distance between independent significant peaks (default: 250)
    
    Returns:
    --------
    DataFrame with clumped significant SNPs (lead SNPs only)
    """
    print(f"\n{'='*80}")
    print(f"CLUMPING GWAS-SIGNIFICANT SNPs (p < {GWAS_SIGNIFICANCE})")
    print(f"{'='*80}")
    print(f"  Parameters:")
    print(f"    - Clumping window: {window_kb} kb")
    print(f"    - Minimum distance between peaks: {min_distance_kb} kb")
    print(f"  Input: {len(df_significant):,} GWAS-significant associations")
    print(f"         {df_significant['rsid'].nunique():,} unique significant SNPs")
    
    clumped = []
    stats_per_chr = []
    
    for chrom in sorted(df_significant['chromosome'].unique(), key=natural_sort_key):
        df_chr = df_significant[df_significant['chromosome'] == chrom].copy()
        original_count = len(df_chr)
        
        # Sort by p-value (most significant first)
        df_chr = df_chr.sort_values('pval').reset_index(drop=True)
        
        kept_indices = []
        kept_positions = []
        
        for idx, snp in df_chr.iterrows():
            is_independent = True
            
            for kept_pos in kept_positions:
                distance_kb = abs(snp['bp'] - kept_pos) / 1000
                
                if distance_kb < min_distance_kb:
                    is_independent = False
                    break
            
            if is_independent:
                kept_indices.append(idx)
                kept_positions.append(snp['bp'])
        
        clumped_chr = df_chr.loc[kept_indices].copy()
        clumped_chr['gwas_clump_status'] = 'lead_significant'
        clumped.append(clumped_chr)
        
        clumped_count = len(clumped_chr)
        removed_count = original_count - clumped_count
        stats_per_chr.append({
            'chr': chrom,
            'original': original_count,
            'clumped': clumped_count,
            'removed': removed_count,
            'pct_kept': (clumped_count / original_count * 100) if original_count > 0 else 0
        })
    
    df_clumped = pd.concat(clumped, ignore_index=True) if clumped else pd.DataFrame()
    
    print(f"\n  Clumping Results (GWAS-Significant SNPs Only):")
    print(f"  {'Chromosome':<12} {'Original':<10} {'Clumped':<10} {'Removed':<10} {'% Kept':<10}")
    print(f"  {'-'*60}")
    for stat in stats_per_chr:
        print(f"  {stat['chr']:<12} {stat['original']:<10,} {stat['clumped']:<10,} "
              f"{stat['removed']:<10,} {stat['pct_kept']:<10.1f}%")
    
    if stats_per_chr:
        total_original = sum(s['original'] for s in stats_per_chr)
        total_clumped = sum(s['clumped'] for s in stats_per_chr)
        total_removed = total_original - total_clumped
        
        print(f"  {'-'*60}")
        print(f"  {'TOTAL':<12} {total_original:<10,} {total_clumped:<10,} "
              f"{total_removed:<10,} {(total_clumped/total_original*100):<10.1f}%")
    
    print(f"\n  Result:")
    print(f"    - Independent GWAS-significant peaks: {df_clumped['rsid'].nunique() if len(df_clumped) > 0 else 0:,}")
    print(f"    - Total significant associations after clumping: {len(df_clumped):,}")
    print(f"{'='*80}\n")
    
    return df_clumped


def assign_ld_colors_to_gwas_snps(df_gwas_all, df_sig_clumped, window_kb=500):
    """
    Assign LD colors to ALL GWAS SNPs based on distance to significant lead SNPs.
    
    This colors the entire GWAS panel to show which SNPs are in LD with
    the independent significant signals.
    
    Parameters:
    -----------
    df_gwas_all : DataFrame
        ALL GWAS SNPs (significant and non-significant)
    df_sig_clumped : DataFrame
        Only the clumped significant lead SNPs
    window_kb : float
        LD coloring window in kb
    
    Returns:
    --------
    DataFrame with 'ld_color_value' column added
    """
    print(f"\n{'='*80}")
    print(f"LD COLOR ASSIGNMENT FOR GWAS PANEL")
    print(f"{'='*80}")
    print(f"  Coloring window: {window_kb} kb")
    print(f"  Significant lead SNPs: {len(df_sig_clumped):,}")
    print(f"  All GWAS SNPs to color: {len(df_gwas_all):,}")
    
    df_gwas_all = df_gwas_all.copy()
    df_gwas_all['ld_color_value'] = 0.0
    df_gwas_all['nearest_sig_lead'] = ''
    df_gwas_all['distance_to_sig_lead_kb'] = np.inf
    
    # Create lookup of significant lead SNPs by chromosome
    sig_lead_dict = {}
    for chrom in df_sig_clumped['chromosome'].unique():
        sig_chr = df_sig_clumped[df_sig_clumped['chromosome'] == chrom]
        sig_lead_dict[chrom] = sig_chr[['bp', 'rsid']].values if 'rsid' in sig_chr.columns else sig_chr[['bp']].values
    
    colored_count = 0
    
    for chrom in df_gwas_all['chromosome'].unique():
        if chrom not in sig_lead_dict:
            continue
        
        chr_mask = df_gwas_all['chromosome'] == chrom
        df_chr = df_gwas_all[chr_mask]
        lead_positions = sig_lead_dict[chrom]
        
        for idx in df_chr.index:
            snp_pos = df_gwas_all.loc[idx, 'bp']
            
            # Find nearest significant lead SNP
            distances = np.abs(lead_positions[:, 0] - snp_pos)
            min_dist_idx = np.argmin(distances)
            min_distance = distances[min_dist_idx] / 1000  # kb
            
            # Calculate LD proxy
            if min_distance <= window_kb:
                ld_proxy = 1.0 - (min_distance / window_kb)
                df_gwas_all.loc[idx, 'ld_color_value'] = ld_proxy
                df_gwas_all.loc[idx, 'distance_to_sig_lead_kb'] = min_distance
                
                if len(lead_positions[0]) > 1:
                    df_gwas_all.loc[idx, 'nearest_sig_lead'] = lead_positions[min_dist_idx, 1]
                
                colored_count += 1
    
    print(f"  SNPs within LD window of significant peaks: {colored_count:,} ({colored_count/len(df_gwas_all)*100:.1f}%)")
    print(f"  SNPs outside LD window: {len(df_gwas_all)-colored_count:,}")
    print(f"{'='*80}\n")
    
    return df_gwas_all


def plot_dual_manhattan_gwas_clumped(df_importance, df_gwas_agg, chr_order, chr_midpoints,
                                      chr_boundaries, output_path, args, lookup_basename='',
                                      entity_name='SNP', df_significant=None, 
                                      df_sig_clumped=None):
    """
    Create dual Manhattan plot with GWAS-significant SNP clumping in bottom panel.
    
    Top panel: ALL importance scores (unchanged)
    Bottom panel: Clumped GWAS-significant SNPs + all non-significant SNPs
    """
    figsize = tuple(map(float, args.figsize.split(',')))
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.05,
                         left=0.08, right=0.98, top=0.95, bottom=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Determine if using LD colors for GWAS panel
    use_ld_colors = args.show_ld_colors and df_sig_clumped is not None and len(df_sig_clumped) > 0
    
    if use_ld_colors:
        print("  • Applying LD-based coloring to GWAS panel...")
        df_gwas_agg = assign_ld_colors_to_gwas_snps(
            df_gwas_agg,
            df_sig_clumped,
            window_kb=args.ld_color_window_kb
        )
    
    # Calculate threshold for importance
    mean_importance = df_importance['Importance_Score'].mean()
    std_importance = df_importance['Importance_Score'].std()
    importance_threshold = mean_importance + 5.0 * std_importance
    
    # ==============================================================
    # TOP PANEL: ALL Importance Scores (NO CLUMPING)
    # ==============================================================
    print("  • Plotting ALL importance scores (top panel - no clumping)...")
    
    colors_main = ['#0C4C8A', '#7F7F7F']
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
    
    # Threshold line
    n_above_threshold = (df_importance['Importance_Score'] > importance_threshold).sum()
    ax1.axhline(y=importance_threshold, color='red',
               linestyle='--', linewidth=1.5, alpha=0.8,
               label=f'Threshold (mean + 5×std): {importance_threshold:.2e}\n    SNPs above: {n_above_threshold:,}',
               zorder=10)
    
    # Highlight GWAS-significant SNPs
    if args.highlight_gwas_sig and df_significant is not None and len(df_significant) > 0:
        sig_rsids = set(df_significant['rsid'].unique())
        df_importance['is_gwas_sig'] = df_importance['rsid'].isin(sig_rsids)
        df_gwas_sig_in_importance = df_importance[df_importance['is_gwas_sig']].copy()
        
        if len(df_gwas_sig_in_importance) > 0:
            ax1.scatter(df_gwas_sig_in_importance['cumulative_pos'],
                       df_gwas_sig_in_importance['Importance_Score'],
                       c='red',
                       s=args.point_size * 3,
                       marker='D',
                       edgecolors='darkred',
                       linewidths=0.8,
                       alpha=0.9,
                       zorder=80,
                       label=f'GWAS-significant (p<5×10⁻⁸): {len(df_gwas_sig_in_importance):,} SNPs')
            
            # Labels
            if args.label_gwas_sig_top_n != -1:
                df_gwas_sig_labeled = df_gwas_sig_in_importance.merge(
                    df_significant[['rsid', 'pval']].drop_duplicates('rsid'),
                    on='rsid', how='left'
                )
                
                if 'pval' in df_gwas_sig_labeled.columns:
                    df_gwas_sig_labeled = df_gwas_sig_labeled.sort_values('pval')
                
                df_to_label = df_gwas_sig_labeled.head(args.label_gwas_sig_top_n) if args.label_gwas_sig_top_n > 0 else df_gwas_sig_labeled
                
                id_col = 'rsid' if 'rsid' in df_to_label.columns else 'snp_id'
                
                for idx, row in df_to_label.iterrows():
                    label_text = f"{row[id_col]}, p={row['pval']:.2e}" if 'pval' in row and pd.notna(row['pval']) else f"{row[id_col]}\n(GWAS sig)"
                    
                    ax1.annotate(label_text,
                                (row['cumulative_pos'], row['Importance_Score']),
                                fontsize=7, alpha=0.8, fontweight='bold', color='darkred',
                                xytext=(5, -15), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='pink',
                                         alpha=0.4, edgecolor='darkred', linewidth=0.5),
                                zorder=90)
    
    # Top importance SNPs labels
    if args.highlight_top_n > 0:
        id_col = 'snp_id' if 'snp_id' in df_importance.columns else 'rsid'
        if id_col in df_importance.columns:
            top_snps = df_importance.nlargest(args.highlight_top_n, 'Importance_Score')
            
            for idx, row in top_snps.iterrows():
                ax1.annotate(row[id_col],
                            (row['cumulative_pos'], row['Importance_Score']),
                            fontsize=7, alpha=0.8, fontweight='bold', color='black',
                            xytext=(5, 5), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                                     alpha=0.5, edgecolor='orange', linewidth=0.5),
                            zorder=100)
    
    # Top panel styling
    ax1.set_ylabel('Importance Score', fontsize=13, fontweight='bold')
    ax1.set_title(f'{entity_name} - Manhattan Plot of SNP Importance (All SNPs)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax1.legend(loc='upper right', frameon=True, fontsize=9,
              framealpha=0.9, edgecolor='black', fancybox=False)
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    for chrom in chr_order[:-1]:
        if chrom in chr_boundaries:
            boundary = chr_boundaries[chrom][1]
            ax1.axvline(x=boundary, color='gray', linestyle='-', linewidth=0.3, alpha=0.3, zorder=0)
    
    # ==============================================================
    # BOTTOM PANEL: GWAS with Clumped Significant SNPs
    # ==============================================================
    print("  • Plotting GWAS panel with clumped significant SNPs...")
    
    df_gwas_agg['-log10_pval'] = -np.log10(df_gwas_agg['pval'] + 1e-300)
    df_gwas_agg['sqrt_log10_pval'] = np.sqrt(df_gwas_agg['-log10_pval'])
    
    # Separate into: non-significant, significant (all), and significant (clumped leads)
    df_nonsig = df_gwas_agg[df_gwas_agg['pval'] >= GWAS_SIGNIFICANCE].copy()
    df_sig_all = df_gwas_agg[df_gwas_agg['pval'] < GWAS_SIGNIFICANCE].copy()
    
    # Further split significant into clumped leads and non-leads
    if df_sig_clumped is not None and len(df_sig_clumped) > 0:
        sig_lead_rsids = set(df_sig_clumped['rsid'].unique())
        df_sig_leads = df_sig_all[df_sig_all['rsid'].isin(sig_lead_rsids)].copy()
        df_sig_nonleads = df_sig_all[~df_sig_all['rsid'].isin(sig_lead_rsids)].copy()
    else:
        df_sig_leads = df_sig_all.copy()
        df_sig_nonleads = pd.DataFrame()
    
    print(f"    - Non-significant: {len(df_nonsig):,}")
    print(f"    - Significant (all): {len(df_sig_all):,}")
    if df_sig_clumped is not None:
        print(f"    - Significant (lead SNPs): {len(df_sig_leads):,}")
        print(f"    - Significant (in LD with leads): {len(df_sig_nonleads):,}")
    
    # Plot non-significant SNPs
    if use_ld_colors:
        # Color by LD to significant peaks
        if len(df_nonsig) > 0:
            scatter = ax2.scatter(
                df_nonsig['cumulative_pos'],
                -df_nonsig['sqrt_log10_pval'],
                c=df_nonsig['ld_color_value'],
                cmap='YlOrRd',
                s=args.point_size * 2,
                alpha=0.7,
                edgecolors='none',
                rasterized=True,
                vmin=0, vmax=1
            )
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax2, pad=0.01, aspect=30)
            cbar.set_label('LD Proxy to\nSig. Lead SNP', fontsize=9, rotation=0,
                          ha='left', va='center')
            cbar.ax.yaxis.set_label_coords(3.5, 0.5)
    else:
        # Standard alternating colors
        for i, chrom in enumerate(chr_order):
            df_chr_nonsig = df_nonsig[df_nonsig['chromosome'] == chrom]
            if len(df_chr_nonsig) == 0:
                continue
            
            color = colors_main[i % 2]
            ax2.scatter(df_chr_nonsig['cumulative_pos'],
                       -df_chr_nonsig['sqrt_log10_pval'],
                       c=color,
                       s=args.point_size * 2,
                       alpha=0.7,
                       edgecolors='none',
                       rasterized=True)
    
    # Plot significant non-lead SNPs (in LD with leads) - if clumping is enabled
    if len(df_sig_nonleads) > 0:
        if use_ld_colors:
            # Use LD color gradient
            ax2.scatter(df_sig_nonleads['cumulative_pos'],
                       -df_sig_nonleads['sqrt_log10_pval'],
                       c=df_sig_nonleads['ld_color_value'],
                       cmap='YlOrRd',
                       s=args.point_size * 2,
                       alpha=0.8,
                       edgecolors='black',
                       linewidths=0.3,
                       rasterized=True,
                       vmin=0, vmax=1,
                       zorder=90)
        else:
            # Orange for significant but not lead
            ax2.scatter(df_sig_nonleads['cumulative_pos'],
                       -df_sig_nonleads['sqrt_log10_pval'],
                       c='orange',
                       s=args.point_size * 2,
                       alpha=0.8,
                       edgecolors='darkorange',
                       linewidths=0.5,
                       zorder=90,
                       label=f'Significant (in LD): {len(df_sig_nonleads):,}')
    
    # Plot significant LEAD SNPs as RED DIAMONDS
    if len(df_sig_leads) > 0:
        n_unique_snps_total = df_gwas_agg['rsid'].nunique()
        n_unique_snps_sig_leads = df_sig_leads['rsid'].nunique()
        
        clump_note = f' (clumped from {len(df_sig_all):,})' if df_sig_clumped is not None else ''
        
        ax2.scatter(df_sig_leads['cumulative_pos'],
                   -df_sig_leads['sqrt_log10_pval'],
                   c='red',
                   s=args.point_size * 3,
                   marker='D',
                   edgecolors='darkred',
                   linewidths=1.2,
                   alpha=0.95,
                   zorder=100,
                   label=f'Independent Significant{clump_note}: {len(df_sig_leads):,} assoc. ({n_unique_snps_sig_leads:,} SNPs)\n    Total: {len(df_gwas_agg):,} assoc. ({n_unique_snps_total:,} SNPs)')
        
        # Labels for lead SNPs
        if args.label_significant and args.label_top_significant_n != -1:
            df_to_label = df_sig_leads if args.label_top_significant_n == 0 else df_sig_leads.nsmallest(args.label_top_significant_n, 'pval')
            
            for idx, row in df_to_label.iterrows():
                label_text = f"{row['rsid']}, p={row['pval']:.2e}"
                ax2.annotate(label_text,
                            (row['cumulative_pos'], -row['sqrt_log10_pval']),
                            fontsize=7, alpha=0.8, fontweight='bold',
                            xytext=(5, 5), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                                     alpha=0.3, edgecolor='none'),
                            zorder=110)
    
    # Threshold line
    threshold_y = -np.sqrt(-np.log10(GWAS_SIGNIFICANCE))
    ax2.axhline(y=threshold_y, color='red', linestyle='--', linewidth=1.5,
               alpha=0.8, label=f'Genome-wide threshold (p=5×10⁻⁸)', zorder=10)
    
    # Bottom panel styling
    clump_note = ' (Significant SNPs Clumped)' if df_sig_clumped is not None else ''
    ax2.set_ylabel('√[-log₁₀(p-value)] [Mirrored]', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Chromosome', fontsize=13, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='upper right', frameon=True, fontsize=9,
              framealpha=0.9, edgecolor='black', fancybox=False)
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    ax2.invert_yaxis()
    
    for chrom in chr_order[:-1]:
        if chrom in chr_boundaries:
            boundary = chr_boundaries[chrom][1]
            ax2.axvline(x=boundary, color='gray', linestyle='-', linewidth=0.3, alpha=0.3, zorder=0)
    
    # X-axis
    ax2.set_xticks([chr_midpoints[chrom] for chrom in chr_order if chrom in chr_midpoints])
    ax2.set_xticklabels(chr_order, fontsize=10)
    ax2.set_xlim(0, df_importance['cumulative_pos'].max())
    
    # Alternating background
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
    
    suffix = '_gwas_sig_clumped'
    print(f"  ✓ Saved dual Manhattan plot{suffix}: {output_path}")


def create_regional_plots_gwas(df_gwas_agg, df_sig_clumped, chr_boundaries, output_dir, args, entity_name='SNP'):
    """Create regional plots for top GWAS-significant loci"""
    print(f"\n{'='*80}")
    print(f"CREATING REGIONAL PLOTS FOR GWAS-SIGNIFICANT LOCI")
    print(f"{'='*80}")
    print(f"  Number of regions: {min(args.n_regional_plots, len(df_sig_clumped))}")
    print(f"  Window size: ±{args.regional_window_kb} kb")
    
    # Get top N significant lead SNPs
    top_leads = df_sig_clumped.nsmallest(min(args.n_regional_plots, len(df_sig_clumped)), 'pval')
    
    for idx, lead_snp in top_leads.iterrows():
        chrom = lead_snp['chromosome']
        center_pos = lead_snp['bp']
        rsid = lead_snp['rsid'] if 'rsid' in lead_snp else f"chr{chrom}:{center_pos}"
        
        start_pos = max(0, center_pos - args.regional_window_kb * 1000)
        end_pos = center_pos + args.regional_window_kb * 1000
        
        # Get all GWAS SNPs in region
        df_region = df_gwas_agg[
            (df_gwas_agg['chromosome'] == chrom) &
            (df_gwas_agg['bp'] >= start_pos) &
            (df_gwas_agg['bp'] <= end_pos)
        ].copy()
        
        if len(df_region) == 0:
            continue
        
        df_region['distance_kb'] = abs(df_region['bp'] - center_pos) / 1000
        df_region['ld_proxy'] = 1.0 - (df_region['distance_kb'] / args.regional_window_kb)
        df_region['ld_proxy'] = df_region['ld_proxy'].clip(0, 1)
        df_region['-log10_pval'] = -np.log10(df_region['pval'] + 1e-300)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        scatter = ax.scatter(
            df_region['bp'] / 1e6,
            df_region['-log10_pval'],
            c=df_region['ld_proxy'],
            cmap='YlOrRd',
            s=30,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Highlight lead SNP
        ax.scatter(
            center_pos / 1e6,
            -np.log10(lead_snp['pval']),
            c='purple',
            s=200,
            marker='D',
            edgecolors='black',
            linewidths=2,
            zorder=100,
            label=f'Lead SNP: {rsid} (p={lead_snp["pval"]:.2e})'
        )
        
        # Significance threshold
        ax.axhline(y=-np.log10(GWAS_SIGNIFICANCE), color='red', linestyle='--',
                  linewidth=1.5, alpha=0.7, label='Genome-wide significance')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('LD Proxy (Distance-based)', fontsize=10)
        
        # Styling
        ax.set_xlabel(f'Position on Chromosome {chrom} (Mb)', fontsize=12, fontweight='bold')
        ax.set_ylabel('-log₁₀(p-value)', fontsize=12, fontweight='bold')
        ax.set_title(f'{entity_name} - Regional Plot: {rsid}\nChr {chrom}:{start_pos/1e6:.2f}-{end_pos/1e6:.2f} Mb',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Save
        output_file = output_dir / f'gwas_regional_plot_{rsid.replace(":", "_")}_chr{chrom}.{args.plot_format}'
        save_kwargs = {'dpi': args.dpi, 'bbox_inches': 'tight'} if args.plot_format == 'png' else {'bbox_inches': 'tight'}
        plt.savefig(output_file, format=args.plot_format, **save_kwargs)
        plt.close()
        
        print(f"  ✓ Saved: {output_file.name}")
    
    print(f"{'='*80}\n")


def normalize_trait_name(trait_name):
    """Normalize trait names to canonical categories"""
    trait_lower = str(trait_name).lower()
    
    if any(keyword in trait_lower for keyword in [
        'diabetes', 'diabetic', 'dm2', 't2d', 't2dm', 'type 2', 'type ii',
        'e11', 'e10', 'glucose', 'glyc', 'hba1c', 'insulin'
    ]):
        return 'Type 2 Diabetes'
    elif any(keyword in trait_lower for keyword in [
        'colorectal', 'colon', 'rectal', 'rectum', 'crc', 'bowel',
        'c18', 'c19', 'c20', 'd12'
    ]):
        return 'Colorectal Cancer'
    elif any(keyword in trait_lower for keyword in [
        'breast', 'mammary', 'c50', 'd05'
    ]):
        return 'Breast Cancer'
    elif any(keyword in trait_lower for keyword in [
        'prostate', 'prostatic', 'c61', 'd07.5'
    ]):
        return 'Prostate Cancer'
    elif any(keyword in trait_lower for keyword in [
        'pancrea', 'c25', 'd13.6'
    ]):
        return 'Pancreatic Cancer'
    elif any(keyword in trait_lower for keyword in [
        'cancer', 'carcinoma', 'neoplasm', 'tumor', 'tumour', 'malign'
    ]):
        return 'Other Cancer'
    else:
        return 'Other'


def aggregate_multiple_associations(df_lookup):
    """Handle SNPs with multiple GWAS associations"""
    print("  • Normalizing trait names...")
    
    df_lookup['normalized_trait'] = df_lookup['trait'].apply(normalize_trait_name)
    
    print(f"    Original unique traits: {df_lookup['trait'].nunique()}")
    print(f"    Normalized to: {df_lookup['normalized_trait'].nunique()} categories")
    
    df_dedup = df_lookup.sort_values('pval').groupby(['rsid', 'normalized_trait'], as_index=False).first()
    
    snp_counts = df_lookup.groupby('rsid').size().reset_index(name='n_total_associations')
    df_dedup = df_dedup.merge(snp_counts, on='rsid', how='left')
    
    trait_counts = df_lookup.groupby('rsid')['normalized_trait'].nunique().reset_index(name='n_unique_normalized_traits')
    df_dedup = df_dedup.merge(trait_counts, on='rsid', how='left')
    
    return df_dedup


def filter_significant_associations(df_gwas, pval_threshold=5e-8):
    """Filter GWAS associations below p-value threshold"""
    df_significant = df_gwas[df_gwas['pval'] < pval_threshold].copy()
    df_significant = df_significant.sort_values('pval')
    return df_significant


def create_summary_statistics(df_importance, df_lookup, df_gwas_agg, df_significant, df_sig_clumped=None):
    """Generate summary statistics"""
    stats = {
        'Total SNPs in importance data': len(df_importance),
        'Total GWAS associations found (raw)': len(df_lookup),
        'Unique SNP-normalized_trait combinations': len(df_gwas_agg),
        'Unique SNPs with GWAS associations': df_gwas_agg['rsid'].nunique() if len(df_gwas_agg) > 0 else 0,
        'Genome-wide significant associations (p<5e-8)': len(df_significant),
        'Unique SNPs with genome-wide significance': df_significant['rsid'].nunique() if len(df_significant) > 0 else 0,
    }
    
    if df_sig_clumped is not None:
        stats['Independent GWAS-significant SNPs (after clumping)'] = df_sig_clumped['rsid'].nunique()
        stats['GWAS-significant associations (after clumping)'] = len(df_sig_clumped)
        if len(df_significant) > 0:
            stats['Reduction in significant SNPs (%)'] = (1 - df_sig_clumped['rsid'].nunique() / df_significant['rsid'].nunique()) * 100
    
    return stats


def main():
    args = parse_args()
    
    print("="*80)
    print("DUAL MANHATTAN PLOT - GWAS-SIGNIFICANT SNP CLUMPING")
    print("="*80)
    print("KEY FEATURES:")
    print("  - Top panel: ALL importance scores (no clumping)")
    print("  - Bottom panel: Clumped GWAS-significant SNPs (p<5e-8)")
    print("  - LD visualization for GWAS panel (optional)")
    print("="*80)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1] Loading data...")
    df_importance = pd.read_csv(args.importance)
    print(f"  ✓ Loaded importance scores: {len(df_importance):,} SNPs")
    
    required_cols = ['chromosome', 'bp', 'Importance_Score']
    missing_cols = [col for col in required_cols if col not in df_importance.columns]
    if missing_cols:
        raise ValueError(f"Importance file missing: {missing_cols}")
    
    if args.min_importance is not None:
        df_importance = df_importance[df_importance['Importance_Score'] >= args.min_importance]
        print(f"  ✓ Filtered to {len(df_importance):,} SNPs")
    
    df_lookup = pd.read_csv(args.lookup)
    print(f"  ✓ Loaded GWAS associations: {len(df_lookup):,}")
    
    # Deduplicate GWAS
    print(f"\n[2] Deduplicating GWAS associations...")
    if 'chromosome' in df_lookup.columns and 'bp' in df_lookup.columns:
        df_lookup = df_lookup.dropna(subset=['chromosome', 'bp'])
    
    df_gwas_agg = aggregate_multiple_associations(df_lookup)
    print(f"  ✓ After deduplication: {len(df_gwas_agg):,}")
    
    # Filter significant
    print(f"\n[3] Filtering genome-wide significant associations...")
    df_significant = filter_significant_associations(df_gwas_agg, GWAS_SIGNIFICANCE)
    
    lookup_basename = Path(args.lookup).stem
    
    if len(df_significant) > 0:
        print(f"  ✓ Found {len(df_significant):,} significant associations")
        print(f"    Affecting {df_significant['rsid'].nunique():,} unique SNPs")
    else:
        print("  ⚠ No genome-wide significant associations found")
        df_significant = pd.DataFrame()
    
    # GWAS-significant SNP clumping
    df_sig_clumped = None
    if args.enable_gwas_clumping and len(df_significant) > 0:
        print("\n[4] Clumping GWAS-significant SNPs...")
        df_sig_clumped = clump_gwas_significant_snps(
            df_significant,
            window_kb=args.clump_window_kb,
            min_distance_kb=args.clump_min_distance_kb
        )
        
        if len(df_sig_clumped) > 0:
            clumped_output = output_dir / "gwas_significant_clumped_lead_snps.csv"
            df_sig_clumped.to_csv(clumped_output, index=False)
            print(f"  ✓ Saved: {clumped_output}")
    else:
        print("\n[4] Skipping GWAS clumping (disabled or no significant SNPs)")
    
    # Prepare data
    print("\n[5] Preparing Manhattan plot data...")
    
    if 'snp_id' in df_importance.columns and 'rsid' not in df_importance.columns:
        df_importance['rsid'] = df_importance['snp_id']
    elif 'rsid' not in df_importance.columns:
        df_importance['rsid'] = 'unknown'
    
    df_importance, chr_order, chr_midpoints, chr_boundaries = prepare_chromosome_data(df_importance)
    
    # Merge GWAS data
    merge_cols = ['rsid', 'chromosome', 'bp', 'Importance_Score']
    if 'snp_id' in df_importance.columns:
        merge_cols.append('snp_id')
    
    df_gwas_agg = df_gwas_agg.merge(
        df_importance[merge_cols],
        on='rsid',
        how='left',
        suffixes=('_gwas', '_importance')
    )
    
    # Handle merged columns
    if 'chromosome_importance' in df_gwas_agg.columns:
        df_gwas_agg['chromosome'] = df_gwas_agg.get('chromosome_gwas', df_gwas_agg['chromosome_importance']).fillna(df_gwas_agg['chromosome_importance'])
        df_gwas_agg = df_gwas_agg.drop(columns=[c for c in ['chromosome_gwas', 'chromosome_importance'] if c in df_gwas_agg.columns])
    
    if 'bp_importance' in df_gwas_agg.columns:
        df_gwas_agg['bp'] = df_gwas_agg.get('bp_gwas', df_gwas_agg['bp_importance']).fillna(df_gwas_agg['bp_importance'])
        df_gwas_agg = df_gwas_agg.drop(columns=[c for c in ['bp_gwas', 'bp_importance'] if c in df_gwas_agg.columns])
    
    df_gwas_agg = df_gwas_agg.dropna(subset=['chromosome', 'bp'])
    
    # Calculate cumulative positions
    df_gwas_agg['chromosome'] = df_gwas_agg['chromosome'].astype(str).str.strip().str.replace('chr', '', case=False)
    df_gwas_agg['cumulative_pos'] = 0.0
    
    for chrom in chr_order:
        gwas_chr_mask = df_gwas_agg['chromosome'] == chrom
        if gwas_chr_mask.sum() == 0:
            continue
        
        imp_chr_data = df_importance[df_importance['chromosome'] == chrom]
        if len(imp_chr_data) == 0:
            continue
        
        chr_min_bp = imp_chr_data['bp'].min()
        chr_start = chr_boundaries[chrom][0]
        
        df_gwas_agg.loc[gwas_chr_mask, 'cumulative_pos'] = (
            df_gwas_agg.loc[gwas_chr_mask, 'bp'] - chr_min_bp + chr_start
        )
    
    df_gwas_agg['chr_order'] = pd.Categorical(df_gwas_agg['chromosome'], categories=chr_order, ordered=True)
    df_gwas_agg = df_gwas_agg.sort_values(['chr_order', 'cumulative_pos']).reset_index(drop=True)
    
    print(f"  ✓ Final data: {len(df_gwas_agg):,} GWAS associations")
    
    # Create Manhattan plot
    print("\n[6] Creating dual Manhattan plot...")
    
    entity_name = detect_entity_name(args.lookup)
    
    suffix = '_gwas_sig_clumped' if args.enable_gwas_clumping else ''
    plot_output = output_dir / f"{lookup_basename}_dual_manhattan{suffix}.{args.plot_format}"
    
    plot_dual_manhattan_gwas_clumped(
        df_importance, df_gwas_agg, chr_order, chr_midpoints,
        chr_boundaries, plot_output, args, lookup_basename, entity_name,
        df_significant=df_significant, df_sig_clumped=df_sig_clumped
    )
    
    # Regional plots
    if args.create_regional_plots and df_sig_clumped is not None and len(df_sig_clumped) > 0:
        print("\n[7] Creating regional plots...")
        create_regional_plots_gwas(
            df_gwas_agg, df_sig_clumped, chr_boundaries,
            output_dir, args, entity_name
        )
    
    # Summary statistics
    print("\n[8] Generating summary...")
    
    stats = create_summary_statistics(df_importance, df_lookup, df_gwas_agg, df_significant, df_sig_clumped)
    
    stats_output = output_dir / "analysis_summary_gwas_clumping.txt"
    with open(stats_output, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DUAL MANHATTAN PLOT - GWAS-SIGNIFICANT SNP CLUMPING\n")
        f.write("="*80 + "\n\n")
        
        f.write("CLUMPING APPROACH:\n")
        f.write("  - Top panel: ALL importance scores (no clumping)\n")
        f.write("  - Bottom panel: GWAS-significant SNPs clumped (p<5e-8)\n")
        if args.enable_gwas_clumping:
            f.write(f"    Window: {args.clump_window_kb} kb\n")
            f.write(f"    Min distance: {args.clump_min_distance_kb} kb\n\n")
        
        f.write("STATISTICS:\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.2f}\n")
            else:
                f.write(f"  {key}: {value:,}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"  ✓ Saved summary: {stats_output}")
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nKey findings:")
    print(f"  • Total importance SNPs: {len(df_importance):,}")
    print(f"  • GWAS-significant SNPs (all): {stats['Unique SNPs with genome-wide significance']:,}")
    if df_sig_clumped is not None:
        print(f"  • Independent GWAS-significant SNPs: {stats['Independent GWAS-significant SNPs (after clumping)']:,}")
        print(f"  • Reduction: {stats['Reduction in significant SNPs (%)']:.1f}%")
    print("="*80)


if __name__ == "__main__":
    main()