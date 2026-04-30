#!/usr/bin/env python3
"""
Enhanced Post-Analysis Script with Genomic Annotation Support 
Now uses CSV as primary source - NPY files are optional!

This script analyzes pre-computed importance scores with optional genomic annotations.
CSV files contain all necessary data, so NPY files are no longer required.

Usage:
    python analyze_importance_scores_annotated.py -scores_dir ./computed_scores/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from datetime import datetime
from sklearn.mixture import GaussianMixture
import re
import glob

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze SNP Importance Scores with Genomic Annotation Support (V2)")
    
    parser.add_argument("-scores_dir", type=str, required=True, help="Directory containing computed scores")
    parser.add_argument("-output_dir", type=str, default=None, help="Output directory (default: scores_dir/analysis_DATE)")
    
    # Threshold methods
    parser.add_argument("-threshold_methods", type=str, default="elbow", help="Comma-separated threshold methods (choices:elbow,percentile,gap,std)")
    parser.add_argument("-percentile_threshold", type=float, default=99.0)
    parser.add_argument("-std_multiplier", type=float, default=3.0)
    parser.add_argument("-min_gap_ratio", type=float, default=1.5)
    
    # Clustering
    parser.add_argument("-cluster_distance", type=int, default=50000)
    parser.add_argument("-min_cluster_size", type=int, default=50)
    
    # Visualization
    parser.add_argument("-create_plots", type=int, default=1, choices=[0, 1])
    parser.add_argument("-plot_format", type=str, default="pdf", choices=["png", "pdf", "svg"])
    
    return parser.parse_args()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def natural_sort_key(text):
    """Natural sorting for chromosomes"""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', str(text))]


# ============================================================================
# THRESHOLD SELECTION
# ============================================================================

class ThresholdSelector:
    @staticmethod
    def find_elbow_derivative(sorted_scores):
        if len(sorted_scores) < 10:
            return len(sorted_scores) // 2
        
        first_deriv = np.diff(sorted_scores)
        second_deriv = np.diff(first_deriv)
        
        window = min(50, len(second_deriv) // 10)
        if window > 2:
            second_deriv_smooth = np.convolve(second_deriv, 
                                             np.ones(window)/window, 
                                             mode='valid')
            elbow_index = np.argmin(second_deriv_smooth) + window // 2 + 2
        else:
            elbow_index = np.argmin(second_deriv) + 2
        
        return min(elbow_index, len(sorted_scores) - 1)
    
    @staticmethod
    def elbow_method(scores):
        sorted_scores = np.sort(scores)[::-1]
        sorted_scores = sorted_scores[sorted_scores > 1e-10]
        
        if len(sorted_scores) < 10:
            return sorted_scores[-1] if len(sorted_scores) > 0 else 0, len(sorted_scores)
        
        elbow_index = ThresholdSelector.find_elbow_derivative(sorted_scores)
        threshold = sorted_scores[elbow_index]
        
        return threshold, elbow_index + 1
    
    @staticmethod
    def percentile_method(scores, percentile=99.0):
        threshold = np.percentile(scores, percentile)
        n_selected = np.sum(scores > threshold)
        return threshold, n_selected
    
    @staticmethod
    def std_method(scores, multiplier=3.0):
        mean = scores.mean()
        std = scores.std()
        threshold = mean + multiplier * std
        n_selected = np.sum(scores > threshold)
        return threshold, n_selected
    
    @staticmethod
    def gap_method(scores, min_gap_ratio=1.5):
        sorted_scores = np.sort(scores)[::-1]
        sorted_scores = sorted_scores[sorted_scores > 0]
        
        if len(sorted_scores) < 2:
            return sorted_scores[0] if len(sorted_scores) > 0 else 0, len(sorted_scores)
        
        ratios = sorted_scores[:-1] / (sorted_scores[1:] + 1e-10)
        gap_indices = np.where(ratios > min_gap_ratio)[0]
        
        if len(gap_indices) > 0:
            gap_index = gap_indices[0]
            threshold = sorted_scores[gap_index + 1]
            n_selected = gap_index + 1
        else:
            threshold = np.percentile(sorted_scores, 99)
            n_selected = int(len(sorted_scores) * 0.01)
        
        return threshold, n_selected
    
    @staticmethod
    def mixture_method(scores, n_components=2):
        scores_nonzero = scores[scores > 0]
        
        if len(scores_nonzero) < 100:
            return ThresholdSelector.percentile_method(scores, 99)
        
        log_scores = np.log10(scores_nonzero + 1e-10).reshape(-1, 1)
        
        try:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(log_scores)
            labels = gmm.predict(log_scores)
            
            means = gmm.means_.flatten()
            signal_component = np.argmax(means)
            
            signal_scores = scores_nonzero[labels == signal_component]
            threshold = signal_scores.min()
            n_selected = len(signal_scores)
            
            return threshold, n_selected
        except:
            return ThresholdSelector.percentile_method(scores, 99)


# ============================================================================
# GENOMIC CLUSTERING
# ============================================================================

class GenomicClusterAnalyzer:
    def __init__(self, max_distance=50000, min_cluster_size=3):
        self.max_distance = max_distance
        self.min_cluster_size = min_cluster_size
    
    def find_clusters_annotated(self, df_selected):
        """Find clusters using actual genomic positions from annotation"""
        if 'chromosome' not in df_selected.columns or 'bp' not in df_selected.columns:
            return self.find_clusters_by_index(df_selected)
        
        clusters = []
        
        # Group by chromosome
        for chrom in sorted(df_selected['chromosome'].unique(), key=natural_sort_key):
            chrom_data = df_selected[df_selected['chromosome'] == chrom].copy()
            
            if len(chrom_data) == 0:
                continue
            
            # Sort by position
            chrom_data = chrom_data.sort_values('bp')
            
            # Find gaps
            positions = chrom_data['bp'].values
            scores = chrom_data['Importance_Score'].values
            snp_indices = chrom_data['SNP_Index'].values
            
            gaps = np.diff(positions)
            cluster_breaks = np.where(gaps > self.max_distance)[0] + 1
            
            cluster_starts = np.concatenate([[0], cluster_breaks])
            cluster_ends = np.concatenate([cluster_breaks, [len(positions)]])
            
            for i, (start, end) in enumerate(zip(cluster_starts, cluster_ends)):
                cluster_positions = positions[start:end]
                cluster_scores = scores[start:end]
                cluster_snp_indices = snp_indices[start:end]
                
                if len(cluster_positions) >= self.min_cluster_size:
                    clusters.append({
                        'cluster_id': len(clusters) + 1,
                        'chromosome': chrom,
                        'n_snps': len(cluster_positions),
                        'start_pos': int(cluster_positions[0]),
                        'end_pos': int(cluster_positions[-1]),
                        'span': int(cluster_positions[-1] - cluster_positions[0]),
                        'mean_score': float(cluster_scores.mean()),
                        'max_score': float(cluster_scores.max()),
                        'min_score': float(cluster_scores.min()),
                        'snp_indices': cluster_snp_indices.tolist(),
                        'positions': cluster_positions.tolist(),
                        'scores': cluster_scores.tolist()
                    })
        
        # Sort by mean score
        clusters.sort(key=lambda x: x['mean_score'], reverse=True)
        for i, cluster in enumerate(clusters):
            cluster['cluster_id'] = i + 1
        
        return clusters
    
    def find_clusters_by_index(self, df_selected):
        """Fallback: cluster by SNP_Index if no annotation"""
        snp_indices = df_selected['SNP_Index'].values
        scores = df_selected['Importance_Score'].values
        
        sorted_idx = np.argsort(snp_indices)
        sorted_positions = snp_indices[sorted_idx]
        sorted_scores = scores[sorted_idx]
        
        gaps = np.diff(sorted_positions)
        cluster_breaks = np.where(gaps > self.max_distance)[0] + 1
        
        cluster_starts = np.concatenate([[0], cluster_breaks])
        cluster_ends = np.concatenate([cluster_breaks, [len(sorted_positions)]])
        
        clusters = []
        for i, (start, end) in enumerate(zip(cluster_starts, cluster_ends)):
            cluster_positions = sorted_positions[start:end]
            cluster_scores = sorted_scores[start:end]
            
            if len(cluster_positions) >= self.min_cluster_size:
                clusters.append({
                    'cluster_id': i + 1,
                    'n_snps': len(cluster_positions),
                    'start_pos': int(cluster_positions[0]),
                    'end_pos': int(cluster_positions[-1]),
                    'span': int(cluster_positions[-1] - cluster_positions[0]),
                    'mean_score': float(cluster_scores.mean()),
                    'max_score': float(cluster_scores.max()),
                    'min_score': float(cluster_scores.min()),
                    'snp_indices': cluster_positions.tolist(),
                    'scores': cluster_scores.tolist()
                })
        
        clusters.sort(key=lambda x: x['mean_score'], reverse=True)
        for i, cluster in enumerate(clusters):
            cluster['cluster_id'] = i + 1
        
        return clusters
    
    def summarize_clusters(self, clusters):
        if not clusters:
            return {
                'n_clusters': 0,
                'total_snps': 0,
                'largest_cluster': 0,
                'mean_cluster_size': 0
            }
        
        return {
            'n_clusters': len(clusters),
            'total_snps': sum(c['n_snps'] for c in clusters),
            'largest_cluster': max(c['n_snps'] for c in clusters),
            'smallest_cluster': min(c['n_snps'] for c in clusters),
            'mean_cluster_size': np.mean([c['n_snps'] for c in clusters]),
            'median_cluster_size': np.median([c['n_snps'] for c in clusters]),
            'mean_span': np.mean([c['span'] for c in clusters]),
            'median_span': np.median([c['span'] for c in clusters])
        }


# ============================================================================
# VISUALIZATION 
# ============================================================================

class AnnotatedImportanceVisualizer:
    def __init__(self, output_dir, format='png'):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
    
    def plot_manhattan(self, df_annotated, entity, threshold_results=None):
        """Create Manhattan plot if genomic positions are available"""
        if 'chromosome' not in df_annotated.columns or 'bp' not in df_annotated.columns:
            print(f"  Skipping Manhattan plot (no genomic annotation)")
            return None
        
        print(f"  Creating Manhattan plot...")
        
        fig, ax = plt.subplots(figsize=(20, 6))
        
        # Prepare data
        df_plot = df_annotated[df_annotated['Importance_Score'] > 0].copy()
        
        # Create cumulative position for x-axis
        chromosomes = sorted(df_plot['chromosome'].unique(), key=natural_sort_key)
        df_plot['chrom_numeric'] = df_plot['chromosome'].apply(
            lambda x: chromosomes.index(x) if x in chromosomes else -1
        )
        
        # Calculate cumulative positions
        chrom_lengths = {}
        cumulative_pos = 0
        chrom_centers = {}
        
        for i, chrom in enumerate(chromosomes):
            chrom_data = df_plot[df_plot['chromosome'] == chrom]
            if len(chrom_data) > 0:
                chrom_max = chrom_data['bp'].max()
                chrom_lengths[chrom] = chrom_max
                chrom_centers[chrom] = cumulative_pos + chrom_max / 2
                
                # Add cumulative position
                mask = df_plot['chromosome'] == chrom
                df_plot.loc[mask, 'pos_cumulative'] = df_plot.loc[mask, 'bp'] + cumulative_pos
                
                cumulative_pos += chrom_max + 5e6  # Add spacing between chromosomes
        
        # Plot points
        colors = plt.cm.Set3(np.linspace(0, 1, len(chromosomes)))
        
        for i, chrom in enumerate(chromosomes):
            chrom_data = df_plot[df_plot['chromosome'] == chrom]
            if len(chrom_data) > 0:
                ax.scatter(chrom_data['pos_cumulative'], 
                          chrom_data['Importance_Score'],
                          c=[colors[i]], s=10, alpha=0.6, label=f'Chr {chrom}')
        
        # Add threshold lines
        if threshold_results:
            line_colors = ['red', 'blue', 'green']
            for i, (method, result) in enumerate(list(threshold_results.items())[:3]):
                ax.axhline(result['threshold'], color=line_colors[i],
                          linestyle='--', linewidth=2, alpha=0.7,
                          label=f"{method}: {result['threshold']:.2e}")
        
        # Formatting
        ax.set_xlabel('Chromosome', fontsize=14, fontweight='bold')
        ax.set_ylabel('Importance Score', fontsize=14, fontweight='bold')
        ax.set_title(f'{entity} - Manhattan Plot of SNP Importance', 
                    fontsize=16, fontweight='bold')
        
        # Set x-axis labels at chromosome centers
        ax.set_xticks([chrom_centers[c] for c in chromosomes])
        ax.set_xticklabels(chromosomes)
        
        ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.plots_dir / f'{entity}_manhattan.{self.format}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved Manhattan plot: {save_path}")
        return save_path
    
    def plot_chromosome_distribution(self, df_annotated, entity):
        """Plot distribution of important SNPs across chromosomes"""
        if 'chromosome' not in df_annotated.columns:
            return None
        
        print(f"  Creating chromosome distribution plot...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Get chromosome counts
        chrom_counts = df_annotated['chromosome'].value_counts()
        chrom_counts = chrom_counts.reindex(
            sorted(chrom_counts.index, key=natural_sort_key)
        )
        
        # 1. Bar plot of counts
        ax = axes[0]
        bars = ax.bar(range(len(chrom_counts)), chrom_counts.values, 
                     color=plt.cm.viridis(np.linspace(0, 1, len(chrom_counts))))
        ax.set_xlabel('Chromosome', fontsize=12)
        ax.set_ylabel('Number of Important SNPs', fontsize=12)
        ax.set_title(f'{entity} - SNP Distribution by Chromosome', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(chrom_counts)))
        ax.set_xticklabels(chrom_counts.index, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, chrom_counts.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}',
                   ha='center', va='bottom', fontsize=9)
        
        # 2. Mean importance score per chromosome
        ax = axes[1]
        chrom_mean_scores = df_annotated.groupby('chromosome')['Importance_Score'].mean()
        chrom_mean_scores = chrom_mean_scores.reindex(
            sorted(chrom_mean_scores.index, key=natural_sort_key)
        )
        
        bars = ax.bar(range(len(chrom_mean_scores)), chrom_mean_scores.values,
                     color=plt.cm.plasma(np.linspace(0, 1, len(chrom_mean_scores))))
        ax.set_xlabel('Chromosome', fontsize=12)
        ax.set_ylabel('Mean Importance Score', fontsize=12)
        ax.set_title(f'{entity} - Mean Importance by Chromosome', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(chrom_mean_scores)))
        ax.set_xticklabels(chrom_mean_scores.index, rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        save_path = self.plots_dir / f'{entity}_chromosome_distribution.{self.format}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved chromosome distribution: {save_path}")
        return save_path
    
    def plot_clusters_with_chromosomes(self, clusters, entity):
        """Enhanced cluster plot with chromosome information"""
        if not clusters:
            return None
        
        print(f"  Creating enhanced cluster plot...")
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Check if clusters have chromosome info
        has_chrom = 'chromosome' in clusters[0]
        
        # 1. Cluster positions
        ax = axes[0]
        
        if has_chrom:
            # Group by chromosome
            chrom_colors = {}
            unique_chroms = sorted(set(c['chromosome'] for c in clusters), 
                                  key=natural_sort_key)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_chroms)))
            
            for i, chrom in enumerate(unique_chroms):
                chrom_colors[chrom] = colors[i]
            
            for cluster in clusters:
                color = chrom_colors[cluster['chromosome']]
                ax.scatter(cluster['start_pos'], cluster['cluster_id'],
                          s=cluster['n_snps']*10, c=[color], alpha=0.7,
                          edgecolors='black')
                ax.plot([cluster['start_pos'], cluster['end_pos']],
                       [cluster['cluster_id'], cluster['cluster_id']],
                       color=color, linewidth=2, alpha=0.5)
                
                # Add chromosome label
                ax.text(cluster['start_pos'], cluster['cluster_id'],
                       f" Chr{cluster['chromosome']}", fontsize=8, 
                       va='center')
            
            ax.set_xlabel('Genomic Position (bp)', fontsize=12)
        else:
            # Fallback to SNP index
            for cluster in clusters:
                ax.scatter((cluster['start_pos'] + cluster['end_pos'])/2,
                          cluster['cluster_id'],
                          s=cluster['n_snps']*10, alpha=0.7,
                          edgecolors='black')
                ax.plot([cluster['start_pos'], cluster['end_pos']],
                       [cluster['cluster_id'], cluster['cluster_id']],
                       'k-', linewidth=2, alpha=0.5)
            
            ax.set_xlabel('SNP Index', fontsize=12)
        
        ax.set_ylabel('Cluster ID', fontsize=12)
        ax.set_title(f'{entity} - Genomic Clusters', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 2. Top clusters
        ax = axes[1]
        
        n_show = min(20, len(clusters))
        top_clusters = sorted(clusters, key=lambda x: x['n_snps'], reverse=True)[:n_show]
        
        if has_chrom:
            labels = [f"C{c['cluster_id']}\nChr{c['chromosome']}" for c in top_clusters]
        else:
            labels = [f"C{c['cluster_id']}" for c in top_clusters]
        
        sizes = [c['n_snps'] for c in top_clusters]
        
        bars = ax.barh(range(len(sizes)), sizes,
                      color=plt.cm.viridis([c['mean_score']/max(c['mean_score'] for c in clusters)
                                           for c in top_clusters]))
        
        ax.set_yticks(range(len(sizes)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Number of SNPs', fontsize=12)
        ax.set_title(f'{entity} - Top {n_show} Clusters', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, cluster) in enumerate(zip(bars, top_clusters)):
            width = bar.get_width()
            label_text = f' {width} SNPs'
            if has_chrom:
                label_text += f'\n {cluster["start_pos"]:,}-{cluster["end_pos"]:,} bp'
            ax.text(width, bar.get_y() + bar.get_height()/2, label_text,
                   ha='left', va='center', fontsize=7)
        
        plt.tight_layout()
        
        save_path = self.plots_dir / f'{entity}_clusters_annotated.{self.format}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved annotated cluster plot: {save_path}")
        return save_path
    
    def plot_distribution(self, scores, entity, threshold_results=None):
        """Standard distribution plot"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        scores_nonzero = scores[scores > 1e-10]
        
        # 1. Linear histogram
        ax = axes[0, 0]
        ax.hist(scores_nonzero, bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{entity} - Score Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if threshold_results:
            colors = ['red', 'blue', 'green', 'orange']
            for i, (method, result) in enumerate(threshold_results.items()):
                ax.axvline(result['threshold'], color=colors[i % len(colors)],
                          linestyle='--', linewidth=2, alpha=0.7,
                          label=f"{method}: {result['threshold']:.2e}")
            ax.legend(fontsize=9)
        
        # 2. Log histogram
        ax = axes[0, 1]
        log_scores = np.log10(scores_nonzero)
        ax.hist(log_scores, bins=100, alpha=0.7, edgecolor='black', color='orange')
        ax.set_xlabel('log10(Importance Score)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{entity} - Score Distribution (Log)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 3. Rank vs Score
        ax = axes[1, 0]
        sorted_scores = np.sort(scores)[::-1]
        n_plot = min(5000, len(sorted_scores))
        ax.plot(range(1, n_plot+1), sorted_scores[:n_plot], linewidth=2)
        ax.set_xlabel('SNP Rank', fontsize=12)
        ax.set_ylabel('Importance Score', fontsize=12)
        ax.set_title(f'{entity} - Top 5000 SNPs', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 4. Cumulative
        ax = axes[1, 1]
        sorted_all = np.sort(scores)[::-1]
        cumsum = np.cumsum(sorted_all)
        cumsum_norm = cumsum / cumsum[-1] * 100
        ax.plot(range(len(cumsum)), cumsum_norm, linewidth=2, color='purple')
        ax.set_xlabel('Number of SNPs', fontsize=12)
        ax.set_ylabel('Cumulative Importance (%)', fontsize=12)
        ax.set_title(f'{entity} - Cumulative Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, min(10000, len(cumsum)))
        
        plt.tight_layout()
        
        save_path = self.plots_dir / f'{entity}_distribution.{self.format}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved distribution plot: {save_path}")
        return save_path


# ============================================================================
# MAIN ANALYSIS PIPELINE WITH IMPROVED FILE DETECTION
# ============================================================================

def detect_file_format(scores_dir):
    """
    Detect the file format from the computation script output
    Returns: (format_type, entities, is_annotated, suffix)
    """
    scores_dir = Path(scores_dir)
    
    # Look for all CSV files
    all_csvs = list(scores_dir.glob('*.csv'))
    
    if not all_csvs:
        raise ValueError("No CSV files found in directory. Please provide CSV files with importance scores.")
    
    # Check for annotated files
    annotated_csvs = list(scores_dir.glob('*_annotated.csv'))
    is_annotated = len(annotated_csvs) > 0
    
    entities = []
    suffix = None
    
    if is_annotated:
        # Annotated format
        for csv in annotated_csvs:
            entity = csv.stem.replace('_annotated', '')
            if entity not in entities:
                entities.append(entity)
    else:
        # Look for patterns
        for csv in all_csvs:
            basename = csv.stem
            
            if '_all_snps_importance_' in basename:
                parts = basename.split('_all_snps_importance_')
                entity = parts[0]
                suffix = parts[1] if len(parts) > 1 else None
                
                if entity not in entities:
                    entities.append(entity)
            elif '_importance_scores' in basename:
                entity = basename.replace('_importance_scores', '')
                if entity not in entities:
                    entities.append(entity)
    
    format_type = "overall" if "overall" in entities else "disease_wise"
    
    return format_type, entities, is_annotated, suffix


def load_scores_from_csv(csv_file):
    """
    Load scores from CSV file - CSV is now the primary source!
    Returns: (scores_array, dataframe)
    """
    df = pd.read_csv(csv_file)
    
    # Verify required columns
    if 'Importance_Score' not in df.columns:
        raise ValueError(f"CSV file {csv_file} missing 'Importance_Score' column")
    
    # Ensure SNP_Index exists
    if 'SNP_Index' not in df.columns:
        # Create SNP_Index from row index
        df['SNP_Index'] = df.index
        print(f"  Warning: Created SNP_Index from row index")
    
    # Extract scores as numpy array
    scores = df['Importance_Score'].values
    
    return scores, df


def load_scores_and_metadata(scores_dir):
    """Load scores with improved file detection"""
    scores_dir = Path(scores_dir)
    
    print("LOADING SCORES FROM CSV FILES")
    print(f"{'='*70}")
    print(f"From: {scores_dir}")
    print(f"Note: NPY files are optional - CSV contains all data")
    
    # Detect file format
    format_type, entities, is_annotated, suffix = detect_file_format(scores_dir)
    
    if not entities:
        raise ValueError("No valid score files found in directory")
    
    print(f"\nDetected format:")
    print(f"  Type: {format_type}")
    print(f"  Entities: {entities}")
    print(f"  Annotated: {is_annotated}")
    if suffix:
        print(f"  File suffix: {suffix}")
    
    # Try to load metadata/config
    metadata = {'importance_scope': format_type, 'entities': entities}
    
    # Look for config or metadata files
    config_files = list(scores_dir.glob('*config*.txt')) + list(scores_dir.glob('*metadata*.json'))
    
    if config_files:
        config_file = config_files[0]
        print(f"\nFound configuration: {config_file.name}")
        
        if config_file.suffix == '.json':
            with open(config_file, 'r') as f:
                loaded_metadata = json.load(f)
                metadata.update(loaded_metadata)
        else:
            # Parse text config
            with open(config_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
    
    # Load data from CSV files
    scores_data = {}
    annotation_data = {}
    
    for entity in entities:
        # Find CSV file
        csv_files = []
        
        if is_annotated:
            csv_files = list(scores_dir.glob(f'{entity}_annotated.csv'))
        
        if not csv_files:
            if suffix:
                csv_files = list(scores_dir.glob(f'{entity}_all_snps_importance_{suffix}.csv'))
            else:
                csv_files = list(scores_dir.glob(f'{entity}*importance*.csv'))
        
        if not csv_files:
            print(f"  ⚠ No CSV file found for {entity}")
            continue
        
        csv_file = csv_files[0]
        
        try:
            # Load from CSV - this is now the PRIMARY and ONLY source we need!
            scores, df = load_scores_from_csv(csv_file)
            
            scores_data[entity] = scores
            annotation_data[entity] = df
            
            if 'chromosome' in df.columns:
                print(f"  ✓ Loaded {entity}: {len(scores):,} SNPs (with genomic annotation) from CSV")
            else:
                print(f"  ✓ Loaded {entity}: {len(scores):,} SNPs from CSV")
                
        except Exception as e:
            print(f"  ✗ Error loading {entity}: {e}")
            continue
    
    return scores_data, annotation_data, metadata


def analyze_entity_annotated(entity, scores, annotation_df, args, 
                             cluster_analyzer, visualizer):
    """Analyze entity with annotation support"""
    print(f"\n{'─'*70}")
    print(f"ANALYZING: {entity.upper()}")
    print(f"{'─'*70}")
    
    has_annotation = annotation_df is not None and 'chromosome' in annotation_df.columns
    
    if has_annotation:
        print(f"  Annotation: Available (chromosome, position, alleles)")
    else:
        print(f"  Annotation: Not available")
    
    # Compute statistics
    scores_nonzero = scores[scores > 1e-10]
    
    stats = {
        'entity': entity,
        'total_snps': len(scores),
        'nonzero_snps': len(scores_nonzero),
        'max_score': float(scores.max()),
        'min_score': float(scores[scores > 0].min()) if np.any(scores > 0) else 0,
        'mean_score': float(scores.mean()),
        'median_score': float(np.median(scores)),
        'std_score': float(scores.std()),
        'q95_score': float(np.percentile(scores, 95)),
        'q99_score': float(np.percentile(scores, 99)),
        'max_median_ratio': float(scores.max() / np.median(scores)) if np.median(scores) > 0 else 0,
        'has_annotation': has_annotation
    }
    
    print(f"\nStatistics:")
    print(f"  Total SNPs: {stats['total_snps']:,}")
    print(f"  Max score: {stats['max_score']:.6e}")
    print(f"  Median score: {stats['median_score']:.6e}")
    print(f"  Max/Median ratio: {stats['max_median_ratio']:.2f}x")
    
    if has_annotation:
        n_chroms = annotation_df['chromosome'].nunique()
        print(f"  Chromosomes: {n_chroms}")
    
    # Apply thresholds
    print(f"\nApplying threshold methods...")
    
    threshold_methods = [m.strip() for m in args.threshold_methods.split(',')]
    threshold_results = {}
    
    for method in threshold_methods:
        print(f"  - {method}...")
        
        if method == 'elbow':
            threshold, n_selected = ThresholdSelector.elbow_method(scores)
        elif method == 'percentile':
            threshold, n_selected = ThresholdSelector.percentile_method(
                scores, args.percentile_threshold
            )
        elif method == 'std':
            threshold, n_selected = ThresholdSelector.std_method(
                scores, args.std_multiplier
            )
        elif method == 'gap':
            threshold, n_selected = ThresholdSelector.gap_method(
                scores, args.min_gap_ratio
            )
        elif method == 'mixture':
            threshold, n_selected = ThresholdSelector.mixture_method(scores)
        else:
            print(f"    Warning: Unknown method '{method}', skipping")
            continue
        
        selected_mask = scores > threshold
        array_indices = np.where(selected_mask)[0]  # These are array positions (0,1,2...)
        selected_scores = scores[array_indices]
        
        # Get the ACTUAL SNP_Index values from the dataframe
        if annotation_df is not None and 'SNP_Index' in annotation_df.columns:
            # Use the actual SNP_Index values from the CSV
            selected_snp_indices = annotation_df.iloc[array_indices]['SNP_Index'].values
        else:
            # Fallback: use array indices if no annotation
            selected_snp_indices = array_indices
        
        
        threshold_results[method] = {
            'threshold': float(threshold),
            'n_selected': int(n_selected),
            'percentage': float(n_selected / len(scores) * 100),
            'selected_indices': selected_snp_indices,  # Real SNP_Index values, already sorted!
            'selected_scores': selected_scores,  # Already in descending order!
        }
        
        print(f"    Selected: {n_selected:,} SNPs ({threshold_results[method]['percentage']:.3f}%)")
    
    # Clustering with annotation support
    print(f"\nPerforming genomic clustering...")
    
    cluster_method = 'elbow' if 'elbow' in threshold_results else threshold_methods[0]
    selected_indices = threshold_results[cluster_method]['selected_indices']
    selected_scores = threshold_results[cluster_method]['selected_scores']
    
    # Create DataFrame for selected SNPs
    if has_annotation:
        df_selected = annotation_df[annotation_df['SNP_Index'].isin(selected_indices)].copy()
        clusters = cluster_analyzer.find_clusters_annotated(df_selected)
    else:
        df_selected = pd.DataFrame({
            'SNP_Index': selected_indices,
            'Importance_Score': selected_scores
        })
        clusters = cluster_analyzer.find_clusters_by_index(df_selected)
    
    cluster_summary = cluster_analyzer.summarize_clusters(clusters)
    
    print(f"  Found {cluster_summary['n_clusters']} clusters")
    if cluster_summary['n_clusters'] > 0:
        print(f"  Largest cluster: {cluster_summary['largest_cluster']} SNPs")
        if has_annotation and clusters:
            chrom_dist = {}
            for c in clusters:
                chrom = c.get('chromosome', 'unknown')
                chrom_dist[chrom] = chrom_dist.get(chrom, 0) + 1
            print(f"  Clusters per chromosome: {chrom_dist}")
    
    # Visualizations
    if args.create_plots:
        print(f"\nCreating visualizations...")
        
        # Standard plots
        visualizer.plot_distribution(scores, entity, threshold_results)
        
        # Enhanced plots if annotated
        if has_annotation:
            # For Manhattan, need full annotation_df with scores
            full_df = annotation_df.copy()
            visualizer.plot_manhattan(full_df, entity, threshold_results)
            visualizer.plot_chromosome_distribution(df_selected, entity)
        
        # Cluster plots
        if clusters:
            visualizer.plot_clusters_with_chromosomes(clusters, entity)
    
    return {
        'stats': stats,
        'thresholds': threshold_results,
        'clusters': clusters,
        'cluster_summary': cluster_summary
    }


def save_results(all_results, metadata, output_dir, annotation_data):
    """Save results with annotation"""
    output_dir = Path(output_dir)
    
    filtered_dir = output_dir / 'filtered_snps'
    clusters_dir = output_dir / 'clusters'
    reports_dir = output_dir / 'reports'
    
    for d in [filtered_dir, clusters_dir, reports_dir]:
        d.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    
    # Save filtered SNPs
    for entity, result in all_results.items():
        df_annotation = annotation_data.get(entity)
        has_annotation = df_annotation is not None and 'chromosome' in df_annotation.columns
        
        for method, tres in result['thresholds'].items():
            # Create base DataFrame with REAL SNP_Index values
            filtered_df = pd.DataFrame({
                'SNP_Index': tres['selected_indices'],  # These are now real SNP_Index values!
                'Importance_Score': tres['selected_scores'],
                'Rank': np.arange(1, len(tres['selected_indices']) + 1)
            })
            
            # Add annotation if available
            if has_annotation:
                # Get annotation columns
                anno_cols = ['SNP_Index', 'chromosome', 'bp']
                if 'snp_id' in df_annotation.columns:
                    anno_cols.append('snp_id')
                if 'ref_allele' in df_annotation.columns:
                    anno_cols.extend(['ref_allele', 'alt_allele'])
                
                filtered_df = filtered_df.merge(
                    df_annotation[anno_cols],
                    on='SNP_Index',
                    how='left'
                )
            
            filtered_path = filtered_dir / f'{entity}_{method}_filtered.csv'
            filtered_df.to_csv(filtered_path, index=False)
            print(f"  ✓ Saved: {filtered_path}")
    
    # Save clusters
    for entity, result in all_results.items():
        if result['clusters']:
            cluster_path = clusters_dir / f'{entity}_clusters.json'
            with open(cluster_path, 'w') as f:
                json.dump({
                    'summary': result['cluster_summary'],
                    'clusters': result['clusters']
                }, f, indent=2)
            print(f"  ✓ Saved: {cluster_path}")
    
    # Generate report
    report_path = reports_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SNP FEATURE IMPORTANCE - ANNOTATED ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if 'timestamp' in metadata:
            f.write(f"Scores Computed: {metadata['timestamp']}\n")
        f.write(f"Importance Scope: {metadata['importance_scope']}\n\n")
        
        # Check if any entity has annotation
        any_annotated = any(r['stats'].get('has_annotation', False) 
                          for r in all_results.values())
        f.write(f"Genomic Annotation: {'Yes' if any_annotated else 'No'}\n\n")
        
        f.write("RESULTS BY ENTITY\n")
        f.write("=" * 80 + "\n\n")
        
        for entity, result in all_results.items():
            f.write(f"{entity.upper()}\n")
            f.write("-" * 80 + "\n")
            
            for key, value in result['stats'].items():
                if key == 'entity':
                    continue
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.6e}\n")
                elif isinstance(value, bool):
                    f.write(f"  {key}: {'Yes' if value else 'No'}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write(f"\nThreshold Results:\n")
            for method, tres in result['thresholds'].items():
                f.write(f"  {method}:\n")
                f.write(f"    Threshold: {tres['threshold']:.6e}\n")
                f.write(f"    SNPs selected: {tres['n_selected']:,} ({tres['percentage']:.3f}%)\n")
            
            f.write(f"\nGenomic Clusters:\n")
            csummary = result['cluster_summary']
            f.write(f"  Number of clusters: {csummary['n_clusters']}\n")
            if csummary['n_clusters'] > 0:
                f.write(f"  Largest cluster: {csummary['largest_cluster']} SNPs\n")
            
            f.write("\n\n")
    
    print(f"  ✓ Saved report: {report_path}")
    
    # Summary table
    summary_data = []
    for entity, result in all_results.items():
        row = {'Entity': entity}
        row.update({k: v for k, v in result['stats'].items() if k != 'entity'})
        for method in result['thresholds']:
            row[f'{method}_n_selected'] = result['thresholds'][method]['n_selected']
        row['N_Clusters'] = result['cluster_summary']['n_clusters']
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = reports_dir / 'summary_table.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Saved summary: {summary_path}")


def main():
    args = parse_args()
    
    print("SNP FEATURE IMPORTANCE - ANNOTATED ANALYSIS")
    print("="*80)
    
    # Load data
    try:
        scores_data, annotation_data, metadata = load_scores_and_metadata(args.scores_dir)
    except Exception as e:
        print(f"\nERROR loading scores: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure directory contains CSV files with 'Importance_Score' column")
        print("  - Files should be from compute_importance_scores.py or snp_annotator.py")
        return
    
    if not scores_data:
        print("ERROR: No scores found!")
        return
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = Path(args.scores_dir) / f'analysis_{timestamp}'
    
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {args.output_dir}")
    
    # Initialize components
    cluster_analyzer = GenomicClusterAnalyzer(
        max_distance=args.cluster_distance,
        min_cluster_size=args.min_cluster_size
    )
    visualizer = AnnotatedImportanceVisualizer(args.output_dir, args.plot_format)
    
    # Analyze each entity
    all_results = {}
    entities = list(scores_data.keys())
    
    for entity in entities:
        scores = scores_data[entity]
        annotation_df = annotation_data[entity]
        result = analyze_entity_annotated(entity, scores, annotation_df, args,
                                         cluster_analyzer, visualizer)
        all_results[entity] = result
    
    # Save results
    save_results(all_results, metadata, args.output_dir, annotation_data)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    
    # Check if annotation was used
    any_annotated = any(r['stats'].get('has_annotation', False) 
                       for r in all_results.values())
    if any_annotated:
        print(f"\n Enhanced visualizations created with genomic annotation:")
        print(f"   • Manhattan plots showing chromosome positions")
        print(f"   • Chromosome distribution analysis")
        print(f"   • Cluster plots with genomic coordinates")
    else:
        print(f"\n Tip: Run snp_annotator.py first to enable enhanced visualizations!")


if __name__ == '__main__':
    main()