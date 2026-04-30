"""
Generate two separate figures (training time and GPU memory) for the CNN-Mamba GWAS paper.

Reads Table 2 from Sheet5 of results_mamba.xlsx and produces:
  - figure_training_time.{pdf,png}: per-epoch training time vs sequence length
  - figure_gpu_memory.{pdf,png}:    peak GPU memory vs sequence length, with the
                                    RTX 3090 24 GB ceiling marked.

In both figures the transformer is plotted only at 1,224 tokens (its OOM-limited
maximum); longer configurations are annotated as OOM to make the architectural
cliff visible.

Usage:
  python plot_table2.py
Outputs are written to the same directory as the input Excel file.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
XLSX_PATH = Path(
    "/vol/research/fmodal_mmmed/Codes/5_disease_experiments/"
    "GWAS_Results_Mamba/results_mamba.xlsx"
)
SHEET = "Sheet5"

df = pd.read_excel(XLSX_PATH, sheet_name=SHEET)

# Strip whitespace from column names so small header variations don't break things.
df.columns = [c.strip() for c in df.columns]
print("Columns found:", list(df.columns))
print(df)

# Map flexible column lookup so the script tolerates minor renaming in the sheet.
def find_col(df, *needles):
    for c in df.columns:
        low = c.lower()
        if all(n.lower() in low for n in needles):
            return c
    raise KeyError(f"No column matching {needles}. Available: {list(df.columns)}")

col_arch  = find_col(df, "architecture")
col_seq   = find_col(df, "sequence", "length")
col_time  = find_col(df, "time", "epoch")          # per-epoch minutes
col_mem   = find_col(df, "memory")                 # peak GB

mamba       = df[df[col_arch].str.contains("Mamba",       case=False, na=False)].sort_values(col_seq)
transformer = df[df[col_arch].str.contains("Transformer", case=False, na=False)].sort_values(col_seq)

# ---------------------------------------------------------------------------
# 2. Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
})

# Match the radar plots in Figs 1–2: red = Mamba, blue = Transformer.
COLOR_MAMBA       = "#C8202B"   # red
COLOR_TRANSFORMER = "#1F4E9D"   # blue
COLOR_OOM         = "#9A9A9A"   # grey for the OOM annotations
COLOR_CEILING     = "#444444"   # dashed GPU ceiling line

seq_lengths = [1224, 2448, 4895, 9789]
x_pos       = np.arange(len(seq_lengths))
bar_w       = 0.35

out_dir = XLSX_PATH.parent          # save next to the input Excel file

# ---------------------------------------------------------------------------
# 3. Figure 1 — Per-epoch training time
# ---------------------------------------------------------------------------
mamba_time = [mamba.loc[mamba[col_seq] == sl, col_time].values[0] for sl in seq_lengths]
trans_time_at_1224 = transformer.loc[transformer[col_seq] == 1224, col_time].values[0]

fig1, ax1 = plt.subplots(figsize=(6.5, 4.5))

ax1.bar(x_pos - bar_w/2, mamba_time, bar_w,
        color=COLOR_MAMBA, label="CNN-Mamba", edgecolor="white", linewidth=0.6)

# Transformer bar only at 1,224
ax1.bar(x_pos[0] + bar_w/2, trans_time_at_1224, bar_w,
        color=COLOR_TRANSFORMER, label="CNN-Transformer", edgecolor="white", linewidth=0.6)

# OOM markers for transformer at the longer lengths
for i in range(1, len(seq_lengths)):
    ax1.bar(x_pos[i] + bar_w/2, 2, bar_w,
            color="none", edgecolor=COLOR_OOM, linewidth=1.0, hatch="///")
    ax1.text(x_pos[i] + bar_w/2, 4.5, "OOM",
             ha="center", va="bottom", fontsize=8.5, color=COLOR_OOM, rotation=90)

# Numeric labels on bars
for xi, v in zip(x_pos - bar_w/2, mamba_time):
    ax1.text(xi, v + 0.8, f"{v}", ha="center", va="bottom", fontsize=9, color=COLOR_MAMBA)
ax1.text(x_pos[0] + bar_w/2, trans_time_at_1224 + 0.8, f"{trans_time_at_1224}",
         ha="center", va="bottom", fontsize=9, color=COLOR_TRANSFORMER)

ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"{s:,}" for s in seq_lengths])
ax1.set_xlabel("Sequence Length (tokens)")
ax1.set_ylabel("Training Time per Epoch (minutes)")
#ax1.set_title("Per-epoch training time")
ax1.set_ylim(0, max(trans_time_at_1224, max(mamba_time)) * 1.25)
ax1.grid(axis="y", linestyle=":", alpha=0.5)
ax1.legend(loc="upper left")

fig1.tight_layout()
fig1_pdf = out_dir / "figure_training_time.pdf"
fig1_png = out_dir / "figure_training_time.png"
fig1.savefig(fig1_pdf, bbox_inches="tight")
fig1.savefig(fig1_png, dpi=300, bbox_inches="tight")
print(f"Saved: {fig1_pdf}")
print(f"Saved: {fig1_png}")

# ---------------------------------------------------------------------------
# 4. Figure 2 — Peak GPU memory
# ---------------------------------------------------------------------------
mamba_mem = [mamba.loc[mamba[col_seq] == sl, col_mem].values[0] for sl in seq_lengths]
trans_mem_at_1224 = transformer.loc[transformer[col_seq] == 1224, col_mem].values[0]

GPU_CEILING = 24  # RTX 3090 VRAM in GB

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

ax2.bar(x_pos - bar_w/2, mamba_mem, bar_w,
        color=COLOR_MAMBA, label="CNN-Mamba", edgecolor="white", linewidth=0.6)
ax2.bar(x_pos[0] + bar_w/2, trans_mem_at_1224, bar_w,
        color=COLOR_TRANSFORMER, label="CNN-Transformer", edgecolor="white", linewidth=0.6)

# OOM markers for transformer at the longer lengths — drawn up to the ceiling
for i in range(1, len(seq_lengths)):
    ax2.bar(x_pos[i] + bar_w/2, GPU_CEILING, bar_w,
            color="none", edgecolor=COLOR_OOM, linewidth=1.0, hatch="///", alpha=0.7)
    ax2.text(x_pos[i] + bar_w/2, GPU_CEILING * 0.5, "OOM",
             ha="center", va="center", fontsize=9, color=COLOR_OOM, rotation=90)

# 24 GB GPU ceiling
ax2.axhline(GPU_CEILING, color=COLOR_CEILING, linestyle="--", linewidth=1.0)
ax2.text(len(seq_lengths) - 0.5, GPU_CEILING + 0.4,
         "RTX 3090 ceiling (24 GB)",
         ha="right", va="bottom", fontsize=8.5, color=COLOR_CEILING, style="italic")

# Numeric labels
for xi, v in zip(x_pos - bar_w/2, mamba_mem):
    ax2.text(xi, v + 0.4, f"~{v}", ha="center", va="bottom", fontsize=9, color=COLOR_MAMBA)
ax2.text(x_pos[0] + bar_w/2, trans_mem_at_1224 + 0.4, f"~{trans_mem_at_1224}",
         ha="center", va="bottom", fontsize=9, color=COLOR_TRANSFORMER)

ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"{s:,}" for s in seq_lengths])
ax2.set_xlabel("Sequence Length (tokens)")
ax2.set_ylabel("Peak GPU Memory (GB)")
#ax2.set_title("Peak GPU memory consumption")
ax2.set_ylim(0, GPU_CEILING + 4)
ax2.grid(axis="y", linestyle=":", alpha=0.5)
ax2.legend(loc="upper left")

fig2.tight_layout()
fig2_pdf = out_dir / "figure_gpu_memory.pdf"
fig2_png = out_dir / "figure_gpu_memory.png"
fig2.savefig(fig2_pdf, bbox_inches="tight")
fig2.savefig(fig2_png, dpi=300, bbox_inches="tight")
print(f"Saved: {fig2_pdf}")
print(f"Saved: {fig2_png}")