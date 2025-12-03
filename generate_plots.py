"""
Generate comprehensive comparison plots between Python port and R DESeq2. 
Uses the SAME plotting style as the original generate_plots.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deseq2_py.plotting import plotMA, plotVolcano, plotPCA
from deseq2_py.size_factors import estimate_size_factors
from deseq2_py.transformations import vst

os.makedirs("plots", exist_ok=True)

# ===========================
# 1.  LOAD DATA
# ===========================
print("Loading data...")

py = pd.read_csv("data/results_python.csv", index_col=0)
r = pd.read_csv("data/deseq2_results.csv", index_col=0)
counts_df = pd.read_csv("data/counts.csv", index_col=0)
coldata_df = pd.read_csv("data/coldata.csv", index_col=0)

# Prepare merged dataframe
r_renamed = r[["baseMean", "log2FoldChange", "pvalue", "padj"]].rename(columns={
    "baseMean": "baseMean_r",
    "log2FoldChange": "log2FoldChange_r",
    "pvalue": "pvalue_r",
    "padj": "padj_r",
})

py_renamed = py[["baseMean", "log2FoldChange", "pvalue", "padj"]].rename(columns={
    "baseMean": "baseMean_py",
    "log2FoldChange": "log2FoldChange_py",
    "pvalue": "pvalue_py",
    "padj": "padj_py",
})

merged = py_renamed.join(r_renamed, how="inner")
print(f"Merged {len(merged)} genes for comparison.")

# Filter extreme values for cleaner plots
plot_df = merged.copy()
plot_df = plot_df[np.abs(plot_df["log2FoldChange_py"]) < 12]
plot_df = plot_df[np.abs(plot_df["log2FoldChange_r"]) < 12]

# ===========================
# 2. MA PLOTS (Improved style)
# ===========================
print("\n1. Generating MA plot comparison...")
 # MA Plot
plotMA(res, alpha=0.05)
plt.savefig("plots/ma_plot.png")
plt.close()
print("Saved MA plot")


# ===========================
# 3.  VOLCANO PLOTS
# ===========================
print("\n2.  Generating Volcano plot comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

configs = [
    (axes[0], "log2FoldChange_py", "pvalue_py", "padj_py", "Python Port"),
    (axes[1], "log2FoldChange_r", "pvalue_r", "padj_r", "Original DESeq2 (R)")
]

for ax, x_col, p_col, padj_col, title in configs:
    lfc = plot_df[x_col]
    pval = plot_df[p_col]
    padj = plot_df[padj_col]
    
    neglogp = -np.log10(pval + 1e-300)
    neglogp = np.clip(neglogp, 0, 50)
    
    sig = padj < 0.05
    
    ax.scatter(lfc[~sig], neglogp[~sig], s=5, color="lightgray", alpha=0.5, rasterized=True)
    ax.scatter(lfc[sig], neglogp[sig], s=8, color="darkred", alpha=0.6, rasterized=True)
    
    ax.axvline(-1, color="k", linestyle="--", lw=0.5)
    ax.axvline(1, color="k", linestyle="--", lw=0.5)
    ax.axhline(-np.log10(0.05), color="k", linestyle="--", lw=0.5)
    
    ax.set_xlabel("log2 Fold Change")
    if ax == axes[0]:
        ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)

plt.suptitle("Volcano Plot Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("plots/compare_volcano_new.png", dpi=300)
plt.close()
print("  Saved: plots/compare_volcano_new.png")


# ===========================
# 4. LFC CORRELATION
# ===========================
print("\n3.  Generating LFC correlation plot...")

df_clean = merged[merged["baseMean_py"] > 10].dropna(subset=["log2FoldChange_py", "log2FoldChange_r"])

x = df_clean["log2FoldChange_r"]
y = df_clean["log2FoldChange_py"]
r_val = x.corr(y)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(x, y, s=5, alpha=0.3, color="purple", rasterized=True)

mn, mx = -10, 10
ax.plot([mn, mx], [mn, mx], 'k-', alpha=0.75)
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)

ax.set_xlabel("R DESeq2 log2FoldChange")
ax.set_ylabel("Python Port log2FoldChange")
ax.set_title(f"Log2 Fold Change Correlation\n(baseMean > 10, Pearson r = {r_val:.4f})")
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig("plots/compare_lfc_correlation_new.png", dpi=300)
plt.close()
print("  Saved: plots/compare_lfc_correlation_new.png")


# ===========================
# 5. TOP OVERLAP
# ===========================
print("\n4. Generating top genes overlap plot...")

Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
overlaps = []

df_clean = merged.dropna(subset=["pvalue_py", "pvalue_r"])

for n in Ns:
    top_py = set(df_clean.nsmallest(n, "pvalue_py").index)
    top_r = set(df_clean.nsmallest(n, "pvalue_r").index)
    overlap_pct = len(top_py & top_r) / n * 100
    overlaps.append(overlap_pct)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Ns, overlaps, marker='o', linestyle='-', color='green', linewidth=2, markersize=8)
ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel("Top N Genes (by p-value)")
ax.set_ylabel("% Overlap with R DESeq2")
ax.set_title("Agreement of Top Differentially Expressed Genes")
ax.set_ylim(0, 105)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)

for n, o in zip(Ns, overlaps):
    ax.annotate(f'{o:.0f}%', (n, o), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("plots/compare_overlap_new.png", dpi=300)
plt.close()
print("  Saved: plots/compare_overlap_new.png")

# ===========================
# 6. PCA PLOT
# ===========================
print("\n5. Generating PCA plot...")

size_factors = estimate_size_factors(counts_df.values)
vst_data = vst(counts_df.values, size_factors=size_factors)

# Select top 500 variable genes
gene_vars = np.var(vst_data, axis=1)
top_idx = np.argsort(gene_vars)[-500:]
vst_subset = vst_data[top_idx, :]

# Center data
vst_centered = vst_subset - vst_subset.mean(axis=1, keepdims=True)

# PCA via SVD
U, S, Vt = np.linalg.svd(vst_centered.T, full_matrices=False)
pc1, pc2 = U[:, 0], U[:, 1]
var_exp = (S ** 2) / np.sum(S ** 2)

fig, ax = plt.subplots(figsize=(8, 6))

conditions = coldata_df['dex'].values
for cond, color, label in [('trt', 'red', 'Treated'), ('untrt', 'gray', 'Untreated')]:
    mask = conditions == cond
    ax.scatter(pc1[mask], pc2[mask], c=color, s=100,
               label=label, alpha=0.8, edgecolors='black')

ax.set_xlabel(f'PC1: {var_exp[0]*100:.1f}% variance')
ax.set_ylabel(f'PC2: {var_exp[1]*100:.1f}% variance')
ax.set_title('PCA of VST-Transformed Data')
ax.legend()

ax.axhline(0, color='gray', linestyle='--', lw=0.5)
ax.axvline(0, color='gray', linestyle='--', lw=0.5)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("plots/pca_comparison.png", dpi=300)
plt.close()
print("  Saved: plots/pca_comparison.png")


# ===========================
# 7. SUMMARY
# ===========================
print("\n" + "="*50)
print("COMPARISON SUMMARY")
print("="*50)

sig_py = (merged["padj_py"] < 0.05).sum()
sig_r = (merged["padj_r"] < 0.05).sum()
both_sig = ((merged["padj_py"] < 0.05) & (merged["padj_r"] < 0.05)).sum()

lfc_corr = merged["log2FoldChange_py"].corr(merged["log2FoldChange_r"])
lfc_corr_filtered = df_clean["log2FoldChange_py"].corr(df_clean["log2FoldChange_r"])

print(f"  Total genes: {len(merged)}")
print(f"  Significant (Python): {sig_py}")
print(f"  Significant (R): {sig_r}")
print(f"  Significant in both: {both_sig}")
print(f"  LFC correlation (all): {lfc_corr:.4f}")
print(f"  LFC correlation (baseMean>10): {lfc_corr_filtered:.4f}")
print(f"  Top 100 overlap: {overlaps[Ns.index(100)]:.1f}%")
print("="*50)

print("\nDone! All plots saved to 'plots/' folder.")
