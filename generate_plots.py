"""
Generate comprehensive comparison plots between Python port and R DESeq2.
Uses the new plotting functions alongside side-by-side comparisons.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deseq2_py.size_factors import estimate_size_factors
from deseq2_py.transformations import vst
from deseq2_py.plotting import plotMA, plotVolcano, plotPCA, plotDispEsts

os.makedirs("plots", exist_ok=True)

# ===========================
# 1. LOAD DATA
# ===========================
print("Loading data...")

# Load Python results
py = pd.read_csv("data/results_python.csv", index_col=0)

# Load R DESeq2 results
r = pd.read_csv("data/deseq2_results.csv", index_col=0)

# Load counts and metadata for PCA
counts_df = pd.read_csv("data/counts.csv", index_col=0)
coldata_df = pd.read_csv("data/coldata.csv", index_col=0)

# Load R dispersions for comparison
r_disp = pd.read_csv("data/r_dispersions.csv", index_col=0)

# Prepare merged dataframe
r_cols = r[["baseMean", "log2FoldChange", "pvalue", "padj"]].rename(columns={
    "baseMean": "baseMean_r",
    "log2FoldChange": "log2FoldChange_r",
    "pvalue": "pvalue_r",
    "padj": "padj_r",
})

py_cols = py[["baseMean", "log2FoldChange", "pvalue", "padj"]].rename(columns={
    "baseMean": "baseMean_py",
    "log2FoldChange": "log2FoldChange_py",
    "pvalue": "pvalue_py",
    "padj": "padj_py",
})

merged = py_cols.join(r_cols, how="inner")
print(f"Merged {len(merged)} genes for comparison.")


# ===========================
# 2. SIDE-BY-SIDE MA PLOTS
# ===========================
print("\n1. Generating MA plot comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

# Python MA plot
ax = axes[0]
py_res = {
    "baseMean": py["baseMean"].values,
    "log2FoldChange": py["log2FoldChange"].values,
    "padj": py["padj"].values
}
plotMA(py_res, alpha=0.05, ax=ax, main="Python Port")

# R MA plot
ax = axes[1]
r_res = {
    "baseMean": r["baseMean"].values,
    "log2FoldChange": r["log2FoldChange"].values,
    "padj": r["padj"].values
}
plotMA(r_res, alpha=0.05, ax=ax, main="Original DESeq2 (R)")

plt.suptitle("MA Plot Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("plots/compare_ma_new.png", dpi=300)
plt.close()
print("  Saved: plots/compare_ma_new.png")


# ===========================
# 3. SIDE-BY-SIDE VOLCANO PLOTS
# ===========================
print("\n2. Generating Volcano plot comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Python Volcano
ax = axes[0]
py_res = {
    "log2FoldChange": py["log2FoldChange"].values,
    "pvalue": py["pvalue"].values,
    "padj": py["padj"].values
}
plotVolcano(py_res, alpha=0.05, lfc_threshold=1.0, ax=ax, main="Python Port")
ax.set_ylim(0, 50)

# R Volcano
ax = axes[1]
r_res = {
    "log2FoldChange": r["log2FoldChange"].values,
    "pvalue": r["pvalue"].values,
    "padj": r["padj"].values
}
plotVolcano(r_res, alpha=0.05, lfc_threshold=1.0, ax=ax, main="Original DESeq2 (R)")
ax.set_ylim(0, 50)

plt.suptitle("Volcano Plot Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("plots/compare_volcano_new.png", dpi=300)
plt.close()
print("  Saved: plots/compare_volcano_new.png")


# ===========================
# 4. DISPERSION COMPARISON
# ===========================
print("\n3. Generating Dispersion plot comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Python dispersions
ax = axes[0]
py_basemean = py["baseMean"].values
py_disp = py["dispersion"].values if "dispersion" in py.columns else None

if py_disp is not None:
    plotDispEsts(py_basemean, py_disp, ax=ax, main="Python Port - Dispersions")
else:
    ax.text(0.5, 0.5, "No dispersion data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Python Port - Dispersions")

# R dispersions
ax = axes[1]
if "dispersion" in r_disp.columns:
    r_basemean = r["baseMean"].values
    r_disp_vals = r_disp["dispersion"].values
    plotDispEsts(r_basemean, r_disp_vals, ax=ax, main="Original DESeq2 (R) - Dispersions")
elif "dispGeneEst" in r_disp.columns:
    r_basemean = r["baseMean"].reindex(r_disp.index).values
    r_disp_vals = r_disp["dispGeneEst"].values
    plotDispEsts(r_basemean, r_disp_vals, ax=ax, main="Original DESeq2 (R) - Dispersions")
else:
    ax.text(0.5, 0.5, "No dispersion data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Original DESeq2 (R) - Dispersions")

plt.suptitle("Dispersion Estimates Comparison", fontsize=14)
plt.tight_layout()
plt.savefig("plots/compare_dispersions.png", dpi=300)
plt.close()
print("  Saved: plots/compare_dispersions.png")


# ===========================
# 5. PCA COMPARISON
# ===========================
print("\n4. Generating PCA plot...")

size_factors = estimate_size_factors(counts_df.values)
vst_data = vst(counts_df.values, size_factors=size_factors)

fig, ax = plt.subplots(figsize=(8, 6))
plotPCA(vst_data, sample_info=coldata_df, color_by="dex", ax=ax, main="PCA of VST-Transformed Data")
plt.tight_layout()
plt.savefig("plots/pca_vst.png", dpi=300)
plt.close()
print("  Saved: plots/pca_vst.png")


# ===========================
# 6. LFC CORRELATION
# ===========================
print("\n5. Generating LFC correlation plot...")

df_clean = merged[merged["baseMean_py"] > 10].dropna(subset=["log2FoldChange_py", "log2FoldChange_r"])

x = df_clean["log2FoldChange_r"]
y = df_clean["log2FoldChange_py"]
r_val = x.corr(y)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(x, y, s=5, alpha=0.3, color="purple")

mn, mx = -10, 10
ax.plot([mn, mx], [mn, mx], "k-", alpha=0.75, zorder=0, label="y=x")
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)

ax.set_xlabel("R DESeq2 log2FoldChange")
ax.set_ylabel("Python Port log2FoldChange")
ax.set_title(f"Log2 Fold Change Correlation\n(Genes with baseMean > 10, Pearson r = {r_val:.4f})")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("plots/compare_lfc_scatter.png", dpi=300)
plt.close()
print("  Saved: plots/compare_lfc_scatter.png")


# ===========================
# 7. P-VALUE CORRELATION
# ===========================
print("\n6. Generating p-value correlation plot...")

df_clean = merged.dropna(subset=["pvalue_py", "pvalue_r"])
df_clean = df_clean[(df_clean["pvalue_py"] > 0) & (df_clean["pvalue_r"] > 0)]

x = -np.log10(df_clean["pvalue_r"])
y = -np.log10(df_clean["pvalue_py"])

x = np.clip(x, 0, 50)
y = np.clip(y, 0, 50)

r_val = pd.Series(x).corr(pd.Series(y))

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(x, y, s=5, alpha=0.3, color="darkgreen")

mn, mx = 0, 50
ax.plot([mn, mx], [mn, mx], "k-", alpha=0.75, zorder=0, label="y=x")
ax.set_xlim(mn, mx)
ax.set_ylim(mn, mx)

ax.set_xlabel("R DESeq2 -log10(p-value)")
ax.set_ylabel("Python Port -log10(p-value)")
ax.set_title(f"P-value Correlation\n(Pearson r = {r_val:.4f})")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("plots/compare_pvalue_scatter.png", dpi=300)
plt.close()
print("  Saved: plots/compare_pvalue_scatter.png")


# ===========================
# 8. TOP GENE OVERLAP
# ===========================
print("\n7. Generating top genes overlap plot...")

Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
overlaps = []

df_clean = merged.dropna(subset=["pvalue_py", "pvalue_r"])

for n in Ns:
    top_py = set(df_clean.nsmallest(n, "pvalue_py").index)
    top_r = set(df_clean.nsmallest(n, "pvalue_r").index)
    overlap_pct = len(top_py & top_r) / n * 100
    overlaps.append(overlap_pct)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(Ns, overlaps, marker="o", linestyle="-", color="green", linewidth=2, markersize=8)
ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="Perfect agreement")

ax.set_xlabel("Top N Genes (by p-value)")
ax.set_ylabel("% Overlap with R DESeq2")
ax.set_title("Agreement of Top Differentially Expressed Genes")
ax.set_ylim(0, 105)
ax.set_xscale("log")
ax.grid(True, alpha=0.3)
ax.legend()

for n, o in zip(Ns, overlaps):
    ax.annotate(f"{o:.0f}%", (n, o), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

plt.tight_layout()
plt.savefig("plots/compare_top_genes_overlap.png", dpi=300)
plt.close()
print("  Saved: plots/compare_top_genes_overlap.png")


# ===========================
# 9. SUMMARY STATISTICS
# ===========================
print("\n8. Generating summary statistics...")

stats = {
    "Total genes": len(merged),
    "Significant (Python, padj<0.05)": (merged["padj_py"] < 0.05).sum(),
    "Significant (R, padj<0.05)": (merged["padj_r"] < 0.05).sum(),
    "LFC correlation (all)": merged["log2FoldChange_py"].corr(merged["log2FoldChange_r"]),
}

df_big = merged[merged["baseMean_py"] > 10]
stats["LFC correlation (baseMean>10)"] = df_big["log2FoldChange_py"].corr(df_big["log2FoldChange_r"])

both_sig = ((merged["padj_py"] < 0.05) & (merged["padj_r"] < 0.05)).sum()
stats["Significant in both"] = both_sig

valid = (merged["log2FoldChange_py"] != 0) & (merged["log2FoldChange_r"] != 0)
sign_agree = (
    np.sign(merged.loc[valid, "log2FoldChange_py"]) ==
    np.sign(merged.loc[valid, "log2FoldChange_r"])
).mean() * 100
stats["Sign agreement (%)"] = sign_agree

print("\n" + "="*50)
print("COMPARISON SUMMARY")
print("="*50)
for key, val in stats.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.4f}")
    else:
        print(f"  {key}: {val}")
print("="*50)

# Save summary
with open("plots/comparison_summary.txt", "w") as f:
    f.write("Python Port vs R DESeq2 Comparison Summary\n")
    f.write("="*50 + "\n\n")
    for key, val in stats.items():
        if isinstance(val, float):
            f.write(f"{key}: {val:.4f}\n")
        else:
            f.write(f"{key}: {val}\n")

print("\nDone! All comparison plots saved to 'plots/' folder.")
