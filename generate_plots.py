import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# ===========================
# 1. LOAD AND MERGE DATA
# ===========================
print("Loading results...")
try:
    py = pd.read_csv("data/results_python.csv", index_col=0)
    r = pd.read_csv("data/deseq2_results.csv", index_col=0)
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Make sure you have run 'run_airway.py' and the R script first.")
    exit(1)

# Select columns from R
r = r[["baseMean", "log2FoldChange", "pvalue", "padj"]]

# Rename DESeq2 (R) columns
r = r.rename(columns={
    "baseMean": "baseMean_r",
    "log2FoldChange": "log2FoldChange_r",
    "pvalue": "pvalue_r",
    "padj": "padj_r",
})

# Rename Python columns
py = py.rename(columns={
    "baseMean": "baseMean_py",
    "log2FoldChange": "log2FoldChange_py",
    "pvalue": "pvalue_py",
    "padj": "padj_py",
})

# Join
merged = py.join(r, how="inner")
print(f"Merged {len(merged)} genes for comparison.")

# Filter for cleaner plotting (remove extreme visual outliers)
plot_df = merged.copy()
plot_df = plot_df[np.abs(plot_df["log2FoldChange_py"]) < 12]
plot_df = plot_df[np.abs(plot_df["log2FoldChange_r"]) < 12]


# ===========================
# 2. PLOTTING FUNCTIONS
# ===========================

def plot_ma_side_by_side(df, filename):
    """
    Generates two MA plots side-by-side with IDENTICAL scaling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
    
    # Determine common Y-limits so plots look identical
    y_min = min(df["log2FoldChange_py"].min(), df["log2FoldChange_r"].min())
    y_max = max(df["log2FoldChange_py"].max(), df["log2FoldChange_r"].max())
    
    configs = [
        (axes[0], "baseMean_py", "log2FoldChange_py", "padj_py", "Python Port (Shrinkage Applied)"),
        (axes[1], "baseMean_r", "log2FoldChange_r", "padj_r", "Original DESeq2 (R)")
    ]

    for ax, x_col, y_col, padj_col, title in configs:
        mu = df[x_col]
        lfc = df[y_col]
        padj = df[padj_col]
        
        sig = padj < 0.05
        non_sig = ~sig
        
        # Downsample non-significant points
        if non_sig.sum() > 0:
            ns_idx = df[non_sig].sample(frac=0.15, random_state=42).index
            ax.scatter(np.log10(mu.loc[ns_idx] + 1e-8), lfc.loc[ns_idx], 
                       s=3, alpha=0.15, color="gray", label="NS")
        
        # Plot significant (red)
        if sig.sum() > 0:
            ax.scatter(np.log10(mu[sig] + 1e-8), lfc[sig],
                       s=6, alpha=0.6, color="red", label="padj < 0.05")
        
        ax.axhline(0, color="black", linewidth=1, linestyle="-")
        ax.set_xlabel("log10(baseMean)")
        
        if ax == axes[0]:
            ax.set_ylabel("log2 Fold Change")
        
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(y_min, y_max) # Enforce same scale

    plt.suptitle("Comparison of MA Plots (Visualizing Shrinkage)", fontsize=14)
    plt.tight_layout()
    out = os.path.join("plots", filename)
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def plot_volcano_side_by_side(df, filename):
    """
    Generates two Volcano plots side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    configs = [
        (axes[0], "log2FoldChange_py", "pvalue_py", "padj_py", "Python Port"),
        (axes[1], "log2FoldChange_r", "pvalue_r", "padj_r", "Original DESeq2 (R)")
    ]

    for ax, x_col, p_col, padj_col, title in configs:
        lfc = df[x_col]
        pval = df[p_col]
        padj = df[padj_col]
        
        neglogp = -np.log10(pval + 1e-300)
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

    plt.tight_layout()
    out = os.path.join("plots", filename)
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def plot_log2fc_correlation(df, filename):
    """
    Direct scatter plot correlation of Log2FC values.
    FILTERS OUT NOISY LOW-COUNT GENES FOR CLEANER PLOT.
    """
    # Filter: Only keep genes with meaningful counts (matches the table statistic)
    # This removes the messy horizontal line at 0 caused by shrinkage differences on noise
    df_clean = df[df["baseMean_py"] > 10].dropna(subset=["log2FoldChange_py", "log2FoldChange_r"])
    
    x = df_clean["log2FoldChange_r"]
    y = df_clean["log2FoldChange_py"]
    
    r_val = x.corr(y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=5, alpha=0.3, color="purple")
    
    # Diagonal line
    mn, mx = -10, 10 
    plt.plot([mn, mx], [mn, mx], 'k-', alpha=0.75, zorder=0, label="y=x")
    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    
    plt.xlabel("R log2FoldChange")
    plt.ylabel("Python log2FoldChange")
    plt.title(f"LFC Correlation (BaseMean > 10)\nPearson r = {r_val:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out = os.path.join("plots", filename)
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def plot_overlap_curve(df, filename):
    """
    Plots the % overlap of the Top N genes.
    """
    Ns = [10, 50, 100, 200, 500, 1000, 2000]
    overlaps = []
    
    df_clean = df.dropna(subset=["pvalue_py", "pvalue_r"])
    
    for n in Ns:
        top_py = set(df_clean.sort_values("pvalue_py").head(n).index)
        top_r = set(df_clean.sort_values("pvalue_r").head(n).index)
        
        if n > 0:
            overlaps.append(len(top_py.intersection(top_r)) / n * 100)
        else:
            overlaps.append(0)
        
    plt.figure(figsize=(6, 4))
    plt.plot(Ns, overlaps, marker='o', linestyle='-', color='green')
    plt.ylim(0, 105)
    plt.xlabel("Top N Genes (sorted by p-value)")
    plt.ylabel("% Overlap")
    plt.title("Agreement of Top Gene Lists")
    plt.grid(True, alpha=0.3)
    
    out = os.path.join("plots", filename)
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# ===========================
# 3. RUN ALL PLOTS
# ===========================
print("Generating comparison plots...")

plot_ma_side_by_side(plot_df, "compare_ma_plots.png")
plot_volcano_side_by_side(plot_df, "compare_volcano_plots.png")
plot_log2fc_correlation(merged, "compare_lfc_correlation.png")
plot_overlap_curve(merged, "compare_overlap_curve.png")

print("Done! Check the 'plots/' directory.")