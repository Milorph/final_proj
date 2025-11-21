import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

# ---- Load and merge data ----
py = pd.read_csv("data/results_python.csv", index_col=0)
r = pd.read_csv("data/deseq2_results.csv", index_col=0)

# Keep / rename DESeq2 (R) columns
r = r[["baseMean", "log2FoldChange", "pvalue", "padj"]].rename(columns={
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

# Inner join on gene ID
merged = py.join(r, how="inner")

# For plotting: filter out ultra-low expression and insane LFC
plot_df = merged.copy()
plot_df = plot_df[np.abs(plot_df["log2FoldChange_py"]) < 10]

# =====================
#  PLOTTING FUNCTIONS
# =====================

def ma_plot(df, filename):
    """MA plot with downsampling and clean style."""
    mu = df["baseMean_py"]
    lfc = df["log2FoldChange_py"]
    padj = df["padj_py"]

    # significance mask
    sig = padj < 0.05
    non_sig = ~sig

    # downsample to avoid smear
    ns_idx = df[non_sig].sample(frac=0.15, random_state=42).index

    plt.figure(figsize=(6.5, 5.5))
    # non-sig points (small, light)
    plt.scatter(np.log10(mu.loc[ns_idx] + 1e-8),
                lfc.loc[ns_idx],
                s=3, alpha=0.18, color="gray")

    # significant up/down
    plt.scatter(np.log10(mu[sig & (lfc >= 0)] + 1e-8),
                lfc[sig & (lfc >= 0)],
                s=6, alpha=0.9, color="red", label="Up")
    plt.scatter(np.log10(mu[sig & (lfc < 0)] + 1e-8),
                lfc[sig & (lfc < 0)],
                s=6, alpha=0.9, color="blue", label="Down")

    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("log10(baseMean)")
    plt.ylabel("log2 fold change")
    plt.title("MA plot (Python DESeq-like)")
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    out = os.path.join("plots", filename)
    plt.savefig(out, dpi=400)
    plt.close()
    print("Saved:", out)



def volcano_plot(df, filename):
    """Volcano plot for Python results only."""
    lfc = df["log2FoldChange_py"]
    pval = df["pvalue_py"]
    padj = df["padj_py"]

    neglogp = -np.log10(pval + 1e-300)
    sig = padj < 0.05

    plt.figure(figsize=(7, 6))
    plt.scatter(lfc[~sig], neglogp[~sig], s=10, color="lightgray", alpha=0.5, label="NS")
    plt.scatter(lfc[sig], neglogp[sig], s=12, color="darkred", alpha=0.8, label="padj<0.05")

    # Optional reference lines
    plt.axvline(-1, color="black", linestyle="--", linewidth=1)
    plt.axvline(1, color="black", linestyle="--", linewidth=1)
    plt.axhline(-np.log10(0.05), color="black", linestyle="--", linewidth=1)

    plt.xlabel("log2 fold change")
    plt.ylabel("-log10(p-value)")
    plt.title("Volcano plot (Python DESeq-like)")
    plt.legend(fontsize=8)
    plt.tight_layout()

    out = os.path.join("plots", filename)
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def dispersion_plot(df, filename):
    """Simple dispersion vs mean plot for Python results."""
    if "dispersion" not in df.columns:
        print("No 'dispersion' column in results_python.csv — skipping dispersion plot.")
        return

    mu = df["baseMean_py"]
    disp = df["dispersion"]

    x = np.log10(mu + 1e-8)
    y = np.log10(disp + 1e-8)

    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=10, color="steelblue", alpha=0.5)
    plt.xlabel("log10(baseMean)")
    plt.ylabel("log10(dispersion)")
    plt.title("Dispersion vs mean (Python)")
    plt.tight_layout()

    out = os.path.join("plots", filename)
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def log2fc_corr(df, filename, label="All genes"):
    """Python vs DESeq2 log2FC correlation scatter."""
    df = df.dropna(subset=["log2FoldChange_py", "log2FoldChange_r"])
    if df.empty:
        print("No overlapping genes for correlation:", filename)
        return

    x = df["log2FoldChange_r"]
    y = df["log2FoldChange_py"]
    r = x.corr(y)

    lim = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = max(lim, 0.5)  # avoid zero range

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=8, alpha=0.4, color="navy")
    plt.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="y = x")

    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.xlabel("log2FC (DESeq2 R)")
    plt.ylabel("log2FC (Python)")
    plt.title(f"log2FC comparison ({label})\nr = {r:.3f}")
    plt.legend()
    plt.tight_layout()

    out = os.path.join("plots", filename)
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


def overlap_curve(df, filename):
    """Top-N overlap of DE genes between Python and DESeq2."""
    df = df.dropna(subset=["pvalue_py", "pvalue_r"])
    Ns = [10, 20, 50, 100, 200, 500]
    overlaps = []

    for N in Ns:
        top_py = set(df.sort_values("pvalue_py").head(N).index)
        top_r = set(df.sort_values("pvalue_r").head(N).index)
        overlaps.append(len(top_py & top_r))

    plt.figure(figsize=(6, 4))
    plt.plot(Ns, overlaps, "-o", color="purple")
    plt.xlabel("Top N genes")
    plt.ylabel("Overlap count")
    plt.title("Top-N overlap: Python vs DESeq2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out = os.path.join("plots", filename)
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved:", out)


# -------- RUN ALL PLOTS --------

# 1) Python-only diagnostics
dispersion_plot(plot_df, "dispersion_vs_mean.png")
ma_plot(plot_df, "ma_plot_python.png")
volcano_plot(plot_df, "volcano_python.png")

# 2) Python vs DESeq2 comparisons
log2fc_corr(merged, "log2fc_corr_all.png", label="all genes")

# High-expression subset (like before, baseMean_r > 50)
high = merged[merged["baseMean_r"] > 50]
log2fc_corr(high, "log2fc_corr_high_expr.png", label="baseMean_r > 50")

overlap_curve(merged, "overlap_curve.png")

print("\nAll plots saved in ./plots")
