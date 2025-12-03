import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from deseq2_py.deseq_optimized import run_deseq
from deseq2_py.size_factors import estimate_size_factors
from deseq2_py.transformations import vst
from deseq2_py.plotting import plotMA, plotVolcano, plotPCA

os.makedirs("plots", exist_ok=True)


def load_data():
    counts_df = pd.read_csv("data/counts.csv", index_col=0)
    coldata_df = pd.read_csv("data/coldata.csv", index_col=0)
    return counts_df, coldata_df


def main():
    counts_df, coldata_df = load_data()
    condition_labels = coldata_df["dex"].values

    # ===========================
    # 1. RUN DESEQ2 PIPELINE
    # ===========================
    print(f"Running DESeq2 on {counts_df.shape[0]} genes...")
    start_time = time.time()

    res = run_deseq(counts_df.values, condition_labels)

    print(f"Done in {time.time() - start_time:.1f} seconds.")

    res_df = pd.DataFrame(res, index=counts_df.index)
    res_df.to_csv("data/results_python.csv")

    # ===========================
    # 2. LOAD R RESULTS
    # ===========================
    print("\nLoading R DESeq2 results for comparison...")
    r = pd.read_csv("data/deseq2_results.csv", index_col=0)

    py_df = res_df[["baseMean", "log2FoldChange", "pvalue", "padj"]].copy()
    py_df.columns = ["baseMean_py", "log2FoldChange_py", "pvalue_py", "padj_py"]

    r_df = r[["baseMean", "log2FoldChange", "pvalue", "padj"]].copy()
    r_df.columns = ["baseMean_r", "log2FoldChange_r", "pvalue_r", "padj_r"]

    merged = py_df.join(r_df, how="inner")
    print(f"Merged {len(merged)} genes for comparison.")

    # Filter extremes
    plot_df = merged.copy()
    plot_df = plot_df[np.abs(plot_df["log2FoldChange_py"]) < 12]
    plot_df = plot_df[np.abs(plot_df["log2FoldChange_r"]) < 12]

    # ===========================
    # 3. Single MA plot
    # ===========================
    print("\nGenerating plots...")

    plotMA(res, alpha=0.05)
    plt.savefig("plots/ma_plot.png", dpi=300)
    plt.close()
    print("  Saved: plots/ma_plot.png")

    # ===========================
    # 4. MA comparison
    # ===========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)

    configs = [
        (axes[0], "baseMean_py", "log2FoldChange_py", "padj_py", "Python Port"),
        (axes[1], "baseMean_r", "log2FoldChange_r", "padj_r", "Original DESeq2 (R)")
    ]

    for ax, x_col, y_col, padj_col, title in configs:
        mu = plot_df[x_col].values
        lfc = plot_df[y_col].values
        padj = plot_df[padj_col].values

        sig = padj < 0.05
        valid = (mu > 0) & np.isfinite(lfc)

        mask_ns = valid & ~sig
        ax.scatter(mu[mask_ns], lfc[mask_ns], s=5, alpha=0.5, c="gray", rasterized=True)

        mask_sig = valid & sig
        ax.scatter(mu[mask_sig], lfc[mask_sig], s=5, alpha=0.5, c="red", rasterized=True)

        ax.set_xscale("log")
        ax.axhline(0, color="blue", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Mean of Normalized Counts")
        ax.set_ylabel("Log2 Fold Change")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("MA Plot Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/compare_ma_plots.png", dpi=300)
    plt.close()
    print("  Saved: plots/compare_ma_plots.png")

    # ===========================
    # 5. Single volcano
    # ===========================
    plotVolcano(res, alpha=0.05, lfc_threshold=1.0)
    plt.ylim(0, 50)
    plt.savefig("plots/volcano_plot.png", dpi=300)
    plt.close()
    print("  Saved: plots/volcano_plot.png")

    # ===========================
    # 6. Volcano comparison
    # ===========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    configs = [
        (axes[0], "log2FoldChange_py", "pvalue_py", "padj_py", "Python Port"),
        (axes[1], "log2FoldChange_r", "pvalue_r", "padj_r", "Original DESeq2 (R)")
    ]

    for ax, x_col, p_col, padj_col, title in configs:
        lfc = plot_df[x_col].values
        pval = plot_df[p_col].values
        padj = plot_df[padj_col].values

        neglogp = -np.log10(pval + 1e-300)
        neglogp = np.clip(neglogp, 0, 50)

        sig = padj < 0.05

        ax.scatter(lfc[~sig], neglogp[~sig], s=5, color="lightgray", alpha=0.5, rasterized=True)
        ax.scatter(lfc[sig], neglogp[sig], s=8, color="darkred", alpha=0.6, rasterized=True)

        ax.axvline(-1, color="k", linestyle="--", lw=0.5)
        ax.axvline(1, color="k", linestyle="--", lw=0.5)
        ax.axhline(-np.log10(0.05), color="k", linestyle="--", lw=0.5)

        ax.set_xlabel("log2 Fold Change")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(title)

    plt.suptitle("Volcano Plot Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/compare_volcano_plots.png", dpi=300)
    plt.close()
    print("  Saved: plots/compare_volcano_plots.png")

    # ===========================
    # 7. PCA plot
    # ===========================
    size_factors = estimate_size_factors(counts_df.values)
    vst_data = vst(counts_df.values, size_factors=size_factors, dispersions=res["dispersion"])

    plotPCA(vst_data, sample_info=coldata_df, color_by="dex")
    plt.savefig("plots/pca_plot.png", dpi=300)
    plt.close()
    print("  Saved: plots/pca_plot.png")

    # ===========================
    # 8. LFC correlation
    # ===========================
    df_clean = merged[merged["baseMean_py"] > 10].dropna(subset=["log2FoldChange_py", "log2FoldChange_r"])

    x = df_clean["log2FoldChange_r"]
    y = df_clean["log2FoldChange_py"]
    r_val = x.corr(y)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=5, alpha=0.3, color="purple", rasterized=True)

    mn, mx = -10, 10
    ax.plot([mn, mx], [mn, mx], "k-", alpha=0.75)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

    ax.set_xlabel("R DESeq2 log2FoldChange")
    ax.set_ylabel("Python Port log2FoldChange")
    ax.set_title(f"Log2 Fold Change Correlation\n(baseMean > 10, Pearson r = {r_val:.4f})")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig("plots/compare_lfc_correlation.png", dpi=300)
    plt.close()
    print("  Saved: plots/compare_lfc_correlation.png")

    # ===========================
    # 9. TOP GENES OVERLAP
    # ===========================
    Ns = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    overlaps = []

    df_pval = merged.dropna(subset=["pvalue_py", "pvalue_r"])

    for n in Ns:
        top_py = set(df_pval.nsmallest(n, "pvalue_py").index)
        top_r = set(df_pval.nsmallest(n, "pvalue_r").index)
        overlap_pct = len(top_py & top_r) / n * 100
        overlaps.append(overlap_pct)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(Ns, overlaps, marker="o", linestyle="-", color="green", linewidth=2, markersize=8)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Top N Genes (by p-value)")
    ax.set_ylabel("% Overlap with R DESeq2")
    ax.set_title("Agreement of Top Differentially Expressed Genes")
    ax.set_ylim(0, 105)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    for n, o in zip(Ns, overlaps):
        ax.annotate(f"{o:.0f}%", (n, o), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/compare_overlap_curve.png", dpi=300)
    plt.close()
    print("  Saved: plots/compare_overlap_curve.png")

    # ===========================
    # 10. SUMMARY
    # ===========================
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)

    sig_py = (merged["padj_py"] < 0.05).sum()
    sig_r = (merged["padj_r"] < 0.05).sum()
    both_sig = ((merged["padj_py"] < 0.05) & (merged["padj_r"] < 0.05)).sum()

    print(f"  Total genes compared: {len(merged)}")
    print(f"  Significant (Python): {sig_py}")
    print(f"  Significant (R): {sig_r}")
    print(f"  Significant in both: {both_sig}")
    print(f"  LFC correlation (baseMean>10): {r_val:.4f}")
    print(f"  Top 100 overlap: {overlaps[Ns.index(100)]:.1f}%")
    print(f"  Top 1000 overlap: {overlaps[Ns.index(1000)]:.1f}%")
    print("=" * 50)

    print("\nDone! All plots saved to 'plots/' folder.")


if __name__ == "__main__":
    main()
