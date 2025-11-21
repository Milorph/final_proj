import pandas as pd
import numpy as np

# ---- Load data ----
py = pd.read_csv("data/results_python.csv", index_col=0)
r = pd.read_csv("data/deseq2_results.csv", index_col=0)

# Keep and rename DESeq2 columns
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

# Merge on gene ID
merged = py.join(r, how="inner").dropna(subset=["log2FoldChange_py", "log2FoldChange_r"])


def summarize(df: pd.DataFrame, label: str):
    print(f"\n=== {label} ===")
    if df.empty:
        print("No genes in this subset.")
        return

    n_genes = df.shape[0]
    print("Number of genes:", n_genes)

    # 1) Correlation of log2FC
    corr = df["log2FoldChange_py"].corr(df["log2FoldChange_r"])
    print("Log2FC correlation:", corr)

    # 2) Top-N overlap for multiple N
    for N in [20, 50, 100]:
        top_py = set(df.sort_values("pvalue_py").head(N).index)
        top_r = set(df.sort_values("pvalue_r").head(N).index)
        overlap = len(top_py & top_r)
        print(f"Top {N} overlap: {overlap} / {N}")

    # 3) Sign agreement (direction of change)
    sign_py = np.sign(df["log2FoldChange_py"])
    sign_r = np.sign(df["log2FoldChange_r"])
    nonzero = (sign_py != 0) & (sign_r != 0)
    if nonzero.sum() > 0:
        agree = (sign_py[nonzero] == sign_r[nonzero]).sum()
        frac_agree = agree / nonzero.sum()
        print(f"Sign agreement (non-zero LFC): {frac_agree*100:.1f}%")
    else:
        print("Sign agreement: N/A (no non-zero LFC genes).")

    # 4) Significant genes counts
    sig_py = (df["padj_py"] < 0.05).sum()
    sig_r = (df["padj_r"] < 0.05).sum()
    print(f"Significant genes (padj < 0.05): Python={sig_py}, DESeq2={sig_r}")


# 1) All genes
summarize(merged, "All genes")

# 2) High-expression genes only (DESeq2 baseMean > 50)
high = merged[merged["baseMean_r"] > 50]
summarize(high, "High-expression genes (baseMean_r > 50)")
