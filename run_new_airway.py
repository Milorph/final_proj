import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from deseq2_py.deseq_optimized import run_deseq
from deseq2_py.size_factors import estimate_size_factors
from deseq2_py.transformations import vst
from deseq2_py.plotting import plotMA, plotVolcano, plotPCA


def load_data():
    counts_df = pd.read_csv("data/counts.csv", index_col=0)
    coldata_df = pd.read_csv("data/coldata.csv", index_col=0)
    return counts_df, coldata_df


def main():
    counts_df, coldata_df = load_data()
    condition_labels = coldata_df["dex"].values

    print(f"Running DESeq2 on {counts_df.shape[0]} genes...")
    start_time = time.time()

    res = run_deseq(counts_df.values, condition_labels)

    print(f"Done in {time.time() - start_time:.1f} seconds.")

    # Save results
    res_df = pd.DataFrame(res, index=counts_df.index)
    res_df.to_csv("data/results_python.csv")

    # MA Plot
    plotMA(res, alpha=0.05)
    plt.savefig("plots/ma_plot.png")
    plt.close()

    # Volcano Plot
    plotVolcano(res, alpha=0.05, lfc_threshold=1.0)
    plt.savefig("plots/volcano_plot.png")
    plt.close()

    # PCA Plot (VST only)
    vst_data = vst(counts_df.values)
    plotPCA(vst_data, sample_info=coldata_df, color_by="dex")
    plt.savefig("plots/pca_plot.png")
    plt.close()

    print("Done! Check plots/ folder for visualizations.")


if __name__ == "__main__":
    main()
