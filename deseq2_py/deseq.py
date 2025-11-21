import numpy as np
from deseq2_py.size_factors import estimate_size_factors
from deseq2_py.dispersion import estimate_dispersions
from deseq2_py.nbinom_wald import nb_glm_wald


def run_deseq(counts, condition_labels):
    """
    Minimal DESeq2-like pipeline for a 2-condition design:

        design = ~ condition

    Parameters
    ----------
    counts : (G, S) array-like
        Raw counts (genes x samples)
    condition_labels : list/array of length S
        Each element is group label, e.g. "A" / "B" or 0 / 1

    Returns
    -------
    result : dict of np.ndarray
        {
          "baseMean": ...,
          "log2FoldChange": ...,
          "lfcSE": ...,
          "stat": ...,
          "pvalue": ...,
          "padj": ...,
          "dispersion": ...
        }
    """
    counts = np.asarray(counts, dtype=float)
    G, S = counts.shape

    # convert condition_labels to 0/1
    cond = np.asarray(condition_labels)
    if cond.shape[0] != S:
        raise ValueError("condition_labels length must equal number of samples")

    # map labels to 0/1
    unique = np.unique(cond)
    if unique.size != 2:
        raise ValueError("This minimal implementation supports exactly 2 conditions")

    cond_numeric = np.zeros_like(cond, dtype=float)
    cond_numeric[cond == unique[1]] = 1.0  # baseline = unique[0], treatment = unique[1]

    # design matrix: intercept + condition
    X = np.column_stack([
        np.ones(S, dtype=float),
        cond_numeric
    ])

    # 1) size factors
    size_factors = estimate_size_factors(counts)

    # 2) dispersions
    base_means, disp_final, disp_gw, disp_trend, disp_map, is_outlier = estimate_dispersions(
        counts, size_factors
    )

    # 3) Wald test
    log2_fc, se_log2, wald, pvals, padj = nb_glm_wald(
        counts, size_factors, disp_final, X
    )

    result = {
        "baseMean": base_means,
        "log2FoldChange": log2_fc,
        "lfcSE": se_log2,
        "stat": wald,
        "pvalue": pvals,
        "padj": padj,
        "dispersion": disp_final,
    }
    return result
