import numpy as np
from deseq2_py.size_factors import estimate_size_factors
from deseq2_py.dispersion import estimate_dispersions
from deseq2_py.nbinom_wald import nb_glm_wald


def run_deseq(counts, condition_labels):
    """
    Minimal DESeq2-like pipeline for a 2-condition design.
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

    # 2) dispersions (With Group Pooling!)
    # We pass condition_labels here so it knows how to group samples for variance calc
    base_means, disp_final, disp_gw, disp_trend, disp_map, is_outlier = estimate_dispersions(
        counts, size_factors, group_labels=condition_labels
    )

    # 3) Wald test
    log2_fc_mle, se_log2, wald, pvals, padj = nb_glm_wald(
        counts, size_factors, disp_final, X
    )

    # 4) LFC Shrinkage (Approximate Empirical Bayes)
    mask_stable = (base_means > np.percentile(base_means, 75)) & np.isfinite(log2_fc_mle)
    if mask_stable.sum() > 10:
        prior_std = np.std(log2_fc_mle[mask_stable])
        if prior_std < 0.1: prior_std = 0.1 
    else:
        prior_std = 1.0 

    prior_var = prior_std ** 2
    
    with np.errstate(divide='ignore', invalid='ignore'):
        shrinkage_factor = 1.0 / (1.0 + (se_log2**2 / prior_var))
    
    shrinkage_factor[~np.isfinite(shrinkage_factor)] = 0.0
    log2_fc_map = log2_fc_mle * shrinkage_factor

    # 5) Outlier Handling
    if is_outlier is not None:
        pvals[is_outlier] = np.nan
        padj[is_outlier] = np.nan

    result = {
        "baseMean": base_means,
        "log2FoldChange": log2_fc_map,
        "lfcMLE": log2_fc_mle,
        "lfcSE": se_log2,
        "stat": wald,
        "pvalue": pvals,
        "padj": padj,
        "dispersion": disp_final,
    }
    return result