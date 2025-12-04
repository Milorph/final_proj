import numpy as np
from deseq2_py.size_factors import estimate_size_factors
# CRITICAL CHANGE: Import from your NEW optimized dispersion file
from deseq2_py.dispersion_optimized import estimate_dispersions 
from deseq2_py.nbinom_wald import nb_glm_wald


def run_deseq(counts, condition_labels):
    """
    Optimized DESeq2 pipeline using CR-APL (Slow but Accurate).
    """
    counts = np.asarray(counts, dtype=float)
    G, S = counts.shape

    cond = np.asarray(condition_labels)
    if cond.shape[0] != S:
        raise ValueError("condition_labels length must equal number of samples")
    unique = np.unique(cond)
    cond_numeric = np.zeros_like(cond, dtype=float)
    cond_numeric[cond == unique[1]] = 1.0 

    # design matrix: intercept + condition
    X = np.column_stack([
        np.ones(S, dtype=float),
        cond_numeric
    ])
    
    num_params = X.shape[1]
    df_resid = S - num_params

    # 1) size factors
    size_factors = estimate_size_factors(counts)

    # 2) dispersions (CR-APL)
    # Uses the function from dispersion_optimized.py
    base_means, disp_final, disp_gw, disp_trend, disp_map, is_outlier = estimate_dispersions(
        counts, size_factors, 
        design_matrix=X,
        degrees_of_freedom=df_resid
    )

    # 3) Wald test
    log2_fc_mle, se_log2, wald, pvals, padj = nb_glm_wald(
        counts, size_factors, disp_final, X
    )

    # 4) LFC Shrinkage
    mask_stable = (base_means > np. percentile(base_means, 75)) & np.isfinite(log2_fc_mle)
    if mask_stable.sum() > 10:
        prior_std = np. std(log2_fc_mle[mask_stable])
        if prior_std < 0.1:
            prior_std = 0.1
    else:
        prior_std = 1.0

    prior_var = prior_std ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        # Correct shrinkage: shrink MORE when SE is large relative to prior
        shrinkage_factor = prior_var / (prior_var + se_log2**2)

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