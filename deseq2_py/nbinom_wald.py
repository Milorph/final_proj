import numpy as np
from scipy.stats import norm
import statsmodels.api as sm


def benjamini_hochberg(pvals):
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvals : array-like

    Returns
    -------
    padj : np.ndarray
    """
    pvals = np.asarray(pvals, dtype=float)
    m = pvals.size
    order = np.argsort(pvals)
    ranked_p = pvals[order]

    # compute adjusted p-values
    adj = ranked_p * m / (np.arange(1, m + 1))
    # enforce monotone non-decreasing when going backwards
    adj_rev = np.minimum.accumulate(adj[::-1])[::-1]

    padj = np.empty_like(adj_rev)
    padj[order] = np.clip(adj_rev, 0, 1)
    return padj


def nb_glm_wald(counts, size_factors, dispersions, design_matrix,
                coef_index=1, min_mu=1e-8):
    """
    Negative Binomial GLM with log link and Wald test for a single coefficient
    (by default the 'condition' coefficient, column 1 of design).

    Uses statsmodels.GLM with family=NegativeBinomial(alpha), which is closer
    in spirit to DESeq2’s NB-GLM than a hand-rolled IRLS.

    Assumes design_matrix encodes a model like: ~ condition
    where the columns are [intercept, condition, ...].

    Parameters
    ----------
    counts : (G, S) array
        Raw counts (genes x samples).
    size_factors : (S,) array
        Per-sample size factors.
    dispersions : (G,) array
        Final per-gene dispersion (alpha_g).
    design_matrix : (S, P) array
        Design matrix (samples x parameters).
    coef_index : int
        Which coefficient index to test (default 1 = condition).

    Returns
    -------
    log2_fc : (G,) array
        Log2 fold change for the chosen coefficient.
    se_log2 : (G,) array
        Standard error of log2 fold change.
    wald_stat : (G,) array
        Wald statistic for the chosen coefficient.
    pval : (G,) array
        Two-sided p-value.
    padj : (G,) array
        Benjamini-Hochberg adjusted p-values.
    """
    Y = np.asarray(counts, dtype=float)
    sf = np.asarray(size_factors, dtype=float)
    disp = np.asarray(dispersions, dtype=float)
    X = np.asarray(design_matrix, dtype=float)

    G, S = Y.shape
    S2, P = X.shape
    if S2 != S:
        raise ValueError("design_matrix must have same number of rows as samples")
    if sf.ndim != 1 or sf.shape[0] != S:
        raise ValueError("size_factors length must equal number of samples")
    if disp.shape[0] != G:
        raise ValueError("dispersions length must equal number of genes")
    if coef_index < 0 or coef_index >= P:
        raise ValueError("coef_index out of bounds")

    # GLM offset = log(size_factors)
    offset = np.log(sf + 1e-12)

    # Output arrays
    log2_fc = np.full(G, np.nan, dtype=float)
    se_log2 = np.full(G, np.nan, dtype=float)
    wald = np.full(G, np.nan, dtype=float)
    pvals = np.ones(G, dtype=float)  # default p=1 if we fail

    for g in range(G):
        y = Y[g, :]

        # skip genes with all zeros
        if y.sum() == 0:
            continue

        # if no variability, GLM not informative
        if np.all(y == y[0]):
            continue

        alpha = float(disp[g])
        if not np.isfinite(alpha) or alpha <= 0:
            alpha = 1e-8

        fam = sm.families.NegativeBinomial(alpha=alpha)

        try:
            model = sm.GLM(y, X, family=fam, offset=offset)
            res = model.fit()
        except Exception:
            # leave this gene as NaN / p=1
            continue

        beta = res.params
        se = res.bse

        if coef_index >= len(beta) or coef_index >= len(se):
            continue

        b = beta[coef_index]
        s = se[coef_index]
        if not np.isfinite(b) or not np.isfinite(s) or s <= 0:
            continue

        # GLM scale → log2
        log2_beta = b / np.log(2.0)
        log2_se = s / np.log(2.0)

        z = b / s
        p = 2.0 * (1.0 - norm.cdf(abs(z)))

        log2_fc[g] = log2_beta
        se_log2[g] = log2_se
        wald[g] = z
        pvals[g] = p

    # Clean p-values and BH correction
    pvals_clean = np.where(np.isfinite(pvals), pvals, 1.0)
    padj = benjamini_hochberg(pvals_clean)

    return log2_fc, se_log2, wald, pvals_clean, padj
