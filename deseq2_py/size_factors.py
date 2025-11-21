import numpy as np

def geometric_mean_poscounts(x):
    """Geometric mean ignoring zeros (DESeq2's poscounts version)."""
    nz = x[x > 0]
    if len(nz) == 0:
        return 0.0
    return np.exp(np.mean(np.log(nz)))


def estimate_size_factors_for_matrix(
    counts,
    loc_func=np.median,
    geo_means=None,
    control_genes=None,
    type="ratio"
):
    """
    Port of DESeq2::estimateSizeFactorsForMatrix

    Parameters
    ----------
    counts : np.ndarray
        2D (genes x samples) raw counts.
    loc_func : function
        Location function, default median.
    geo_means : np.ndarray or None
        Precomputed geometric means.
    control_genes : list or ndarray or None
        Optional subset to compute size factors.
    type : {"ratio", "poscounts"}

    Returns
    -------
    np.ndarray of size factors (length = num samples)
    """

    counts = np.asarray(counts, dtype=float)
    G, S = counts.shape

    # ---- Determine geometric means per gene ----
    if geo_means is None:
        incoming_geo_means = False

        if type == "ratio":
            # geometric mean per gene using all values
            with np.errstate(divide="ignore", invalid="ignore"):
                log_geomeans = np.mean(np.log(counts), axis=1)

        elif type == "poscounts":
            # geometric mean using only positive counts
            lc = np.log(counts, where=(counts > 0), out=np.zeros_like(counts))
            log_geomeans = np.mean(lc, axis=1)
            all_zero = np.sum(counts, axis=1) == 0
            log_geomeans[all_zero] = -np.inf

    else:
        incoming_geo_means = True
        if len(geo_means) != counts.shape[0]:
            raise ValueError("geoMeans should be as long as number of genes")

        log_geomeans = np.log(geo_means)

    if np.all(np.isinf(log_geomeans)):
        raise ValueError("every gene contains at least one zero; cannot compute size factors")

    # ---- Apply control genes (optional) ----
    if control_genes is not None:
        log_geomeans_sub = log_geomeans[control_genes]
        counts_sub = counts[control_genes, :]
    else:
        log_geomeans_sub = log_geomeans
        counts_sub = counts

    # ---- Compute per-sample size factors ----
    size_factors = np.zeros(S)

    for j in range(S):
        c = counts_sub[:, j]
        mask = np.isfinite(log_geomeans_sub) & (c > 0)

        # per-sample median of log ratios
        vals = (np.log(c[mask]) - log_geomeans_sub[mask])
        size_factors[j] = np.exp(loc_func(vals))

    # ---- If external geoMeans supplied, normalize to geometric mean 1 ----
    if incoming_geo_means:
        size_factors = size_factors / np.exp(np.mean(np.log(size_factors)))

    return size_factors


def estimate_size_factors(
    counts,
    type="ratio",
    loc_func=np.median,
    geo_means=None,
    control_genes=None
):
    """
    Simplified Python version of estimateSizeFactors.DESeqDataSet
    (only covers the matrix path)
    """
    return estimate_size_factors_for_matrix(
        counts,
        loc_func=loc_func,
        geo_means=geo_means,
        control_genes=control_genes,
        type=type
    )
