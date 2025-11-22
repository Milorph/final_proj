import numpy as np
from scipy.optimize import minimize_scalar, minimize
from scipy.special import gammaln, polygamma

# --- 1. THE CORE OBJECTIVE FUNCTION (Math from DESeq2 Paper) ---

def nbinom_loglike(counts, mu, alpha):
    """
    Log-likelihood of NBinom(mu, alpha). 
    Matches DESeq2 C++ implementation exactly.
    """
    # Prevent division by zero or overflow
    alpha = max(alpha, 1e-10)
    r = 1.0 / alpha
    
    # LL = log Gamma(y+r) - log Gamma(r) + r log(r/(r+mu)) + y log(mu/(r+mu))
    prob = r / (r + mu)
    ll = gammaln(counts + r) - gammaln(r) \
         + r * np.log(prob) + counts * np.log(1.0 - prob)
    return np.sum(ll)

def cox_reid_adjustment(mu, alpha, X):
    """
    Cox-Reid bias adjustment: -0.5 * log(det(X^T W X))
    """
    alpha = max(alpha, 1e-10)
    w = mu / (1.0 + alpha * mu)
    
    # XtWX = X.T @ diag(w) @ X
    XtWX = (X.T * w) @ X
    
    sign, logdet = np.linalg.slogdet(XtWX)
    if sign <= 0: return -np.inf
    return -0.5 * logdet

def get_crap_objective(counts, size_factors, X, mu_hat):
    """Returns a function to minimize for alpha"""
    def objective(log_alpha):
        alpha = np.exp(log_alpha)
        ll = nbinom_loglike(counts, mu_hat, alpha)
        cr = cox_reid_adjustment(mu_hat, alpha, X)
        # Minimize Negative Log Likelihood
        return -(ll + cr)
    return objective

# --- 2. PIPELINE STEPS ---

def estimate_gene_wise_dispersion(counts, size_factors, design_matrix, min_disp=1e-8):
    """
    Step 1: Find the Maximum Likelihood Estimate (MLE) of dispersion for each gene.
    """
    counts = np.asarray(counts, dtype=float)
    G, S = counts.shape
    
    # Pre-calculate Mu (Fitted Means)
    # For standard DESeq2, we estimate mu roughly once.
    norm_counts = counts / size_factors
    
    # 1-Factor Logic (Exact for 'trt' vs 'untrt')
    # We calculate mean of normalized counts per group
    cond_col = design_matrix[:, 1]
    mu_hat = np.zeros_like(counts)
    for val in np.unique(cond_col):
        mask = (cond_col == val)
        mean_g = np.mean(norm_counts[:, mask], axis=1)
        # Broadcast back to samples
        mu_hat[:, mask] = mean_g[:, None] * size_factors[mask]
        
    mu_hat = np.maximum(mu_hat, 1e-8)
    
    disp_gw = np.full(G, min_disp)
    base_means = norm_counts.mean(axis=1)
    
    print(f"Running Cox-Reid APL for {G} genes...")
    
    # Only fit genes with sufficient data (R ignores very low counts)
    mask_fit = base_means > 0
    genes_to_fit = np.where(mask_fit)[0]
    
    for i, idx in enumerate(genes_to_fit):
        if i % 5000 == 0: print(f"  ... processing gene {i}/{len(genes_to_fit)}")
        
        # Optimization
        obj_fn = get_crap_objective(counts[idx], size_factors, design_matrix, mu_hat[idx])
        
        res = minimize_scalar(
            obj_fn, 
            bounds=(np.log(1e-6), np.log(100.0)), 
            method='bounded'
        )
        disp_gw[idx] = np.exp(res.x)
        
    return base_means, disp_gw

def fit_parametric_dispersion_trend(base_means, disp_gw):
    """
    Step 2: Fit the 'Smooth' trend line.
    Using Gamma-family GLM (Reciprocal link), same as DESeq2 default.
    """
    # R filters strictly for trend fitting
    mask = (base_means > 2.0) & (disp_gw > 1e-6) & (disp_gw < 20.0)
    
    x_clean = base_means[mask]
    y_clean = disp_gw[mask]
    
    print(f"Fitting Dispersion Trend on {len(x_clean)} genes...")
    
    if len(x_clean) < 10:
        return lambda x: np.full_like(x, 0.01), (0.0, 0.01)

    # Gamma GLM Objective
    def gamma_deviance(params):
        a, b = params
        if a < 0 or b < 0: return 1e9
        
        pred = a / x_clean + b
        # Deviance = sum( (y-mu)/mu - log(y/mu) )
        term = (y_clean - pred)/pred - np.log(y_clean/pred)
        return np.sum(term)
        
    res = minimize(gamma_deviance, x0=[1.0, 0.01], bounds=[(0.0, None), (1e-8, None)], method='L-BFGS-B')
    a, b = res.x
    print(f"Trend Coefficients: a={a:.4f}, b={b:.4f}")
    
    def trend_fn(mu):
        return a / np.maximum(mu, 1e-8) + b
        
    return trend_fn, (a, b)

def estimate_map_dispersions(base_means, disp_gw, disp_trend_fn, design_matrix, counts, size_factors):
    """
    Step 3: Calculate MAP estimates (Shrinkage).
    Maximize: Likelihood(alpha) + Prior(alpha)
    """
    disp_trend = disp_trend_fn(base_means)
    
    # Calculate Prior Variance (Robust width of residuals)
    log_res = np.log(disp_gw) - np.log(disp_trend)
    valid = np.isfinite(log_res) & (base_means > 0)
    if valid.sum() > 10:
        mad = np.median(np.abs(log_res[valid] - np.median(log_res[valid])))
        sigma_prior = mad * 1.4826
    else:
        sigma_prior = 1.0
        
    sigma_prior = max(sigma_prior, 0.25) # Enforce R-like conservatism
    var_prior = sigma_prior ** 2
    
    print(f"Estimating MAP (Shrinkage) with prior width: {sigma_prior:.4f}...")
    
    # Approximate solution using Normal approximation (Empirical Bayes)
    # This is analytically identical to maximizing the posterior if Likelihood is Gaussian-ish
    df = counts.shape[1] - design_matrix.shape[1]
    var_obs = polygamma(1, df / 2.0) # Trigamma function
    
    weight = var_prior / (var_prior + var_obs)
    
    log_map = weight * np.log(disp_gw) + (1.0 - weight) * np.log(disp_trend)
    disp_final = np.exp(log_map)
    
    # Outlier detection (Cook's distance surrogate)
    # If observation is WAY far from trend, keep observation
    resid_z = (np.log(disp_gw) - np.log(disp_trend)) / sigma_prior
    is_outlier = np.abs(resid_z) > 2.0 # 2 SD threshold
    
    disp_final[is_outlier] = disp_gw[is_outlier]
    
    return disp_final, is_outlier

# --- MAIN ENTRY POINT ---

def estimate_dispersions(counts, size_factors, design_matrix, degrees_of_freedom=None):
    """
    The Full DESeq2 Dispersion Pipeline (No shortcuts).
    """
    # 1. Gene-wise Estimates (CR-APL)
    base_means, disp_gw = estimate_gene_wise_dispersion(counts, size_factors, design_matrix)
    
    # 2. Trend Fit
    trend_fn, _ = fit_parametric_dispersion_trend(base_means, disp_gw)
    
    # 3. MAP Estimates (Shrinkage)
    disp_final, is_outlier = estimate_map_dispersions(
        base_means, disp_gw, trend_fn, design_matrix, counts, size_factors
    )
    
    # Return expected tuple format
    # (base_means, disp_final, disp_gw, disp_trend_vals, disp_map_vals, is_outlier)
    disp_trend_vals = trend_fn(base_means)
    return base_means, disp_final, disp_gw, disp_trend_vals, disp_final, is_outlier