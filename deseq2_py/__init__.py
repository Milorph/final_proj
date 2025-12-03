"""
DESeq2-like differential expression analysis for RNA-seq data in Python.

This package provides a Python implementation of the core DESeq2 methodology
for analyzing RNA-seq count data to identify differentially expressed genes.

Main Classes:
    DESeqDataSet : Container class for managing DESeq2 analysis

Main Functions:
    run_deseq : Run the full DESeq2 pipeline on count data
    vst : Variance stabilizing transformation
    rlog : Regularized log transformation
    plotMA : MA plot of results
    plotPCA : PCA plot of samples
    plotVolcano : Volcano plot of results

References:
    Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
    and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
"""

# Core pipeline
from .deseq_optimized import run_deseq
from .deseq_dataset import DESeqDataSet

# Size factors
from .size_factors import estimate_size_factors

# Dispersions
from .dispersion_optimized import estimate_dispersions
from .dispersion_local import (
    fit_local_dispersion_trend,
    fit_mean_dispersion,
    fit_dispersion_trend
)

# Statistical tests
from .nbinom_wald import nb_glm_wald, benjamini_hochberg
from .lrt import likelihood_ratio_test, lrt_test

# LFC shrinkage
from .lfc_shrinkage import lfcShrink, apeglm_shrinkage, normal_shrinkage

# Transformations
from .transformations import vst, rlog, normTransform

# Results and contrasts
from .results import results, lfc_threshold_pvalue, summary

# Outlier detection
from .outliers import (
    calculate_cooks_distance,
    detect_outliers,
    replace_outliers
)

# Independent filtering
from .independent_filtering import (
    independent_filtering,
    find_optimal_threshold,
    filter_by_expression
)

# Design matrices
from .design import (
    create_design_matrix,
    model_matrix,
    parse_formula,
    get_contrast_vector
)

# Plotting
from .plotting import (
    plotMA,
    plotVolcano,
    plotDispEsts,
    plotPCA,
    plotCounts,
    plotHeatmap
)

# Utilities
from .utils import fpm, fpkm, tpm, normalize_counts, filter_low_counts

# PyDESeq2 compatibility
from .pydeseq2_compat import (
    is_pydeseq2_available,
    PyDESeq2Wrapper,
    run_deseq2_analysis
)

__version__ = "0.1.0"

__all__ = [
    # Core
    'DESeqDataSet',
    'run_deseq',
    
    # Size factors
    'estimate_size_factors',
    
    # Dispersions
    'estimate_dispersions',
    'fit_local_dispersion_trend',
    'fit_mean_dispersion',
    'fit_dispersion_trend',
    
    # Tests
    'nb_glm_wald',
    'benjamini_hochberg',
    'likelihood_ratio_test',
    'lrt_test',
    
    # LFC shrinkage
    'lfcShrink',
    'apeglm_shrinkage',
    'normal_shrinkage',
    
    # Transformations
    'vst',
    'rlog',
    'normTransform',
    
    # Results
    'results',
    'lfc_threshold_pvalue',
    'summary',
    
    # Outliers
    'calculate_cooks_distance',
    'detect_outliers',
    'replace_outliers',
    
    # Filtering
    'independent_filtering',
    'find_optimal_threshold',
    'filter_by_expression',
    
    # Design
    'create_design_matrix',
    'model_matrix',
    'parse_formula',
    'get_contrast_vector',
    
    # Plotting
    'plotMA',
    'plotVolcano',
    'plotDispEsts',
    'plotPCA',
    'plotCounts',
    'plotHeatmap',
    
    # Utilities
    'fpm',
    'fpkm',
    'tpm',
    'normalize_counts',
    'filter_low_counts',
    
    # PyDESeq2 compat
    'is_pydeseq2_available',
    'PyDESeq2Wrapper',
    'run_deseq2_analysis',
]
