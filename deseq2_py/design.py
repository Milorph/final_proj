"""
Design matrix construction for DESeq2-like analysis.

This module provides functions to parse R-style formulas and create
design matrices for generalized linear models.

References:
    - Wilkinson GN, Rogers CE (1973). Symbolic description of factorial 
      models for analysis of variance. Applied Statistics 22:392-399
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
"""

import numpy as np
import pandas as pd
from patsy import dmatrix, dmatrices


def parse_formula(formula):
    """
    Parse an R-style formula string.
    
    Handles formulas like:
    - "~ condition"
    - "~ condition + batch"
    - "~ condition * batch" (includes interaction)
    - "~ condition:batch" (interaction only)
    
    Parameters
    ----------
    formula : str
        R-style formula string (e.g., "~ condition + batch").
        
    Returns
    -------
    dict
        Parsed formula components:
        - 'terms': list of main effect terms
        - 'interactions': list of interaction terms
        - 'has_intercept': bool
        
    Examples
    --------
    >>> parse_formula("~ condition + batch")
    {'terms': ['condition', 'batch'], 'interactions': [], 'has_intercept': True}
    
    >>> parse_formula("~ condition * batch")
    {'terms': ['condition', 'batch'], 'interactions': [('condition', 'batch')], 
     'has_intercept': True}
    """
    formula = formula.strip()
    
    # Remove leading ~ if present
    if formula.startswith("~"):
        formula = formula[1:].strip()
    
    has_intercept = True
    if formula.startswith("0") or formula.startswith("-1"):
        has_intercept = False
        formula = formula.lstrip("0-1").lstrip("+").strip()
    
    terms = []
    interactions = []
    
    # Split by + while respecting *
    parts = [p.strip() for p in formula.split("+")]
    
    for part in parts:
        if "*" in part:
            # Interaction term: a * b expands to a + b + a:b
            sub_terms = [t.strip() for t in part.split("*")]
            terms.extend(sub_terms)
            # Add pairwise interactions
            for i in range(len(sub_terms)):
                for j in range(i + 1, len(sub_terms)):
                    interactions.append((sub_terms[i], sub_terms[j]))
        elif ":" in part:
            # Explicit interaction only
            sub_terms = tuple(t.strip() for t in part.split(":"))
            interactions.append(sub_terms)
        else:
            if part:
                terms.append(part)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique_terms.append(t)
    
    return {
        'terms': unique_terms,
        'interactions': interactions,
        'has_intercept': has_intercept
    }


def create_design_matrix(coldata, formula="~ condition"):
    """
    Create a design matrix from sample metadata and a formula.
    
    Uses patsy to construct a full-rank design matrix suitable for
    use in generalized linear models.
    
    Parameters
    ----------
    coldata : pd.DataFrame
        Sample metadata with experimental variables as columns.
        Row names should be sample identifiers.
    formula : str, default "~ condition"
        R-style formula specifying the model. Use 'C(var)' for
        explicit categorical treatment.
        
    Returns
    -------
    np.ndarray
        Design matrix (samples x parameters).
    list
        Column names for the design matrix.
        
    Raises
    ------
    ValueError
        If formula references columns not in coldata.
        
    Examples
    --------
    >>> import pandas as pd
    >>> coldata = pd.DataFrame({
    ...     'condition': ['ctrl', 'ctrl', 'treat', 'treat'],
    ...     'batch': ['A', 'B', 'A', 'B']
    ... })
    >>> X, names = create_design_matrix(coldata, "~ condition")
    >>> print(names)  # ['Intercept', 'condition[T.treat]']
    
    >>> X, names = create_design_matrix(coldata, "~ condition + batch")
    >>> print(names)  # ['Intercept', 'condition[T.treat]', 'batch[T.B]']
    
    >>> X, names = create_design_matrix(coldata, "~ condition * batch")
    >>> # Includes interaction term
    
    Notes
    -----
    The design matrix uses treatment (dummy) coding by default, where
    the first level of each categorical variable is the reference.
    Use `C(var, Treatment(reference='level'))` to specify a different
    reference level.
    """
    if not isinstance(coldata, pd.DataFrame):
        raise TypeError("coldata must be a pandas DataFrame")
    
    # Create design matrix using patsy
    design_info = dmatrix(formula, data=coldata, return_type='dataframe')
    
    X = design_info.values
    column_names = list(design_info.columns)
    
    return X, column_names


def model_matrix(coldata, formula="~ condition", return_dataframe=False):
    """
    Create a model matrix from sample metadata.
    
    This is an alias for create_design_matrix with an option to
    return a DataFrame instead of array.
    
    Parameters
    ----------
    coldata : pd.DataFrame
        Sample metadata with experimental variables as columns.
    formula : str, default "~ condition"
        R-style formula specifying the model.
    return_dataframe : bool, default False
        If True, return a DataFrame with named columns.
        
    Returns
    -------
    np.ndarray or pd.DataFrame
        Design matrix (samples x parameters).
        
    Examples
    --------
    >>> import pandas as pd
    >>> coldata = pd.DataFrame({'condition': ['A', 'A', 'B', 'B']})
    >>> X = model_matrix(coldata, "~ condition")
    >>> X_df = model_matrix(coldata, "~ condition", return_dataframe=True)
    """
    X, names = create_design_matrix(coldata, formula)
    
    if return_dataframe:
        return pd.DataFrame(X, columns=names, index=coldata.index)
    return X


def get_contrast_vector(design_matrix_columns, contrast):
    """
    Create a numeric contrast vector from a contrast specification.
    
    Parameters
    ----------
    design_matrix_columns : list
        Column names of the design matrix.
    contrast : list or tuple or np.ndarray
        Contrast specification. Can be:
        - List of three strings: [factor, level1, level2] for level1 - level2
        - Numeric array of length matching design matrix columns
        
    Returns
    -------
    np.ndarray
        Numeric contrast vector.
        
    Examples
    --------
    >>> cols = ['Intercept', 'condition[T.treat]']
    >>> vec = get_contrast_vector(cols, ['condition', 'treat', 'ctrl'])
    >>> print(vec)  # [0, 1] (treat vs ctrl)
    
    >>> vec = get_contrast_vector(cols, [0, 1])
    >>> print(vec)  # [0, 1]
    
    Notes
    -----
    For a three-element list [factor, numerator, denominator]:
    - Finds the column corresponding to the numerator level
    - The contrast tests numerator vs denominator
    - Returns a vector with 1 for numerator, 0 elsewhere
    """
    if isinstance(contrast, np.ndarray) or (isinstance(contrast, list) and 
                                            all(isinstance(x, (int, float)) for x in contrast)):
        # Already numeric
        vec = np.array(contrast, dtype=float)
        if len(vec) != len(design_matrix_columns):
            raise ValueError(f"Contrast vector length ({len(vec)}) must match "
                           f"number of design matrix columns ({len(design_matrix_columns)})")
        return vec
    
    if isinstance(contrast, (list, tuple)) and len(contrast) == 3:
        factor, numerator, denominator = contrast
        
        # Look for the coefficient column
        vec = np.zeros(len(design_matrix_columns), dtype=float)
        
        # Try to find the column for numerator level
        # Pattern: factor[T.level] or factor[level]
        patterns = [
            f"{factor}[T.{numerator}]",
            f"{factor}[{numerator}]",
            f"C({factor})[T.{numerator}]",
            f"C({factor})[{numerator}]"
        ]
        
        found = False
        for i, col in enumerate(design_matrix_columns):
            for pattern in patterns:
                if col == pattern:
                    vec[i] = 1.0
                    found = True
                    break
            if found:
                break
        
        if not found:
            # Maybe denominator is the reference level (intercept case)
            # Try finding numerator directly
            for i, col in enumerate(design_matrix_columns):
                if numerator in col and factor in col:
                    vec[i] = 1.0
                    found = True
                    break
        
        if not found:
            raise ValueError(f"Could not find contrast column for {contrast}")
        
        return vec
    
    raise ValueError(f"Invalid contrast specification: {contrast}")


def check_full_rank(X):
    """
    Check if a design matrix is full rank.
    
    A full-rank design matrix is required for unique parameter
    estimation in GLMs.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (samples x parameters).
        
    Returns
    -------
    bool
        True if matrix is full rank, False otherwise.
        
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 0], [1, 0], [1, 1], [1, 1]])
    >>> check_full_rank(X)  # True
    
    >>> X_singular = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 2], [1, 1, 2]])
    >>> check_full_rank(X_singular)  # False (columns are linearly dependent)
    """
    X = np.asarray(X, dtype=float)
    rank = np.linalg.matrix_rank(X)
    return rank == min(X.shape)


def drop_low_variance_columns(X, threshold=1e-10):
    """
    Remove columns with near-zero variance from design matrix.
    
    Columns with very low variance can cause numerical instability
    in GLM fitting.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (samples x parameters).
    threshold : float, default 1e-10
        Minimum variance threshold.
        
    Returns
    -------
    np.ndarray
        Design matrix with low-variance columns removed.
    np.ndarray
        Boolean mask of kept columns.
    """
    X = np.asarray(X, dtype=float)
    variances = np.var(X, axis=0)
    keep_mask = variances > threshold
    return X[:, keep_mask], keep_mask
