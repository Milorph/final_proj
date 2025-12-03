"""
Plotting functions for DESeq2-like analysis.

This module provides visualization functions commonly used in
RNA-seq differential expression analysis.

References:
    - Love MI, Huber W, Anders S (2014). Moderated estimation of fold change 
      and dispersion for RNA-seq data with DESeq2. Genome Biology 15:550
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plotMA(result, alpha=0.05, ylim=None, main="MA Plot", 
           point_size=5, point_alpha=0.5, show_legend=True,
           colSig="red", colNonSig="gray", ax=None):
    """
    MA plot showing log2 fold change vs mean expression.
    
    The MA plot is a scatter plot where each point represents a gene,
    with the x-axis showing mean expression (A) and y-axis showing
    log2 fold change (M).
    
    Parameters
    ----------
    result : dict or pd.DataFrame
        DESeq2 results containing 'baseMean', 'log2FoldChange', and 'padj'.
    alpha : float, default 0.05
        Significance threshold for coloring points.
    ylim : tuple, optional
        Y-axis limits. If None, determined automatically.
    main : str, default "MA Plot"
        Plot title.
    point_size : float, default 5
        Size of scatter points.
    point_alpha : float, default 0.5
        Transparency of points.
    show_legend : bool, default True
        Whether to show legend.
    colSig : str, default "red"
        Color for significant genes.
    colNonSig : str, default "gray"
        Color for non-significant genes.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
        
    Examples
    --------
    >>> from deseq2_py.plotting import plotMA
    >>> plotMA(result, alpha=0.05)
    >>> plt.show()
    """
    if isinstance(result, dict):
        base_mean = np.asarray(result['baseMean'])
        log2_fc = np.asarray(result['log2FoldChange'])
        padj = np.asarray(result.get('padj', result.get('pvalue', np.ones(len(base_mean)))))
    else:
        base_mean = result['baseMean'].values
        log2_fc = result['log2FoldChange'].values
        padj = result['padj'].values if 'padj' in result.columns else result['pvalue'].values
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Define colors based on significance
    significant = (padj < alpha) & np.isfinite(padj)
    colors = np.where(significant, colSig, colNonSig)
    
    # Filter valid values
    valid = (base_mean > 0) & np.isfinite(log2_fc)
    
    # Plot non-significant genes first
    mask_ns = valid & ~significant
    ax.scatter(base_mean[mask_ns], log2_fc[mask_ns], 
               c=colNonSig, s=point_size, alpha=point_alpha, label='NS')
    
    # Plot significant genes on top
    mask_sig = valid & significant
    ax.scatter(base_mean[mask_sig], log2_fc[mask_sig],
               c=colSig, s=point_size, alpha=point_alpha, label=f'padj < {alpha}')
    
    ax.set_xscale('log')
    ax.axhline(y=0, color='blue', linestyle='--', linewidth=0.5)
    
    ax.set_xlabel('Mean of Normalized Counts')
    ax.set_ylabel('Log2 Fold Change')
    ax.set_title(main)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if show_legend:
        ax.legend(loc='upper right')
    
    return ax


def plotVolcano(result, alpha=0.05, lfc_threshold=1.0, main="Volcano Plot",
                point_size=5, point_alpha=0.5, ax=None,
                colUp="red", colDown="blue", colNS="gray"):
    """
    Volcano plot showing -log10(p-value) vs log2 fold change.
    
    Points are colored by significance and direction of change.
    
    Parameters
    ----------
    result : dict or pd.DataFrame
        DESeq2 results containing 'log2FoldChange' and 'pvalue'.
    alpha : float, default 0.05
        Significance threshold (applied to adjusted p-values if available).
    lfc_threshold : float, default 1.0
        Log2 fold change threshold for highlighting.
    main : str, default "Volcano Plot"
        Plot title.
    point_size : float, default 5
        Size of scatter points.
    point_alpha : float, default 0.5
        Transparency of points.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    colUp : str, default "red"
        Color for upregulated genes.
    colDown : str, default "blue"
        Color for downregulated genes.
    colNS : str, default "gray"
        Color for non-significant genes.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
        
    Examples
    --------
    >>> from deseq2_py.plotting import plotVolcano
    >>> plotVolcano(result, alpha=0.05, lfc_threshold=1.0)
    >>> plt.show()
    """
    if isinstance(result, dict):
        log2_fc = np.asarray(result['log2FoldChange'])
        pvalue = np.asarray(result['pvalue'])
        padj = np.asarray(result.get('padj', pvalue))
    else:
        log2_fc = result['log2FoldChange'].values
        pvalue = result['pvalue'].values
        padj = result['padj'].values if 'padj' in result.columns else pvalue
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate -log10(pvalue)
    with np.errstate(divide='ignore'):
        neg_log10_p = -np.log10(pvalue)
    neg_log10_p = np.clip(neg_log10_p, 0, 300)  # Avoid inf
    
    # Define categories
    significant = (padj < alpha) & np.isfinite(padj)
    up = significant & (log2_fc > lfc_threshold)
    down = significant & (log2_fc < -lfc_threshold)
    ns = ~(up | down)
    
    # Plot
    valid = np.isfinite(log2_fc) & np.isfinite(neg_log10_p)
    
    ax.scatter(log2_fc[valid & ns], neg_log10_p[valid & ns],
               c=colNS, s=point_size, alpha=point_alpha, label='NS')
    ax.scatter(log2_fc[valid & up], neg_log10_p[valid & up],
               c=colUp, s=point_size, alpha=point_alpha, label='Up')
    ax.scatter(log2_fc[valid & down], neg_log10_p[valid & down],
               c=colDown, s=point_size, alpha=point_alpha, label='Down')
    
    # Add threshold lines
    ax.axhline(y=-np.log10(alpha), color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=lfc_threshold, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=-lfc_threshold, color='gray', linestyle='--', linewidth=0.5)
    
    ax.set_xlabel('Log2 Fold Change')
    ax.set_ylabel('-Log10(p-value)')
    ax.set_title(main)
    ax.legend(loc='upper right')
    
    return ax


def plotDispEsts(base_means, disp_gw, disp_trend=None, disp_final=None,
                 main="Dispersion Estimates", ax=None,
                 show_legend=True, xlim=None, ylim=None):
    """
    Plot dispersion estimates vs mean expression with trend line.
    
    Shows gene-wise dispersion estimates (gray), fitted trend (red),
    and final (MAP) estimates (blue).
    
    Parameters
    ----------
    base_means : np.ndarray
        Mean normalized counts per gene.
    disp_gw : np.ndarray
        Gene-wise dispersion estimates.
    disp_trend : np.ndarray, optional
        Fitted trend values.
    disp_final : np.ndarray, optional
        Final (MAP) dispersion estimates.
    main : str, default "Dispersion Estimates"
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    show_legend : bool, default True
        Whether to show legend.
    xlim, ylim : tuple, optional
        Axis limits.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
        
    Examples
    --------
    >>> from deseq2_py.plotting import plotDispEsts
    >>> plotDispEsts(base_means, disp_gw, disp_trend)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter valid values
    valid = (base_means > 0) & (disp_gw > 1e-10) & np.isfinite(disp_gw)
    
    # Plot gene-wise estimates
    ax.scatter(base_means[valid], disp_gw[valid], c='black', s=2, alpha=0.3,
               label='Gene-wise')
    
    # Plot final estimates if provided
    if disp_final is not None:
        valid_f = valid & (disp_final > 1e-10) & np.isfinite(disp_final)
        ax.scatter(base_means[valid_f], disp_final[valid_f], c='dodgerblue', 
                   s=2, alpha=0.5, label='Final')
    
    # Plot trend if provided
    if disp_trend is not None:
        # Sort for line plot
        order = np.argsort(base_means[valid])
        ax.plot(base_means[valid][order], disp_trend[valid][order], 
                c='red', linewidth=2, label='Trend')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Mean of Normalized Counts')
    ax.set_ylabel('Dispersion')
    ax.set_title(main)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if show_legend:
        ax.legend(loc='upper right')
    
    return ax


def plotPCA(transformed_data, sample_info=None, color_by=None, 
            main="PCA Plot", n_top=500, ax=None, point_size=50):
    """
    PCA plot of transformed expression data.
    
    Performs PCA on the most variable genes and plots the first two
    principal components, colored by an experimental factor.
    
    Parameters
    ----------
    transformed_data : np.ndarray or pd.DataFrame
        Transformed expression data (genes x samples), e.g., from vst() or rlog().
    sample_info : pd.DataFrame, optional
        Sample metadata with columns for coloring.
    color_by : str, optional
        Column name in sample_info to use for coloring points.
    main : str, default "PCA Plot"
        Plot title.
    n_top : int, default 500
        Number of most variable genes to use for PCA.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    point_size : float, default 50
        Size of scatter points.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
    tuple
        Variance explained by PC1 and PC2.
        
    Examples
    --------
    >>> from deseq2_py.plotting import plotPCA
    >>> from deseq2_py.transformations import vst
    >>> vst_data = vst(counts, size_factors)
    >>> plotPCA(vst_data, sample_info, color_by='condition')
    >>> plt.show()
    """
    if isinstance(transformed_data, pd.DataFrame):
        data = transformed_data.values
        sample_names = transformed_data.columns
    else:
        data = np.asarray(transformed_data)
        sample_names = None
    
    G, S = data.shape
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Select top variable genes
    var_genes = np.var(data, axis=1)
    top_idx = np.argsort(var_genes)[-n_top:]
    data_subset = data[top_idx, :]
    
    # Center the data
    data_centered = data_subset - data_subset.mean(axis=1, keepdims=True)
    
    # SVD for PCA
    U, S_vals, Vt = np.linalg.svd(data_centered.T, full_matrices=False)
    
    # PC scores
    pc1 = U[:, 0]
    pc2 = U[:, 1]
    
    # Variance explained
    var_exp = (S_vals ** 2) / np.sum(S_vals ** 2)
    var_pc1 = var_exp[0] * 100
    var_pc2 = var_exp[1] * 100
    
    # Color by factor if provided
    if sample_info is not None and color_by is not None:
        if isinstance(sample_info, pd.DataFrame) and color_by in sample_info.columns:
            factor = sample_info[color_by].values
            unique_levels = np.unique(factor)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_levels)))
            
            for i, level in enumerate(unique_levels):
                mask = factor == level
                ax.scatter(pc1[mask], pc2[mask], c=[colors[i]], 
                          s=point_size, label=str(level), alpha=0.8)
            ax.legend(title=color_by)
        else:
            ax.scatter(pc1, pc2, s=point_size, alpha=0.8)
    else:
        ax.scatter(pc1, pc2, s=point_size, alpha=0.8)
    
    ax.set_xlabel(f'PC1: {var_pc1:.1f}% variance')
    ax.set_ylabel(f'PC2: {var_pc2:.1f}% variance')
    ax.set_title(main)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    
    return ax, (var_pc1, var_pc2)


def plotCounts(counts, gene, size_factors=None, condition=None,
               main=None, ax=None, jitter=0.1, normalized=True):
    """
    Plot normalized counts for a single gene.
    
    Shows individual sample counts, optionally grouped by experimental
    condition.
    
    Parameters
    ----------
    counts : np.ndarray or pd.DataFrame
        Raw count matrix (genes x samples).
    gene : int or str
        Gene index (if array) or gene name (if DataFrame).
    size_factors : np.ndarray, optional
        Size factors for normalization. If None and normalized=True,
        will be estimated.
    condition : np.ndarray or pd.Series, optional
        Condition labels for grouping samples.
    main : str, optional
        Plot title. If None, uses gene name/index.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.
    jitter : float, default 0.1
        Amount of horizontal jitter for points.
    normalized : bool, default True
        Whether to plot normalized counts.
        
    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the plot.
        
    Examples
    --------
    >>> from deseq2_py.plotting import plotCounts
    >>> plotCounts(counts_df, gene='BRCA1', condition=coldata['dex'])
    >>> plt.show()
    """
    if isinstance(counts, pd.DataFrame):
        if isinstance(gene, str):
            gene_counts = counts.loc[gene].values
            gene_name = gene
        else:
            gene_counts = counts.iloc[gene].values
            gene_name = counts.index[gene]
    else:
        gene_counts = np.asarray(counts)[gene, :]
        gene_name = f"Gene {gene}"
    
    S = len(gene_counts)
    
    if normalized:
        if size_factors is None:
            from .size_factors import estimate_size_factors
            size_factors = estimate_size_factors(counts if not isinstance(counts, pd.DataFrame) 
                                                  else counts.values)
        gene_counts = gene_counts / size_factors
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    if main is None:
        main = f"Counts: {gene_name}"
    
    if condition is not None:
        condition = np.asarray(condition)
        unique_cond = np.unique(condition)
        x_pos = []
        for i, cond in enumerate(unique_cond):
            mask = condition == cond
            n_samples = mask.sum()
            x_jitter = i + np.random.uniform(-jitter, jitter, n_samples)
            x_pos.append((i, cond, mask, x_jitter))
        
        for i, cond, mask, x_j in x_pos:
            ax.scatter(x_j, gene_counts[mask], label=str(cond), s=40, alpha=0.7)
        
        ax.set_xticks(range(len(unique_cond)))
        ax.set_xticklabels(unique_cond)
        ax.set_xlabel('Condition')
    else:
        ax.scatter(range(S), gene_counts, s=40, alpha=0.7)
        ax.set_xlabel('Sample')
    
    ax.set_ylabel('Normalized Counts' if normalized else 'Raw Counts')
    ax.set_title(main)
    
    return ax


def plotHeatmap(data, row_labels=None, col_labels=None, 
                cluster_rows=True, cluster_cols=True,
                cmap='RdBu_r', vmin=None, vmax=None,
                main="Heatmap", figsize=(10, 8)):
    """
    Plot a heatmap of expression data with optional clustering.
    
    Parameters
    ----------
    data : np.ndarray or pd.DataFrame
        Expression data matrix (genes x samples).
    row_labels : list, optional
        Labels for rows (genes).
    col_labels : list, optional
        Labels for columns (samples).
    cluster_rows : bool, default True
        Whether to cluster rows.
    cluster_cols : bool, default True
        Whether to cluster columns.
    cmap : str, default 'RdBu_r'
        Colormap name.
    vmin, vmax : float, optional
        Value range for colormap.
    main : str, default "Heatmap"
        Plot title.
    figsize : tuple, default (10, 8)
        Figure size.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    matplotlib.axes.Axes
        The axes object.
        
    Examples
    --------
    >>> from deseq2_py.plotting import plotHeatmap
    >>> # Plot top 50 variable genes
    >>> var_idx = np.argsort(np.var(vst_data, axis=1))[-50:]
    >>> plotHeatmap(vst_data[var_idx, :], cluster_rows=True)
    >>> plt.show()
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    
    if isinstance(data, pd.DataFrame):
        if row_labels is None:
            row_labels = data.index.tolist()
        if col_labels is None:
            col_labels = data.columns.tolist()
        data = data.values
    
    data = np.asarray(data, dtype=float)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Cluster if requested
    row_order = np.arange(data.shape[0])
    col_order = np.arange(data.shape[1])
    
    if cluster_rows and data.shape[0] > 1:
        try:
            row_dist = pdist(data)
            row_link = linkage(row_dist, method='average')
            row_order = dendrogram(row_link, no_plot=True)['leaves']
        except Exception:
            pass
    
    if cluster_cols and data.shape[1] > 1:
        try:
            col_dist = pdist(data.T)
            col_link = linkage(col_dist, method='average')
            col_order = dendrogram(col_link, no_plot=True)['leaves']
        except Exception:
            pass
    
    data_ordered = data[row_order, :][:, col_order]
    
    # Plot heatmap
    im = ax.imshow(data_ordered, aspect='auto', cmap=cmap, 
                   vmin=vmin, vmax=vmax)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add labels
    if col_labels is not None and len(col_labels) <= 50:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels([col_labels[i] for i in col_order], 
                          rotation=45, ha='right')
    
    if row_labels is not None and len(row_labels) <= 50:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels([row_labels[i] for i in row_order])
    
    ax.set_title(main)
    
    plt.tight_layout()
    
    return fig, ax
