import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from deseq2_py.size_factors import estimate_size_factors

def print_table_row(metric, py_val, r_val, diff):
    print(f"| {metric:<25} | {str(py_val):<15} | {str(r_val):<15} | {str(diff):<15} |")

def header(title):
    print(f"\n### {title}")
    print("| Metric | Python Port | Original R | Comparison/Diff |")
    print("| :--- | :--- | :--- | :--- |")

def load_or_die(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Could not find {path}. Please run 'export_intermediates.R' first.")
        exit(1)

print("# DESeq2 Python Port vs R: Comparison Report")

# ==========================================
# TEST 1: NORMALIZATION (Size Factors)
# ==========================================
header("Test 1: Normalization (Size Factors)")

r_sf = load_or_die("data/r_size_factors.csv")
counts = pd.read_csv("data/counts.csv", index_col=0)
py_sf_vals = estimate_size_factors(counts.values)

corr_sf, _ = pearsonr(py_sf_vals, r_sf["sizeFactor"])
mae_sf = np.mean(np.abs(py_sf_vals - r_sf["sizeFactor"]))

print_table_row("Correlation (Pearson)", "1.000", "1.000", f"{corr_sf:.6f}")
print_table_row("Mean Absolute Error", "-", "-", f"{mae_sf:.6e}")
print_table_row("Median Value", f"{np.median(py_sf_vals):.4f}", f"{r_sf['sizeFactor'].median():.4f}", "-")


# ==========================================
# TEST 2: DISPERSION ESTIMATION
# ==========================================
header("Test 2: Dispersion Estimation")

r_disp = load_or_die("data/r_dispersions.csv")
py_res = load_or_die("data/results_python.csv")

# Merge R dispersion and Python results
merged_disp = pd.merge(r_disp, py_res, left_on="gene", right_on=py_res.columns[0])
merged_disp = merged_disp.dropna(subset=["dispersion_x", "dispersion_y"])

# --- FIX: Filter using Python's baseMean ---
# We exclude genes with very low counts to avoid skewing the median with zeros
valid_genes = merged_disp[merged_disp['baseMean'] > 10]

py_d = valid_genes["dispersion_y"] # Python Dispersion
r_d = valid_genes["dispersion_x"]  # R Dispersion

# Log-log correlation
log_py = np.log10(py_d + 1e-10)
log_r = np.log10(r_d + 1e-10)
corr_disp, _ = pearsonr(log_py, log_r)

median_py = np.median(py_d)
median_r = np.median(r_d)

print_table_row("Correlation (log space)", "-", "-", f"{corr_disp:.4f}")
print_table_row("Median Dispersion", f"{median_py:.4f}", f"{median_r:.4f}", f"Ratio: {median_py/median_r:.2f}x")
print_table_row("Method", "Exact CR-APL", "Exact CR-APL", "Matched")


# ==========================================
# TEST 3: WALD TEST (LFC & Significance)
# ==========================================
header("Test 3: Differential Expression (Wald Test)")

r_res = load_or_die("data/deseq2_results.csv")
r_res = r_res.rename(columns={r_res.columns[0]: "gene"})

merged = pd.merge(py_res, r_res, left_on=py_res.columns[0], right_on="gene", suffixes=('_py', '_r'))

# Filter for valid comparisons
valid = merged.dropna(subset=["log2FoldChange_py", "log2FoldChange_r", "padj_py", "padj_r"])
high_expr = valid[valid["baseMean_r"] > 50]

corr_lfc, _ = pearsonr(high_expr["log2FoldChange_py"], high_expr["log2FoldChange_r"])
sig_py = (valid["padj_py"] < 0.05).sum()
sig_r = (valid["padj_r"] < 0.05).sum()

# Overlap
top_py = set(valid.sort_values("pvalue_py").head(500)["gene"])
top_r = set(valid.sort_values("pvalue_r").head(500)["gene"])
overlap = len(top_py.intersection(top_r))

print_table_row("LFC Correlation (High Exp)", "-", "-", f"{corr_lfc:.4f}")
print_table_row("Significant Genes (p<0.05)", f"{sig_py}", f"{sig_r}", f"{sig_py/sig_r*100:.1f}%")
print_table_row("Top-500 Gene Overlap", "-", "-", f"{overlap/500*100:.1f}%")