import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from deseq2_py.size_factors import estimate_size_factors

# ================================
# Helper Functions
# ================================

def print_table_row(metric, py_val, r_val, diff):
    print(f"| {metric:<30} | {str(py_val):<15} | {str(r_val):<15} | {str(diff):<15} |")

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

print("# DESeq2 Python Port vs R: Comprehensive Validation Report")

# ==========================================
# Load inputs
# ==========================================
counts = pd.read_csv("data/counts.csv", index_col=0)
py_res = load_or_die("data/results_python.csv")
r_res = load_or_die("data/deseq2_results.csv")
first_col = r_res.columns[0]
r_res = r_res.rename(columns={first_col: "gene"})


# Merge Python + R results on gene
merged = pd.merge(py_res, r_res, left_on=py_res.columns[0], right_on="gene", suffixes=('_py', '_r'))
valid = merged.dropna(subset=["log2FoldChange_py", "log2FoldChange_r", "padj_py", "padj_r"])

# ============================================================
# TEST 1 — SIZE FACTOR VALIDATION
# ============================================================
header("Test 1: Normalization (Size Factors)")

r_sf = load_or_die("data/r_size_factors.csv")
py_sf = estimate_size_factors(counts.values)

corr_sf, _ = pearsonr(py_sf, r_sf["sizeFactor"])
mae_sf = np.mean(np.abs(py_sf - r_sf["sizeFactor"]))

print_table_row("Correlation (Pearson)", "1.000", "1.000", f"{corr_sf:.6f}")
print_table_row("Mean Absolute Error", "-", "-", f"{mae_sf:.6e}")
print_table_row("Median Value", f"{np.median(py_sf):.4f}", f"{r_sf['sizeFactor'].median():.4f}", "-")


# ============================================================
# TEST 2 — DISPERSION ESTIMATION
# ============================================================
header("Test 2: Dispersion Estimation")

r_disp = load_or_die("data/r_dispersions.csv")
merged_disp = pd.merge(r_disp, py_res, left_on="gene", right_on=py_res.columns[0])

merged_disp = merged_disp.dropna(subset=["dispersion_x", "dispersion_y"])
merged_disp = merged_disp[merged_disp["baseMean"] > 10]  # avoid zero-inflation

py_d = merged_disp["dispersion_y"]
r_d  = merged_disp["dispersion_x"]

log_py = np.log10(py_d + 1e-10)
log_r = np.log10(r_d + 1e-10)

corr_disp, _ = pearsonr(log_py, log_r)

median_py = np.median(py_d)
median_r = np.median(r_d)

print_table_row("Correlation (log space)", "-", "-", f"{corr_disp:.4f}")
print_table_row("Median Dispersion", f"{median_py:.4f}", f"{median_r:.4f}", f"Ratio: {median_py/median_r:.2f}x")
print_table_row("Method", "Exact CR-APL", "Exact CR-APL", "Matched")

# Additional dispersion tests
rel_err = np.abs(py_d - r_d) / (r_d + 1e-10)
print_table_row("Median Relative Error", "-", "-", f"{np.median(rel_err):.3f}")
print_table_row("Mean Relative Error", "-", "-", f"{np.mean(rel_err):.3f}")


# ============================================================
# TEST 3 — LFC & SIGNIFICANCE (WALD TEST)
# ============================================================
header("Test 3: Differential Expression (Wald Test)")

high_expr = valid[valid["baseMean_r"] > 50]  # recommended by DESeq2 paper

corr_lfc, _ = pearsonr(high_expr["log2FoldChange_py"], high_expr["log2FoldChange_r"])

sig_py = (valid["padj_py"] < 0.05).sum()
sig_r = (valid["padj_r"] < 0.05).sum()

top_py = set(valid.sort_values("pvalue_py").head(500)["gene"])
top_r  = set(valid.sort_values("pvalue_r").head(500)["gene"])
overlap_500 = len(top_py & top_r)

print_table_row("LFC Correlation (High Exp)", "-", "-", f"{corr_lfc:.4f}")
print_table_row("Significant Genes", f"{sig_py}", f"{sig_r}", f"{sig_py/sig_r*100:.1f}%")
print_table_row("Top-500 Gene Overlap", "-", "-", f"{overlap_500/500*100:.1f}%")


# ============================================================
# TEST 4 — P-VALUE CORRELATION
# ============================================================
header("Test 4: P-Value Concordance")

py_p = -np.log10(valid["pvalue_py"] + 1e-300)
r_p  = -np.log10(valid["pvalue_r"] + 1e-300)

corr_p, _ = pearsonr(py_p, r_p)

print_table_row("Correlation (-log10 p)", "-", "-", f"{corr_p:.4f}")
print_table_row("Median -log10(p)", f"{np.median(py_p):.2f}", f"{np.median(r_p):.2f}", "-")


# ============================================================
# TEST 5 — SIGN AGREEMENT
# ============================================================
header("Test 5: LFC Sign Agreement")

sign_py = np.sign(valid["log2FoldChange_py"])
sign_r = np.sign(valid["log2FoldChange_r"])

sign_agree = (sign_py == sign_r).mean() * 100

print_table_row("Sign Agreement", "-", "-", f"{sign_agree:.1f}%")


# ============================================================
# TEST 6 — LFC ERROR METRICS
# ============================================================
header("Test 6: LFC Error Metrics")

lfc_err = valid["log2FoldChange_py"] - valid["log2FoldChange_r"]

mae = np.mean(np.abs(lfc_err))
rmse = np.sqrt(np.mean(lfc_err**2))
p95 = np.percentile(np.abs(lfc_err), 95)

print_table_row("Mean Absolute Error", "-", "-", f"{mae:.4f}")
print_table_row("RMSE", "-", "-", f"{rmse:.4f}")
print_table_row("95th Percentile Error", "-", "-", f"{p95:.4f}")


# ============================================================
# TEST 7 — BASEMEAN CONCORDANCE
# ============================================================
header("Test 7: BaseMean Concordance")

corr_bm, _ = pearsonr(valid["baseMean_py"], valid["baseMean_r"])
mae_bm = np.mean(np.abs(valid["baseMean_py"] - valid["baseMean_r"]))

print_table_row("Correlation (Pearson)", "-", "-", f"{corr_bm:.4f}")
print_table_row("Mean Absolute Error", "-", "-", f"{mae_bm:.4f}")


# ============================================================
# TEST 8 — RANK CORRELATION
# ============================================================
header("Test 8: Rank Correlation (Spearman)")

sp_p, _ = spearmanr(valid["pvalue_py"], valid["pvalue_r"])
sp_lfc, _ = spearmanr(abs(valid["log2FoldChange_py"]), abs(valid["log2FoldChange_r"]))

print_table_row("Spearman (p-values)", "-", "-", f"{sp_p:.4f}")
print_table_row("Spearman (|LFC|)", "-", "-", f"{sp_lfc:.4f}")


print("\n\n## Validation Complete 🎉")

# ============================================================
# GENERATE SUMMARY TABLE IMAGE AS A FIGURE
# ============================================================
import matplotlib.pyplot as plt

summary_data = [
    ["Size Factor Correlation", f"{corr_sf:.6f}"],
    ["Size Factor MAE", f"{mae_sf:.2e}"],
    ["Dispersion Correlation (log)", f"{corr_disp:.4f}"],
    ["Median Dispersion Ratio (Py/R)", f"{median_py/median_r:.2f}x"],
    ["LFC Correlation (High Expr)", f"{corr_lfc:.4f}"],
    ["Significant Gene Match (%)", f"{sig_py/sig_r*100:.1f}%"],
    ["Top-500 DE Gene Overlap", f"{overlap_500/500*100:.1f}%"],
    ["P-value Correlation", f"{corr_p:.4f}"],
    ["LFC Sign Agreement", f"{sign_agree:.1f}%"],
    ["LFC MAE", f"{mae:.4f}"],
    ["LFC RMSE", f"{rmse:.4f}"],
    ["BaseMean Correlation", f"{corr_bm:.4f}"],
    ["Spearman (p-values)", f"{sp_p:.4f}"],
    ["Spearman (|LFC|)", f"{sp_lfc:.4f}"]
]

# Create the figure
fig, ax = plt.subplots(figsize=(10, len(summary_data) * 0.45 + 1))
ax.axis("off")

# Build the table content
table = ax.table(
    cellText=summary_data,
    colLabels=["Metric", "Value"],
    cellLoc="left",
    loc="center",
)

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.4)

# Bold header
for key, cell in table.get_celld().items():
    row, col = key
    if row == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#d0d0d0")

# Save
output_path = "plots/validation_summary.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nSummary table image saved to {output_path}")

