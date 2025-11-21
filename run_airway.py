import pandas as pd
import numpy as np

from deseq2_py.deseq import run_deseq

# Load data
counts = pd.read_csv("data/counts.csv", index_col=0).values
coldata = pd.read_csv("data/coldata.csv", index_col=0)

# Extract condition labels (DESeq2 convention: "dex" = treated vs control)
condition = coldata["dex"].values

# Run pipeline
result = run_deseq(counts, condition)

# Convert to DataFrame for human readability
res_df = pd.DataFrame(result)

print("\nTop DE genes:")
print(res_df.sort_values("pvalue").head(10))

# Get rownames from counts DataFrame index instead of default 0..N
counts_df = pd.read_csv("data/counts.csv", index_col=0)
res_df.index = counts_df.index

res_df.to_csv("data/results_python.csv")
print("\nSaved corrected: data/results_python.csv")
