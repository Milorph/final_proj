import pandas as pd
import numpy as np
import time

# CRITICAL CHANGE: Import from the new OPTIMIZED module
from deseq2_py.deseq_optimized import run_deseq 

def load_data():
    print("Loading data/counts.csv...")
    counts_df = pd.read_csv("data/counts.csv", index_col=0)
    print("Loading data/coldata.csv...")
    coldata_df = pd.read_csv("data/coldata.csv", index_col=0)
    return counts_df, coldata_df

def main():
    counts_df, coldata_df = load_data()
    
    # Use the column 'dex' which contains 'untrt' vs 'trt'
    condition_labels = coldata_df['dex'].values
    
    print(f"Running Optimized DESeq2 (CR-APL) on {counts_df.shape[0]} genes...")
    start_time = time.time()
    
    # This now calls the function in deseq_optimized.py
    res = run_deseq(counts_df.values, condition_labels)
    
    end_time = time.time()
    print(f"Done in {end_time - start_time:.1f} seconds.")
    
    # Convert results to DataFrame
    res_df = pd.DataFrame(res, index=counts_df.index)
    
    print("Saving results to data/results_python.csv...")
    res_df.to_csv("data/results_python.csv")

if __name__ == "__main__":
    main()