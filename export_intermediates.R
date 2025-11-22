# Save this as export_intermediates.R and run it with: Rscript export_intermediates.R
suppressPackageStartupMessages(library(DESeq2))

print("Loading data...")
counts <- read.csv("data/counts.csv", row.names=1)
coldata <- read.csv("data/coldata.csv", row.names=1)

# Ensure factors
coldata$dex <- factor(coldata$dex)

print("Running DESeq2...")
dds <- DESeqDataSetFromMatrix(countData = counts, colData = coldata, design = ~ dex)
dds <- DESeq(dds)

# --- TEST 1 DATA: Size Factors ---
print("Exporting Size Factors...")
sf <- sizeFactors(dds)
write.csv(data.frame(sample=names(sf), sizeFactor=sf), "data/r_size_factors.csv", row.names=FALSE)

# --- TEST 2 DATA: Dispersions ---
print("Exporting Dispersions...")
# Get the final dispersions used by the model
disps <- dispersions(dds)
# Map them to gene names
df_disp <- data.frame(gene=rownames(dds), dispersion=disps)
write.csv(df_disp, "data/r_dispersions.csv", row.names=FALSE)

print("Done! Saved data/r_size_factors.csv and data/r_dispersions.csv")