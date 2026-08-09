args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop("Usage: Rscript R/02_export_model_matrix.R <workspace>")

workspace <- normalizePath(args[[1L]], mustWork = TRUE)
project_lib <- file.path(workspace, ".r-library")
.libPaths(c(project_lib, .libPaths()))

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
})

vst_path <- file.path(workspace, "artifacts", "interim", "tcga_luad_selected_vst.rds")
map_path <- file.path(workspace, "artifacts", "interim", "gencode_v36_gene_mapping.csv")
cohort_path <- file.path(workspace, "artifacts", "interim", "tcga_luad_patient_cohort.csv")
out_path <- file.path(workspace, "artifacts", "interim", "tcga_luad_model_matrix.csv.gz")
audit_path <- file.path(workspace, "artifacts", "audit", "model_matrix_audit.json")

vst_object <- readRDS(vst_path)
stopifnot(is.list(vst_object), is.matrix(vst_object$matrix))
vst <- vst_object$matrix
mapping <- fread(map_path)
cohort <- fread(cohort_path)
stopifnot(!anyDuplicated(mapping$gene_id), all(rownames(vst) %in% mapping$gene_id))
mapping <- mapping[match(rownames(vst), gene_id)]
stopifnot(identical(rownames(vst), mapping$gene_id))
stopifnot(identical(colnames(vst), cohort$selected_barcode))

keep <- mapping$gene_type == "protein_coding" & nzchar(mapping$gene_symbol)
x <- vst[keep, , drop = FALSE]
symbols <- mapping$gene_symbol[keep]
variances <- apply(x, 1L, var)

# A few GENCODE symbols have multiple exact loci (notably pseudoautosomal genes).
# Select the highest-variance exact locus deterministically; never sum VST values.
ord <- order(symbols, -variances, rownames(x))
x <- x[ord, , drop = FALSE]
symbols <- symbols[ord]
variances <- variances[ord]
deduplicated <- !duplicated(symbols)
x <- x[deduplicated, , drop = FALSE]
symbols <- symbols[deduplicated]
variances <- variances[deduplicated]
rownames(x) <- symbols
unique_protein_coding_symbols <- nrow(x)

top_n <- min(2000L, nrow(x))
top <- order(variances, decreasing = TRUE)[seq_len(top_n)]
x <- x[top, , drop = FALSE]

model_dt <- as.data.table(t(x), keep.rownames = "selected_barcode")
stopifnot(identical(model_dt$selected_barcode, cohort$selected_barcode))
fwrite(model_dt, out_path, compress = "gzip")

audit <- list(
  outcome_used = FALSE,
  source = basename(vst_path),
  source_gene_rows = nrow(vst),
  source_samples = ncol(vst),
  protein_coding_exact_rows = sum(keep),
  unique_protein_coding_symbols = unique_protein_coding_symbols,
  duplicated_symbols_resolved_by_highest_variance = sum(duplicated(mapping$gene_symbol[keep])),
  outcome_blind_variance_features = top_n,
  output_samples = nrow(model_dt),
  output_features = ncol(model_dt) - 1L,
  output_file = basename(out_path)
)
write_json(audit, audit_path, auto_unbox = TRUE, pretty = TRUE)
cat("Exported", nrow(model_dt), "samples x", ncol(model_dt) - 1L, "features\n")
