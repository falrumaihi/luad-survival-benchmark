args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop(
    "Usage: Rscript R/01_normalize_counts.R ",
    "<workspace> <selected_counts.csv> <vst_output.rds> <audit_output.json>"
  )
}

workspace <- normalizePath(args[[1L]], mustWork = TRUE)
counts_path <- normalizePath(args[[2L]], mustWork = TRUE)
vst_output <- args[[3L]]
audit_output <- args[[4L]]
project_lib <- file.path(workspace, ".r-library")
.libPaths(c(project_lib, .libPaths()))

required <- c("data.table", "DESeq2", "jsonlite", "digest")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  stop("Missing required R packages: ", paste(missing, collapse = ", "))
}

dir.create(dirname(vst_output), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(audit_output), recursive = TRUE, showWarnings = FALSE)

counts_dt <- data.table::fread(counts_path, check.names = FALSE, showProgress = TRUE)
if (names(counts_dt)[[1L]] != "gene") {
  stop("Selected raw-count matrix must have 'gene' as its first column")
}
gene_ids <- counts_dt[[1L]]
if (anyNA(gene_ids) || any(gene_ids == "") || anyDuplicated(gene_ids)) {
  stop("Gene identifiers must be present and exactly unique")
}

counts_dt[, gene := NULL]
sample_ids <- names(counts_dt)
counts <- as.matrix(counts_dt)
storage.mode(counts) <- "integer"
rownames(counts) <- gene_ids
if (anyNA(counts) || any(counts < 0L)) {
  stop("Counts must be non-negative integers without missing values")
}

minimum_samples <- max(10L, ceiling(ncol(counts) * 0.10))
keep <- rowSums(counts >= 10L) >= minimum_samples
filtered_counts <- counts[keep, , drop = FALSE]
if (nrow(filtered_counts) < 1000L) {
  stop("Expression filter retained fewer than 1000 genes; check input orientation")
}

col_data <- data.frame(
  intercept = rep.int(1L, length(sample_ids)),
  row.names = sample_ids
)
dds <- DESeq2::DESeqDataSetFromMatrix(
  countData = filtered_counts,
  colData = col_data,
  design = ~1
)
dds <- DESeq2::estimateSizeFactors(dds)
vst <- DESeq2::vst(dds, blind = TRUE)
vst_matrix <- SummarizedExperiment::assay(vst)

saveRDS(
  list(
    matrix = vst_matrix,
    size_factors = DESeq2::sizeFactors(dds),
    filter = list(minimum_count = 10L, minimum_samples = minimum_samples),
    gene_ids = rownames(vst_matrix),
    sample_ids = colnames(vst_matrix)
  ),
  file = vst_output,
  compress = "xz"
)

size_factors <- unname(DESeq2::sizeFactors(dds))
audit <- list(
  source_file = basename(counts_path),
  source_sha256 = digest::digest(file = counts_path, algo = "sha256"),
  input_gene_rows = length(gene_ids),
  input_samples = length(sample_ids),
  retained_gene_rows = nrow(vst_matrix),
  removed_gene_rows = length(gene_ids) - nrow(vst_matrix),
  filter_minimum_count = 10L,
  filter_minimum_samples = minimum_samples,
  size_factor_min = min(size_factors),
  size_factor_median = stats::median(size_factors),
  size_factor_max = max(size_factors),
  transform = "DESeq2 varianceStabilizingTransformation; blind=TRUE; design=~1",
  outcome_used = FALSE,
  r_version = R.version.string,
  package_versions = list(
    DESeq2 = as.character(packageVersion("DESeq2")),
    SummarizedExperiment = as.character(packageVersion("SummarizedExperiment")),
    data.table = as.character(packageVersion("data.table"))
  )
)
jsonlite::write_json(audit, audit_output, pretty = TRUE, auto_unbox = TRUE)
cat(
  "Normalized selected count matrix:",
  nrow(vst_matrix), "genes x", ncol(vst_matrix), "samples\n"
)
