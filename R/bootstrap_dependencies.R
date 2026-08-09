args <- commandArgs(trailingOnly = TRUE)
workspace <- if (length(args) >= 1L) normalizePath(args[[1L]], mustWork = TRUE) else getwd()
project_lib <- file.path(workspace, ".r-library")
dir.create(project_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(project_lib, .libPaths()))

if (!requireNamespace("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", repos = "https://cloud.r-project.org", lib = project_lib)
}

cran_packages <- c("data.table", "jsonlite", "digest")
missing_cran <- cran_packages[!vapply(cran_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_cran)) {
  install.packages(missing_cran, repos = "https://cloud.r-project.org", lib = project_lib)
}

bioc_packages <- c(
  "DESeq2",
  "BiocParallel"
)
missing_bioc <- bioc_packages[!vapply(bioc_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_bioc)) {
  BiocManager::install(
    missing_bioc,
    lib = project_lib,
    ask = FALSE,
    update = FALSE,
    dependencies = TRUE
  )
}

required <- c(cran_packages, bioc_packages)
status <- vapply(required, requireNamespace, logical(1), quietly = TRUE)
for (package in required) {
  version <- if (status[[package]]) as.character(packageVersion(package)) else "MISSING"
  cat(package, status[[package]], version, "\n")
}
if (!all(status)) {
  stop("One or more required R packages remain unavailable")
}
