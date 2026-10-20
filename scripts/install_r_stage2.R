options(Ncpus = 12, repos = c(CRAN = "https://cloud.r-project.org"), timeout = 1800)
lib <- .libPaths()[1]
inst <- function(p) p %in% rownames(installed.packages())
B <- "/home/holiday01/2026_cm/baselines"

# xbioc: required by seAMLess per its README; not on CRAN
if (!inst("xbioc")) {
  cat(">>> installing pkgmaker/xbioc deps\n")
  for (p in c("pkgmaker","AnnotationDbi","Biobase","digest","stringr","plyr","reshape2","assertthat"))
    if (!inst(p)) try(BiocManager::install(p, lib=lib, ask=FALSE, update=FALSE))
  cat(">>> installing renozao/xbioc\n")
  try(remotes::install_github("renozao/xbioc", lib=lib, upgrade="never", dependencies=TRUE))
}

# seAMLessData from the author's drat repo
if (!inst("seAMLessData")) {
  cat(">>> installing seAMLessData from drat\n")
  r <- getOption("repos"); r["eonurk"] <- "https://eonurk.github.io/drat/"; options(repos = r)
  try(install.packages("seAMLessData", lib=lib))
}

# local clones
for (d in c("CLIMB","seAMLess","ClimbTheCliff")) {
  p <- file.path(B, d)
  cat(sprintf(">>> installing local package from %s\n", p))
  try(remotes::install_local(p, lib=lib, upgrade="never", dependencies=TRUE, force=TRUE))
}

cat("\n=== STAGE 2 FINAL STATUS ===\n")
for (p in c("Biobase","glmnet","optparse","data.table","randomForest","ggtern","nnls","TOAST",
            "MuSiC","xbioc","seAMLessData","climb","seAMLess","ClimbTheCliff"))
  cat(sprintf("  %-16s %s\n", p, if (inst(p)) paste("OK", as.character(packageVersion(p))) else "MISSING"))
