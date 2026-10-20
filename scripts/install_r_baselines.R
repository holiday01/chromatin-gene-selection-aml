options(Ncpus = 12, repos = c(CRAN = "https://cloud.r-project.org"), timeout = 1800)
lib <- .libPaths()[1]
cat("lib:", lib, "\n")
inst <- function(p) p %in% rownames(installed.packages())

cran <- c("optparse","randomForest","nnls","plyr","reshape2","ggtern")
need <- cran[!sapply(cran, inst)]
if (length(need)) { cat("CRAN install:", paste(need, collapse=", "), "\n"); install.packages(need, lib=lib) }

if (!inst("TOAST")) { cat("Bioc install: TOAST\n"); BiocManager::install("TOAST", lib=lib, ask=FALSE, update=FALSE) }

if (!inst("MuSiC")) {
  cat("GitHub install: xuranw/MuSiC\n")
  remotes::install_github("xuranw/MuSiC", lib=lib, upgrade="never", dependencies=TRUE)
}

if (!inst("seAMLessData")) {
  cat("drat install: seAMLessData\n")
  r <- getOption("repos"); r["eonurk"] <- "https://eonurk.github.io/drat/"; options(repos = r)
  try(install.packages("seAMLessData", lib=lib))
}

cat("\n=== FINAL STATUS ===\n")
for (p in c("Biobase","glmnet","optparse","data.table","randomForest","ggtern","MuSiC","seAMLessData","TOAST","nnls","SingleCellExperiment"))
  cat(sprintf("  %-22s %s\n", p, if (inst(p)) paste("OK", as.character(packageVersion(p))) else "MISSING"))
