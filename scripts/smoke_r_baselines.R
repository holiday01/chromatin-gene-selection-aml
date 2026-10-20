sink(stdout(), type = "message")
cat("######## 1. climb on its bundled in-vitro example ########\n")
suppressPackageStartupMessages({library(climb); library(ClimbTheCliff); library(Biobase); library(glmnet); library(reshape2)})
# CLIMB's NAMESPACE declares only glmnet and Biobase, but its code calls glmnet(), dcast() and
# flatten() unqualified. glmnet and reshape2 are attached above. `flatten` is not purrr's
# (purrr::flatten rejects a matrix, and the argument here is built with do.call(rbind, ...)).
# Context is quantile(flatten(M), p = 0.95), i.e. the 95th percentile of every entry of M, so the
# faithful equivalent is as.vector(). Supplied in the global environment, which the package
# namespace falls through to; the package source is NOT modified.
flatten <- function(x) as.vector(x)
data(sc.es); data(bulk.es)
cat("sc.es dim  :", paste(dim(sc.es), collapse=" x "), "\n")
cat("bulk.es dim:", paste(dim(bulk.es), collapse=" x "), "\n")
true.prop <- as.matrix(pData(bulk.es))
cat("true proportions:\n"); print(round(true.prop, 3))
t0 <- Sys.time()
# CLIMB 1.1.2 crashes on the default "abundance" path with
# `object 'S_pred_mapping_norm' not found` because that array is used but never initialised.
# ClimbTheCliff (the author's newer 2026-08 package) initialises it at R/climb.R:231, so use that one.
res <- ClimbTheCliff::climb(sc.es, bulk.es, verbose = FALSE)
cat("climb() finished in", round(as.numeric(difftime(Sys.time(), t0, units="secs")),1), "s\n")
cat("climb() returned:", paste(class(res), collapse="/"), "| names:", paste(names(res), collapse=", "), "\n")
W <- res$props.corrected
if (is.null(W)) W <- res$props
if (is.null(W)) { cand <- names(res)[sapply(res, function(x) is.matrix(x)||is.data.frame(x))]; cat("matrix-like slots:", paste(cand, collapse=", "), "\n"); W <- res[[cand[1]]] }
cat("estimated proportions dim:", paste(dim(W), collapse=" x "), "\n"); print(round(as.matrix(W)[, 1:min(4,ncol(W))], 3))
Wm <- as.matrix(W)
cat("W dimnames rows:", paste(head(rownames(Wm)), collapse=", "), "| cols:", paste(head(colnames(Wm)), collapse=", "), "\n")
# orient W so that rows = bulk samples, cols = cell types
if (all(rownames(true.prop) %in% rownames(Wm))) ep <- Wm[rownames(true.prop), , drop=FALSE] else
if (all(rownames(true.prop) %in% colnames(Wm))) ep <- t(Wm)[rownames(true.prop), , drop=FALSE] else ep <- NULL
if (!is.null(ep)) {
  ct <- intersect(colnames(true.prop), colnames(ep))
  if (length(ct) >= 2) {
    r <- cor(as.vector(true.prop[, ct]), as.vector(ep[, ct]), method="pearson")
    cat(sprintf("CLIMB vs true proportions: Pearson r = %.3f over %d values (%d cell types)\n", r, length(ct)*nrow(true.prop), length(ct)))
  } else cat("cell-type names:", paste(colnames(ep), collapse=", "), "\n")
} else cat("could not align sample names; W rows/cols shown above\n")

cat("\n######## 2. seAMLess on its bundled TCGA example ########\n")
suppressPackageStartupMessages({library(seAMLess); library(xbioc)})
data(exampleTCGA)
cat("exampleTCGA dim:", paste(dim(exampleTCGA), collapse=" x "), "\n")
print(exampleTCGA[1:3, 1:4])
t0 <- Sys.time()
out <- seAMLess(exampleTCGA, verbose = FALSE)
cat("seAMLess() finished in", round(as.numeric(difftime(Sys.time(), t0, units="mins")),2), "min\n")
cat("returned:", paste(class(out), collapse="/"), "| names:", paste(names(out), collapse=", "), "\n")
for (nm in names(out)) {
  o <- out[[nm]]
  cat("--", nm, ":", paste(class(o), collapse="/"), if (!is.null(dim(o))) paste(dim(o), collapse=" x ") else length(o), "\n")
  if (!is.null(dim(o))) print(round(as.matrix(o)[1:min(3,nrow(o)), 1:min(5,ncol(o))], 4))
  else print(utils::head(o, 5))
}
cat("\nSMOKE TESTS DONE\n")
