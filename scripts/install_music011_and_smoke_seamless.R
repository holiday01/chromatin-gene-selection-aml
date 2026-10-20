options(Ncpus = 12, repos = c(CRAN = "https://cloud.r-project.org"), timeout = 1800)
lib011 <- "/home/holiday01/R/lib_music011"
dir.create(lib011, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(file.path(lib011, "MuSiC", "DESCRIPTION"))) {
  cat(">>> installing MuSiC at commit 60c08c7 (Version 0.1.1, the version seAMLess's paper used) into", lib011, "\n")
  remotes::install_github("xuranw/MuSiC", ref = "60c08c7", lib = lib011,
                          upgrade = "never", dependencies = FALSE, force = TRUE)
}
.libPaths(c(lib011, .libPaths()))
cat("MuSiC version now visible:", as.character(packageVersion("MuSiC")), "from",
    dirname(dirname(system.file(package = "MuSiC"))), "\n")
cat("music_prop signature:\n"); print(args(MuSiC::music_prop))

cat("\n######## seAMLess on its bundled TCGA example (MuSiC 0.1.1) ########\n")
suppressPackageStartupMessages({library(seAMLess); library(xbioc); library(Biobase)})
data(exampleTCGA)
cat("exampleTCGA dim:", paste(dim(exampleTCGA), collapse = " x "), "\n")
t0 <- Sys.time()
out <- seAMLess(exampleTCGA, verbose = FALSE)
cat("seAMLess() finished in", round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 2), "min\n")
cat("returned:", paste(class(out), collapse = "/"), "| names:", paste(names(out), collapse = ", "), "\n")
for (nm in names(out)) {
  o <- out[[nm]]
  cat("--", nm, ":", paste(class(o), collapse = "/"),
      if (!is.null(dim(o))) paste(dim(o), collapse = " x ") else length(o), "\n")
  if (!is.null(dim(o))) print(round(as.matrix(o)[1:min(3, nrow(o)), 1:min(6, ncol(o))], 4)) else print(utils::head(o, 6))
}
cat("\nSEAMLESS SMOKE TEST DONE\n")
