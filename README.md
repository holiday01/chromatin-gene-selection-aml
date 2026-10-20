# Gene selection by linked-peak accessibility in acute myeloid leukaemia

Analysis code for a controlled comparison of two gene-selection rules for predicting
response to induction chemotherapy in acute myeloid leukaemia. Within a fixed cell-type
programme, a fixed eligible gene pool and a fixed gene budget, genes are selected either by
the accessibility of their linked regulatory peaks (arm A) or by the gene's own expression
specificity (arm R), so that two gene sets of the same size differ only in the rule that
produced them.

Chiu Y-J. *Gene selection by linked-peak accessibility improves prediction of induction
response in acute myeloid leukaemia.* Submitted.

## What is here

| Path | Contents |
|---|---|
| `scripts/` | Analysis code, one script per result |
| `results/` | Derived summary statistics behind every figure and table |
| `data/ref/` | GENCODE v38 TSS coordinates; the Ng et al. leukaemic stem cell gene list and its resolution to current gene identifiers |

## What is not here, and why

**Patient-level data.** Beat AML (dbGaP accession `phs001657`) is controlled access under a
Data Use Certification. Expression and clinical tables are not redistributable and are not
included. Qualified investigators can obtain them through an approved dbGaP request.

**Single-cell references.** Open but large. Download from GEO:

- `GSE194122` — healthy bone-marrow multiome (Luecken et al., NeurIPS 2021)
- `GSE235308` — paediatric AML scATAC (Lambo et al., *Cancer Cell* 2023)
- `GSE235063` — paediatric AML scRNA (same study)

`scripts/fetch_lambo_aml.sh` downloads the two AML series selectively, skipping fragment and
raw matrix files. Note that GEO per-sample URLs live under
`samples/<GSM prefix>nnn/<GSM>/suppl/`; the series `suppl/` path returns 404 and, without
`curl --fail`, silently writes HTML error pages that look like completed downloads.

**Manuscript, figures and tables.** Not part of this repository.

## Expected layout

Scripts read from paths set at the top of each file. The defaults assume:

```
data/raw/beataml2/          Beat AML waves 1-4 tables (dbGaP)
data/interim/               cohort and expression tables written by prep_beataml_dose.py
data/programs/              cell-type programmes written by the build scripts
data/ref/                   included here
```

## Order to run

1. `prep_beataml_dose.py` — cohort, expression and drug-response tables
2. `build_programs.py` — programmes from the healthy multiome, both arms
3. `build_programs_aml.py <atac_dir> <rna_dir> <out.npz> {malignant|all}` — the two AML
   reference constructions
4. `export_specificity.py` — peak and gene specificity matrices, needed by the permutation
   and locus analyses
5. `eval_programs.py <programs.npz> <label>` — the main comparison for one reference
6. `robustness_check.py` — four penalties, count-normalised score variants, Nadeau-Bengio
   inference and patient-level bootstrap
7. Controls: `budget_sweep.py`, `baseline_lsc17.py`, `baseline_lsc104.py`,
   `confound_audit.py`, `link_permutation.py`, `locus_arm.py`, `distributed_test.py`,
   `state_vs_abundance.py`, `clinical_adjust.py`, `eval_os.py`
8. `recompute_pvalues.py`, `recompute_remaining.py` — all paired comparisons on one
   inferential footing
9. `make_tables.py`, `make_figures.py`, `build_supplementary.py` — outputs

## Inference

Repeated stratified cross-validation reuses every patient across repeats, so paired tests
over the individual folds are anticonservative. All comparisons between gene sets are
reported as per-repeat mean differences (n = 20) with the Nadeau-Bengio variance correction,
and the held-out wave comparison additionally as a patient-level bootstrap. Counts of
reference x penalty x split cells favouring one arm are descriptive tallies; no sign test is
computed across them.

## Environment

Python 3.11.15 with numpy 2.4.3, scipy 1.17.1, pandas 2.3.3, scikit-learn 1.8.0,
anndata 0.12.10, scikit-survival 0.27.0, matplotlib and openpyxl. Run scripts with
`python -I`. Set BLAS threads to 1 per worker; several scripts use a process pool.

## Funding

National Science and Technology Council, Taiwan (NSTC 113-2221-E-182-066-MY3).

## Contact

Yen-Jung Chiu, Department of Biomedical Engineering, Chang Gung University, Taoyuan, Taiwan.
d000020163@cgu.edu.tw
