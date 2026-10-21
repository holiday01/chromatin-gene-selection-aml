"""Assemble the Supporting Information workbook for the Cancer Medicine submission.

Table S7 is the one artefact a reader can actually reuse: the gene sets themselves, for both
arms, for the HSC program, in all three reference constructions, annotated with the score
that selected each gene, its connectivity, and whether it is in the published LSC sets.
Everything else is the analysis the main text had to compress.
"""
import os, json, numpy as np, pandas as pd
ROOT = '/home/holiday01/2026_cm'
OUT  = f'{ROOT}/manuscript/supplementary'
os.makedirs(OUT, exist_ok=True)
REFS = {'BMMC (healthy marrow)': 'programs',  # note: eval/OS outputs for this one are tagged 'programs_bmmc' / 'programs'
        'AML malignant': 'programs_aml_malignant_v3',
        'AML all cells': 'programs_aml_all_v3'}
LSC104 = {l.strip() for l in open(f'{ROOT}/data/ref/published_signatures/ng2016_lsc104_resolved.txt') if l.strip()}
LSC17  = {l.strip() for l in open(f'{ROOT}/data/ref/published_signatures/ng2016_lsc17.txt') if l.strip()}
ALIAS  = {'LOC284422': 'C19orf77'}
LSC17  = {ALIAS.get(g, g) for g in LSC17}

# ---- Table S7: the gene sets -----------------------------------------------------------
rows = []
for label, tag in REFS.items():
    pr = np.load(f'{ROOT}/data/programs/{tag}.npz', allow_pickle=True)
    g = pr['genes'].astype(str); cts = list(pr['cell_types'].astype(str))
    jH = cts.index('HSC')
    nlink = np.bincount(pr['link_g'], minlength=len(g)).astype(int)
    for arm, W in (('A (linked-peak accessibility)', pr['W_A']),
                   ('R (expression specificity)',    pr['W_R'])):
        s = W[:, jH]
        idx = np.flatnonzero(s)
        idx = idx[np.argsort(-s[idx])]
        for rank, i in enumerate(idx, 1):
            rows.append(dict(reference=label, arm=arm, rank=rank, gene=g[i],
                             selection_score=float(s[i]), n_linked_peaks=int(nlink[i]),
                             in_Ng2016_LSC104=g[i] in LSC104, in_Ng2016_LSC17=g[i] in LSC17))
S7 = pd.DataFrame(rows)
print(f'Table S7: {len(S7)} rows | {S7.groupby(["reference","arm"]).size().to_dict()}')

def rd(p, **kw):
    try: return pd.read_csv(f'{ROOT}/results/{p}', **kw)
    except Exception as e: print(f'  skip {p}: {type(e).__name__}'); return None
def rj(p):
    try: return json.load(open(f'{ROOT}/results/{p}'))
    except Exception as e: print(f'  skip {p}: {type(e).__name__}'); return None

# ---- S9: per-programme, all references ---------------------------------------------------
S9 = []
for label, tag in REFS.items():
    # eval_programs.py names its output after the npz, so the healthy reference writes
    # eval_programs_per_program.csv; an older run also left eval_programs_bmmc_*.csv behind
    t2 = tag
    d = rd(f'eval_{t2}_per_program.csv')
    if d is not None: d.insert(0, 'reference', label); S9.append(d)
S9 = pd.concat(S9, ignore_index=True) if S9 else None
if S9 is not None:
    # the per-fold Wilcoxon columns are the statistic the paper abandoned; keeping them
    # here would put two inference methods in the same submission
    S9 = S9.drop(columns=[c for c in S9.columns if c.startswith('p_')])
    # eval_programs.py stores the PAIRED DIFFERENCE arm A minus arm R, not either arm's
    # own AUROC. Name the columns accordingly so the sheet cannot be read as performance.
    S9 = S9.rename(columns={'cv_0.01': 'A_minus_R_cv_C0.01', 'cv_0.1': 'A_minus_R_cv_C0.1',
                            'xw_0.01': 'A_minus_R_xwave_C0.01',
                            'xw_0.1': 'A_minus_R_xwave_C0.1'})

# ---- S4/S10: hyperparameter and budget sweeps -------------------------------------------
S4, S10 = rd('sweep_programs.csv'), rd('budget_sweep.csv')
# results/ keeps the column names the producing scripts wrote, because
# check_reported_numbers.py asserts against them. The workbook is the reader-facing
# artefact, so every column is renamed here to say what it holds.
if S10 is not None:
    S10 = S10[S10.reference != 'reference'].rename(columns={
        'A_cv_0.01': 'A_cv_AUROC_C0.01', 'R_cv_0.01': 'R_cv_AUROC_C0.01',
        'A_cv_0.1': 'A_cv_AUROC_C0.1', 'R_cv_0.1': 'R_cv_AUROC_C0.1',
        'd_cv_0.01': 'A_minus_R_cv_C0.01', 'd_xw_0.01': 'A_minus_R_xwave_C0.01',
        'd_cv_0.1': 'A_minus_R_cv_C0.1', 'd_xw_0.1': 'A_minus_R_xwave_C0.1'})
if S4 is not None:
    S4 = S4.rename(columns={c: c.replace('A_C', 'armA_cv_AUROC_C')
                                 .replace('R_C', 'armR_cv_AUROC_C')
                                 .replace('M_C', 'variance_matched_cv_AUROC_C')
                            for c in S4.columns if c[:2] in ('A_', 'R_', 'M_')})
# both sweeps were written with the uncorrected per-fold Wilcoxon the paper no longer uses;
# keep the effect sizes, drop the P values rather than show two inference methods
for _d in (S4, S10):
    if _d is not None:
        _d.drop(columns=[c for c in _d.columns if c.startswith('p_') or c == 'p'],
                inplace=True, errors='ignore')

# ---- S12: connectivity audit -------------------------------------------------------------
S12 = []
for label, tag in REFS.items():
    j = rj(f'confound_{tag}.json')
    if not j: continue
    for st, prof in j.get('profile', {}).items():
        S12.append(dict(reference=label, measure='profile_median', arm=st, **prof))
    for arm, cv in j.get('cv', {}).items():
        r = dict(reference=label, measure='AUROC', arm=arm)
        r.update({f'cv_C{k}': v for k, v in cv.items()})
        r.update({f'xwave_C{k}': v for k, v in j['xwave'][arm].items()})
        S12.append(r)
S12 = pd.DataFrame(S12) if S12 else None
if S12 is not None:
    S12 = S12.rename(columns={c: c.replace('cv_C', 'cv_AUROC_C').replace('xwave_C', 'xwave_AUROC_C')
                            for c in S12.columns if c.startswith(('cv_C', 'xwave_C'))})

# ---- S13: permuted-link nulls -----------------------------------------------------------
# 'beats' in the source JSON counts permutations where the mean difference is positive AND
# an uncorrected per-fold Wilcoxon reaches P < 0.05. That is not the corrected resampled
# t-test this paper uses, and shipping it would put two inference methods in one
# submission, so it is dropped here as it is in S9, S4 and S10.
ovl = rd('permutation_overlap.csv')
ovl = ovl.set_index('null') if ovl is not None else None
S13 = []
for src, f, key in (('permuted links, genome-wide pool', 'link_permutation.json', 'global'),
                    ('permuted links, within window', 'link_permutation_window100.json', 'window')):
    j = rj(f)
    if not j: continue
    for k, v in j.items():
        if not k.startswith(key):  # the 20-perm file also carries an early window run
            continue
        r = dict(null=src, setting=k, n_permutations=v['n'], real_cv_AUROC=v['real_cv'],
                 null_cv_AUROC_mean=v['null_cv_mean'], null_cv_AUROC_sd=v['null_cv_sd'],
                 empirical_p_cv=v['emp_p'], real_xwave_AUROC=v['real_xw'],
                 null_xwave_AUROC_mean=v['null_xw_mean'],
                 n_perms_real_higher_xwave=v['above'])
        if ovl is not None and key in ovl.index:
            r.update(gene_overlap_median=ovl.loc[key, 'overlap_median'],
                     gene_overlap_min=ovl.loc[key, 'overlap_min'],
                     gene_overlap_max=ovl.loc[key, 'overlap_max'],
                     distinct_peaks_real=ovl.loc[key, 'distinct_peaks_real'],
                     distinct_peaks_permuted_median=ovl.loc[key, 'distinct_peaks_median'])
        S13.append(r)
S13 = pd.DataFrame(S13) if S13 else None

# ---- S14: linking-free locus arms -------------------------------------------------------
S14 = []
j = rj('locus_arm.json')
if j:
    for arm, cv in j['cv'].items():
        S14.append(dict(arm=arm, gene_overlap_with_armA=j['overlap_with_armA'][arm],
                        **{f'cv_AUROC_C{k}': v for k, v in cv.items()},
                        **{f'xwave_AUROC_C{k}': v for k, v in j['xwave'][arm].items()}))
S14 = pd.DataFrame(S14) if S14 else None

# ---- S15/S16: LSC overlap and enrichment ---------------------------------------------------------------------
S16, S15 = rd('enrich_enrichr.csv'), rd('enrich_lsc_overlap.csv')
if S16 is not None:
    S16 = S16[S16.q < 0.25].sort_values(['gene_set', 'library', 'q']).rename(columns={
        'p': 'enrichr_p', 'q': 'enrichr_q_BH', 'odds': 'odds_ratio',
        'combined': 'enrichr_combined_score'})
if S15 is not None:
    S15 = S15.rename(columns={'p': 'fisher_one_sided_p', 'odds': 'odds_ratio',
                              'n': 'n_in_gene_set', 'overlap': 'n_overlap',
                              'expected': 'n_overlap_expected'})

# ---- S5/S6: the LSC104 transcription and its resolution ------------------------------------
S5 = pd.read_csv(f'{ROOT}/data/ref/published_signatures/ng2016_table1_transcribed.tsv', sep='\t')
S6 = pd.read_csv(f'{ROOT}/data/ref/published_signatures/ng2016_resolution_map.csv')

# ---- overall survival ----------------------------------------------------------------
_surv = []
for label, tag in REFS.items():
    f = 'os_programs.json' if tag == 'programs' else f'os_{tag}.json'
    j = rj(f)
    if not j: continue
    for arm, cv in j['cv'].items():
        r = dict(reference=label, arm=arm, endpoint='overall survival',
                 n=j['n'], events=j['events'])
        r.update({f'cv_alpha{k}': v for k, v in cv.items()})
        r.update({f'xwave_alpha{k}': v for k, v in j['xwave'][arm].items()})
        _surv.append(r)
S17 = pd.DataFrame(_surv) if _surv else None
if S17 is not None:
    S17 = S17.rename(columns={c: c.replace('cv_alpha', 'cv_cindex_alpha')
                                 .replace('xwave_alpha', 'xwave_cindex_alpha')
                              for c in S17.columns if 'alpha' in c})

# ---- S18: the four-penalty HSC comparison the main text reports -------------------------
# eval_os.py evaluates every programme at two penalties; the paired comparison quoted in
# the text comes from recompute_remaining.py at four penalties and was not in this workbook.
S18 = rd('recomputed_remaining.csv')
if S18 is not None:
    S18 = S18[S18.analysis == 'overall_survival'].drop(columns='analysis').rename(
        columns={'C': 'alpha', 'diff': 'A_minus_R_cindex', 'nb_t': 'corrected_t',
                 'nb_p': 'corrected_p'})
    S18 = S18[['reference', 'alpha', 'A_minus_R_cindex', 'corrected_t', 'corrected_p']]

# ---- S11: clinical covariates and ELN2017 ------------------------------------------------
S11 = rd('recomputed_remaining.csv')
if S11 is not None:
    S11 = S11[S11.analysis == 'clinical'].drop(columns=['analysis', 'reference'])
    # the source 'diff' column carries an absolute AUROC on the rows that name one model
    # and a paired increment on the rows that name a contrast; split them so neither can
    # be read as the other
    is_delta = S11.comparison.str.contains(' - ')
    S11.insert(1, 'quantity', np.where(is_delta, 'delta_AUROC', 'AUROC'))
    S11.insert(2, 'AUROC', np.where(is_delta, np.nan, S11['diff']))
    S11.insert(3, 'delta_AUROC', np.where(is_delta, S11['diff'], np.nan))
    S11 = S11.drop(columns='diff').rename(columns={'nb_t': 'corrected_t',
                                                   'nb_p': 'corrected_p'})

# ---- S3/S8: the cross-programme sweep ------------------------------------------------------
# All cells are cross-validated. Programmes share the cohort, the endpoint and the folds,
# so the aggregate tests are descriptive and the pairing is by cell identity alone.
S8 = rd('cross_reference_aggregate.csv')
if S8 is not None:
    S8 = S8.rename(columns={
        'cell': 'reference_and_penalty', 'n': 'n_programmes',
        'positive': 'n_programmes_favouring_armA', 'mean': 'mean_A_minus_R',
        'median': 'median_A_minus_R', 'sign_p': 'sign_test_p_over_programmes',
        'wilcoxon_p': 'wilcoxon_signedrank_p_over_programmes'})
S3 = rd('cross_reference_programs.csv')
if S3 is not None:
    S3 = S3.rename(columns={
        'bmmc': 'healthy_programme', 'aml': 'malignant_programme',
        'b001': 'healthy_A_minus_R_C0.01', 'a001': 'malignant_A_minus_R_C0.01',
        'b01': 'healthy_A_minus_R_C0.1', 'a01': 'malignant_A_minus_R_C0.1',
        'mean4': 'mean_over_four_cv_cells', 'allpos': 'positive_in_all_four',
        'stem': 'myeloid_stem_progenitor'})

# ---- S1/S2: the inclusion flow and what the drug-assay criterion removed -------------------
S1 = rd('cohort_flow.csv')
S2 = rd('cohort_flow_excluded.csv')

SHEETS = [('S1_inclusion_flow', S1), ('S2_cohort_composition', S2),
          ('S3_matched_programmes', S3), ('S4_hyperparameter_sweep', S4),
          ('S5_Ng2016_transcribed', S5), ('S6_Ng2016_resolution', S6),
          ('S7_gene_sets', S7), ('S8_programme_sweep', S8),
          ('S9_per_programme', S9), ('S10_budget_sweep', S10),
          ('S11_clinical_models', S11), ('S12_connectivity_audit', S12),
          ('S13_permuted_link_nulls', S13), ('S14_linking_free_arms', S14),
          ('S15_LSC_overlap', S15), ('S16_enrichment', S16),
          ('S17_survival_all_programmes', S17), ('S18_survival_HSC_four_alphas', S18)]
path = f'{OUT}/Supporting_Information_Tables.xlsx'
with pd.ExcelWriter(path, engine='openpyxl') as w:
    for name, df in SHEETS:
        if df is None or len(df) == 0:
            print(f'  MISSING {name}'); continue
        df.to_excel(w, sheet_name=name[:31], index=False)
        print(f'  {name:28s} {len(df):6d} rows x {df.shape[1]} cols')
S7.to_csv(f'{OUT}/TableS7_gene_sets.csv', index=False)
print(f'\nwrote {path}')
print(f'wrote {OUT}/TableS7_gene_sets.csv')
