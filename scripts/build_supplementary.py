"""Assemble the Supporting Information workbook for the Cancer Medicine submission.

Table S1 is the one artefact a reader can actually reuse: the gene sets themselves, for both
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

# ---- Table S1: the gene sets -----------------------------------------------------------
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
S1 = pd.DataFrame(rows)
print(f'Table S1: {len(S1)} rows | {S1.groupby(["reference","arm"]).size().to_dict()}')

def rd(p, **kw):
    try: return pd.read_csv(f'{ROOT}/results/{p}', **kw)
    except Exception as e: print(f'  skip {p}: {type(e).__name__}'); return None
def rj(p):
    try: return json.load(open(f'{ROOT}/results/{p}'))
    except Exception as e: print(f'  skip {p}: {type(e).__name__}'); return None

# ---- S2: per-program, all references ---------------------------------------------------
S2 = []
for label, tag in REFS.items():
    # eval_programs.py names its output after the npz, so the healthy reference writes
    # eval_programs_per_program.csv; an older run also left eval_programs_bmmc_*.csv behind
    t2 = tag
    d = rd(f'eval_{t2}_per_program.csv')
    if d is not None: d.insert(0, 'reference', label); S2.append(d)
S2 = pd.concat(S2, ignore_index=True) if S2 else None
if S2 is not None:
    # the per-fold Wilcoxon columns are the statistic the paper abandoned; keeping them
    # here would put two inference methods in the same submission
    S2 = S2.drop(columns=[c for c in S2.columns if c.startswith('p_')])
    # eval_programs.py stores the PAIRED DIFFERENCE arm A minus arm R, not either arm's
    # own AUROC. Name the columns accordingly so the sheet cannot be read as performance.
    S2 = S2.rename(columns={'cv_0.01': 'A_minus_R_cv_C0.01', 'cv_0.1': 'A_minus_R_cv_C0.1',
                            'xw_0.01': 'A_minus_R_xwave_C0.01',
                            'xw_0.1': 'A_minus_R_xwave_C0.1'})

# ---- S3/S4: hyperparameter and budget sweeps -------------------------------------------
S3, S4 = rd('sweep_programs.csv'), rd('budget_sweep.csv')
# results/ keeps the column names the producing scripts wrote, because
# check_reported_numbers.py asserts against them. The workbook is the reader-facing
# artefact, so every column is renamed here to say what it holds.
if S4 is not None:
    S4 = S4[S4.reference != 'reference'].rename(columns={
        'A_cv_0.01': 'A_cv_AUROC_C0.01', 'R_cv_0.01': 'R_cv_AUROC_C0.01',
        'A_cv_0.1': 'A_cv_AUROC_C0.1', 'R_cv_0.1': 'R_cv_AUROC_C0.1',
        'd_cv_0.01': 'A_minus_R_cv_C0.01', 'd_xw_0.01': 'A_minus_R_xwave_C0.01',
        'd_cv_0.1': 'A_minus_R_cv_C0.1', 'd_xw_0.1': 'A_minus_R_xwave_C0.1'})
if S3 is not None:
    S3 = S3.rename(columns={c: c.replace('A_C', 'armA_cv_AUROC_C')
                                 .replace('R_C', 'armR_cv_AUROC_C')
                                 .replace('M_C', 'variance_matched_cv_AUROC_C')
                            for c in S3.columns if c[:2] in ('A_', 'R_', 'M_')})
# both sweeps were written with the uncorrected per-fold Wilcoxon the paper no longer uses;
# keep the effect sizes, drop the P values rather than show two inference methods
for _d in (S3, S4):
    if _d is not None:
        _d.drop(columns=[c for c in _d.columns if c.startswith('p_') or c == 'p'],
                inplace=True, errors='ignore')

# ---- S5: connectivity audit -------------------------------------------------------------
S5 = []
for label, tag in REFS.items():
    j = rj(f'confound_{tag}.json')
    if not j: continue
    for st, prof in j.get('profile', {}).items():
        S5.append(dict(reference=label, measure='profile_median', arm=st, **prof))
    for arm, cv in j.get('cv', {}).items():
        r = dict(reference=label, measure='AUROC', arm=arm)
        r.update({f'cv_C{k}': v for k, v in cv.items()})
        r.update({f'xwave_C{k}': v for k, v in j['xwave'][arm].items()})
        S5.append(r)
S5 = pd.DataFrame(S5) if S5 else None
if S5 is not None:
    S5 = S5.rename(columns={c: c.replace('cv_C', 'cv_AUROC_C').replace('xwave_C', 'xwave_AUROC_C')
                            for c in S5.columns if c.startswith(('cv_C', 'xwave_C'))})

# ---- S6a: permuted-link nulls -----------------------------------------------------------
# 'beats' in the source JSON counts permutations where the mean difference is positive AND
# an uncorrected per-fold Wilcoxon reaches P < 0.05. That is not the corrected resampled
# t-test this paper uses, and shipping it would put two inference methods in one
# submission, so it is dropped here as it is in S2, S3 and S4.
ovl = rd('permutation_overlap.csv')
ovl = ovl.set_index('null') if ovl is not None else None
S6a = []
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
        S6a.append(r)
S6a = pd.DataFrame(S6a) if S6a else None

# ---- S6b: linking-free locus arms -------------------------------------------------------
S6b = []
j = rj('locus_arm.json')
if j:
    for arm, cv in j['cv'].items():
        S6b.append(dict(arm=arm, gene_overlap_with_armA=j['overlap_with_armA'][arm],
                        **{f'cv_AUROC_C{k}': v for k, v in cv.items()},
                        **{f'xwave_AUROC_C{k}': v for k, v in j['xwave'][arm].items()}))
S6b = pd.DataFrame(S6b) if S6b else None

# ---- S7: enrichment ---------------------------------------------------------------------
S7a, S7b = rd('enrich_enrichr.csv'), rd('enrich_lsc_overlap.csv')
if S7a is not None:
    S7a = S7a[S7a.q < 0.25].sort_values(['gene_set', 'library', 'q']).rename(columns={
        'p': 'enrichr_p', 'q': 'enrichr_q_BH', 'odds': 'odds_ratio',
        'combined': 'enrichr_combined_score'})
if S7b is not None:
    S7b = S7b.rename(columns={'p': 'fisher_one_sided_p', 'odds': 'odds_ratio',
                              'n': 'n_in_gene_set', 'overlap': 'n_overlap',
                              'expected': 'n_overlap_expected'})

# ---- S8: the LSC104 transcription and its resolution ------------------------------------
S8a = pd.read_csv(f'{ROOT}/data/ref/published_signatures/ng2016_table1_transcribed.tsv', sep='\t')
S8b = pd.read_csv(f'{ROOT}/data/ref/published_signatures/ng2016_resolution_map.csv')

# ---- S9: overall survival ----------------------------------------------------------------
S9 = []
for label, tag in REFS.items():
    f = 'os_programs.json' if tag == 'programs' else f'os_{tag}.json'
    j = rj(f)
    if not j: continue
    for arm, cv in j['cv'].items():
        r = dict(reference=label, arm=arm, endpoint='overall survival',
                 n=j['n'], events=j['events'])
        r.update({f'cv_alpha{k}': v for k, v in cv.items()})
        r.update({f'xwave_alpha{k}': v for k, v in j['xwave'][arm].items()})
        S9.append(r)
S9a = pd.DataFrame(S9) if S9 else None
if S9a is not None:
    S9a = S9a.rename(columns={c: c.replace('cv_alpha', 'cv_cindex_alpha')
                                 .replace('xwave_alpha', 'xwave_cindex_alpha')
                              for c in S9a.columns if 'alpha' in c})

# ---- S9b: the four-penalty HSC comparison the main text reports -------------------------
# eval_os.py evaluates every programme at two penalties; the paired comparison quoted in
# the text comes from recompute_remaining.py at four penalties and was not in this workbook.
S9b = rd('recomputed_remaining.csv')
if S9b is not None:
    S9b = S9b[S9b.analysis == 'overall_survival'].drop(columns='analysis').rename(
        columns={'C': 'alpha', 'diff': 'A_minus_R_cindex', 'nb_t': 'corrected_t',
                 'nb_p': 'corrected_p'})
    S9b = S9b[['reference', 'alpha', 'A_minus_R_cindex', 'corrected_t', 'corrected_p']]

# ---- S10: clinical covariates and ELN2017 ------------------------------------------------
S10 = rd('recomputed_remaining.csv')
if S10 is not None:
    S10 = S10[S10.analysis == 'clinical'].drop(columns=['analysis', 'reference'])
    # the source 'diff' column carries an absolute AUROC on the rows that name one model
    # and a paired increment on the rows that name a contrast; split them so neither can
    # be read as the other
    is_delta = S10.comparison.str.contains(' - ')
    S10.insert(1, 'quantity', np.where(is_delta, 'delta_AUROC', 'AUROC'))
    S10.insert(2, 'AUROC', np.where(is_delta, np.nan, S10['diff']))
    S10.insert(3, 'delta_AUROC', np.where(is_delta, S10['diff'], np.nan))
    S10 = S10.drop(columns='diff').rename(columns={'nb_t': 'corrected_t',
                                                   'nb_p': 'corrected_p'})

# ---- S11: the cross-programme sweep ------------------------------------------------------
# All cells are cross-validated. Programmes share the cohort, the endpoint and the folds,
# so the aggregate tests are descriptive and the pairing is by cell identity alone.
S11a = rd('cross_reference_aggregate.csv')
if S11a is not None:
    S11a = S11a.rename(columns={
        'cell': 'reference_and_penalty', 'n': 'n_programmes',
        'positive': 'n_programmes_favouring_armA', 'mean': 'mean_A_minus_R',
        'median': 'median_A_minus_R', 'sign_p': 'sign_test_p_over_programmes',
        'wilcoxon_p': 'wilcoxon_signedrank_p_over_programmes'})
S11b = rd('cross_reference_programs.csv')
if S11b is not None:
    S11b = S11b.rename(columns={
        'bmmc': 'healthy_programme', 'aml': 'malignant_programme',
        'b001': 'healthy_A_minus_R_C0.01', 'a001': 'malignant_A_minus_R_C0.01',
        'b01': 'healthy_A_minus_R_C0.1', 'a01': 'malignant_A_minus_R_C0.1',
        'mean4': 'mean_over_four_cv_cells', 'allpos': 'positive_in_all_four',
        'stem': 'myeloid_stem_progenitor'})

# ---- S12: the inclusion flow and what the drug-assay criterion removed -------------------
S12a = rd('cohort_flow.csv')
S12b = rd('cohort_flow_excluded.csv')

SHEETS = [('S1_gene_sets', S1), ('S2_per_program', S2), ('S3_hyperparameter_sweep', S3),
          ('S4_budget_sweep', S4), ('S5_connectivity_audit', S5),
          ('S6a_permuted_link_nulls', S6a), ('S6b_linking_free_arms', S6b),
          ('S7a_enrichment', S7a), ('S7b_LSC_overlap', S7b),
          ('S8a_Ng2016_transcribed', S8a), ('S8b_Ng2016_resolution', S8b),
          ('S9a_survival_all_programmes', S9a), ('S9b_survival_HSC_four_alphas', S9b),
          ('S10_clinical_models', S10),
          ('S11a_programme_sweep', S11a), ('S11b_matched_programmes', S11b),
          ('S12a_inclusion_flow', S12a), ('S12b_cohort_composition', S12b)]
path = f'{OUT}/Supporting_Information_Tables.xlsx'
with pd.ExcelWriter(path, engine='openpyxl') as w:
    for name, df in SHEETS:
        if df is None or len(df) == 0:
            print(f'  MISSING {name}'); continue
        df.to_excel(w, sheet_name=name[:31], index=False)
        print(f'  {name:28s} {len(df):6d} rows x {df.shape[1]} cols')
S1.to_csv(f'{OUT}/TableS1_gene_sets.csv', index=False)
print(f'\nwrote {path}')
print(f'wrote {OUT}/TableS1_gene_sets.csv')
