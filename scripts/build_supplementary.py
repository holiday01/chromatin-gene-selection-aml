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

# ---- S3/S4: hyperparameter and budget sweeps -------------------------------------------
S3, S4 = rd('sweep_programs.csv'), rd('budget_sweep.csv')
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

# ---- S6: nulls and linking-free arms ----------------------------------------------------
S6 = []
for src, f in (('permuted links (20 perms)', 'link_permutation.json'),
               ('permuted links, window (100 perms)', 'link_permutation_window100.json')):
    j = rj(f)
    if j:
        for k, v in j.items(): S6.append(dict(analysis=src, setting=k, **v))
j = rj('locus_arm.json')
if j:
    for arm, cv in j['cv'].items():
        S6.append(dict(analysis='linking-free locus arms', setting=arm,
                       overlap_with_armA=j['overlap_with_armA'][arm],
                       **{f'cv_C{k}': v for k, v in cv.items()},
                       **{f'xwave_C{k}': v for k, v in j['xwave'][arm].items()}))
S6 = pd.DataFrame(S6) if S6 else None

# ---- S7: enrichment ---------------------------------------------------------------------
S7a, S7b = rd('enrich_enrichr.csv'), rd('enrich_lsc_overlap.csv')
if S7a is not None: S7a = S7a[S7a.q < 0.25].sort_values(['gene_set','library','q'])

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
S9 = pd.DataFrame(S9) if S9 else None

# ---- S10: clinical covariates and ELN2017 ------------------------------------------------
S10 = rd('recomputed_remaining.csv')
if S10 is not None:
    S10 = S10[S10.analysis == 'clinical'].drop(columns='analysis')

# ---- S11: the cross-programme sweep ------------------------------------------------------
# All cells are cross-validated. Programmes share the cohort, the endpoint and the folds,
# so the aggregate tests are descriptive and the pairing is by cell identity alone.
S11a = rd('cross_reference_aggregate.csv')
S11b = rd('cross_reference_programs.csv')
if S11b is not None:
    S11b = S11b.rename(columns={
        'bmmc': 'healthy_programme', 'aml': 'malignant_programme',
        'b001': 'healthy_A_minus_R_C0.01', 'a001': 'malignant_A_minus_R_C0.01',
        'b01': 'healthy_A_minus_R_C0.1', 'a01': 'malignant_A_minus_R_C0.1',
        'mean4': 'mean_over_four_cv_cells', 'allpos': 'positive_in_all_four',
        'stem': 'myeloid_stem_progenitor'})

SHEETS = [('S1_gene_sets', S1), ('S2_per_program', S2), ('S3_hyperparameter_sweep', S3),
          ('S4_budget_sweep', S4), ('S5_connectivity_audit', S5), ('S6_null_models', S6),
          ('S7a_enrichment', S7a), ('S7b_LSC_overlap', S7b),
          ('S8a_Ng2016_transcribed', S8a), ('S8b_Ng2016_resolution', S8b),
          ('S9_overall_survival', S9), ('S10_clinical_models', S10),
          ('S11a_programme_sweep', S11a), ('S11b_matched_programmes', S11b)]
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
