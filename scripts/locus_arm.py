"""Positive control for the locus-level reading: is peak-to-gene linking needed at all?

S21.2 showed that replacing a gene's correlated peaks with random peaks from the same
+/-250 kb window costs only ~0.019 AUROC and is not significant. If the signal really is a
property of the locus rather than of the linked peaks, then scoring a gene by ALL peaks in
its window -- no correlation, no linking, no r threshold -- should match arm A. If it does,
the method loses a whole step and the claim becomes much simpler.

Arms built here, all from the same eligible pool and the same 200-gene budget:
  locus_mean   mean cell-type accessibility specificity over every peak within +/-250 kb
  locus_sum    the same but summed, so it also carries peak density
  locus_top10  mean over the 10 most cell-type-specific peaks in the window
Compared against arm A (correlation-linked), arm R, and n_links only.
"""
import os, sys, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
C_GRID = [1e-2, 1e-1]; WINDOW = 250_000; TOP_M = 200

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
sp_ = np.load(f'{ROOT}/data/programs/specificity_bmmc.npz', allow_pickle=True)
pgenes = pr['genes'].astype(str); cts = list(pr['cell_types'].astype(str))
elig = pr['eligible']; peaks = pr['peaks'].astype(str)
link_g = pr['link_g']
S_peak = sp_['S_peak']; jH = cts.index('HSC')
W_A_hsc = np.array(pr['W_A'][:, jH]); W_R_hsc = np.array(pr['W_R'][:, jH])

ann = pd.read_csv(f'{ROOT}/data/ref/gencode_v38_genes_tss.tsv', sep='\t', header=None,
                  names=['chrom','start','end','strand','tss','ensg','symbol','gene_type'])
ann = ann.drop_duplicates('symbol').set_index('symbol')
tss = np.array([float(ann.tss[g]) if g in ann.index else np.nan for g in pgenes])
gchrom = np.array([ann.chrom[g] if g in ann.index else '' for g in pgenes])
parts = pd.Series(peaks).str.split('-', n=2, expand=True)
pchrom = parts[0].values
pmid = ((parts[1].astype(int) + parts[2].astype(int)) // 2).values
sp_h = S_peak[:, jH]

locus_mean = np.full(len(pgenes), -np.inf, dtype=np.float64)
locus_sum  = np.full(len(pgenes), -np.inf, dtype=np.float64)
locus_top  = np.full(len(pgenes), -np.inf, dtype=np.float64)
for c in np.unique(pchrom):
    idx = np.where(pchrom == c)[0]
    o = idx[np.argsort(pmid[idx])]; mids = pmid[o]; vals = sp_h[o]
    cs = np.concatenate([[0.0], np.cumsum(vals.astype(np.float64))])
    gi = np.where((gchrom == c) & np.isfinite(tss))[0]
    if len(gi) == 0: continue
    lo = np.searchsorted(mids, tss[gi] - WINDOW); hi = np.searchsorted(mids, tss[gi] + WINDOW)
    n = (hi - lo).astype(float)
    s = cs[hi] - cs[lo]
    ok = n > 0
    locus_sum[gi[ok]] = s[ok]
    locus_mean[gi[ok]] = s[ok] / n[ok]
    for k, (a, b) in enumerate(zip(lo, hi)):
        if b - a <= 0: continue
        v = vals[a:b]
        locus_top[gi[k]] = np.sort(v)[-10:].mean()
n_links = np.bincount(link_g, minlength=len(pgenes)).astype(float)
print(f'locus scores computed | genes with a window: {(np.isfinite(locus_mean)).sum()}')

expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr.drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
S = spec.set_index('dbgap_rnaseq_sample')
lab, coh = S.induction_response.to_dict(), S.cohort.to_dict()
yall = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
cw = np.array([str(coh.get(s)) for s in cols])
okm = yall >= 0; y = yall[okm]
tr, te = okm & (cw == 'Waves1+2'), okm & (cw == 'Waves3+4')
XF = expr.loc[:, cols].to_numpy(dtype=np.float64).T
EX = {g: i for i, g in enumerate(expr.index)}
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
             .split(np.zeros((okm.sum(), 1)), y))
def top_elig(score, k=TOP_M):
    s = np.where(elig, np.nan_to_num(score, nan=-np.inf, neginf=-np.inf), -np.inf)
    idx = np.argsort(-s)[:k]
    return np.array([EX[pgenes[i]] for i in idx if np.isfinite(s[i]) and pgenes[i] in EX])
def run(sel):
    Mw = XF[:, sel]; M = Mw[okm]; out = {}
    for C in C_GRID:
        cv = []
        for a, b in folds:
            mu, sd = M[a].mean(0), M[a].std(0) + 1e-8
            cv.append(roc_auc_score(y[b], LogisticRegression(C=C, max_iter=5000)
                      .fit((M[a]-mu)/sd, y[a]).decision_function((M[b]-mu)/sd)))
        mu, sd = Mw[tr].mean(0), Mw[tr].std(0) + 1e-8
        out[C] = (np.array(cv), roc_auc_score(yall[te], LogisticRegression(C=C, max_iter=5000)
                  .fit((Mw[tr]-mu)/sd, yall[tr]).decision_function((Mw[te]-mu)/sd)))
    return out
ARMS = {'arm A (correlation-linked)': top_elig(W_A_hsc),
        'arm R (expression)':         top_elig(W_R_hsc),
        'locus_sum (no linking)':     top_elig(locus_sum),
        'locus_mean (no linking)':    top_elig(locus_mean),
        'locus_top10 (no linking)':   top_elig(locus_top),
        'n_links only':               top_elig(n_links)}
res = {n: run(s) for n, s in ARMS.items()}
realA = set(ARMS['arm A (correlation-linked)'])
print(f'\n{"arm":30s}{"n":>4s}{"ovl A":>7s}' + ''.join(f'{f"CV C={C:g}":>12s}{"xwave":>9s}' for C in C_GRID))
for n, s in ARMS.items():
    print(f'  {n:28s}{len(s):4d}{len(realA & set(s)):7d}' +
          ''.join(f'{res[n][C][0].mean():12.3f}{res[n][C][1]:9.3f}' for C in C_GRID))
print('\narm A vs each linking-free arm (positive values mean linking helps):')
for k2 in ['locus_sum (no linking)', 'locus_mean (no linking)', 'locus_top10 (no linking)',
           'n_links only', 'arm R (expression)']:
    for C in C_GRID:
        d = res['arm A (correlation-linked)'][C][0] - res[k2][C][0]
        print(f'  C={C:g}  A vs {k2:28s} CV {d.mean():+.4f} p={wilcoxon(d).pvalue:9.3g} '
              f'| xwave {res["arm A (correlation-linked)"][C][1]-res[k2][C][1]:+.4f}')
print('\nbest linking-free arm vs arm R:')
for k1 in ['locus_sum (no linking)', 'locus_mean (no linking)', 'locus_top10 (no linking)']:
    for C in C_GRID:
        d = res[k1][C][0] - res['arm R (expression)'][C][0]
        print(f'  C={C:g}  {k1:28s} vs arm R  CV {d.mean():+.4f} p={wilcoxon(d).pvalue:9.3g} '
              f'| xwave {res[k1][C][1]-res["arm R (expression)"][C][1]:+.4f}')
json.dump({'cv': {n: {str(C): float(res[n][C][0].mean()) for C in C_GRID} for n in res},
           'xwave': {n: {str(C): float(res[n][C][1]) for C in C_GRID} for n in res},
           'overlap_with_armA': {n: len(realA & set(s)) for n, s in ARMS.items()}},
          open(f'{ROOT}/results/locus_arm.json','w'), indent=2)
print('\nwrote results/locus_arm.json')
