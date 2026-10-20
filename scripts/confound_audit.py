"""Bias audit for arm A: is the chromatin score a proxy for gene connectivity?

S_chrom[g,c] = sum over a gene's linked peaks of r * S_peak[peak,c]  -- it is a SUM, so a
gene with more linked peaks scores higher by construction. If arm A simply selects
well-connected genes (long genes, genes in peak-dense or broadly open loci), the result is an
artefact. S8 controlled expression variance; connectivity was never controlled.

Three tests:
  1. describe arm A vs arm R vs background on n_links, gene span, local peak density, mean r
  2. arms built from connectivity ALONE, with no cell-type accessibility information at all
  3. link-count-matched random draws -- the connectivity analogue of the variance-matched control

Usage: confound_audit.py [programs.npz] [label]
"""
import os, sys, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, mannwhitneyu, spearmanr
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
NPZ   = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/data/programs/programs.npz'
LABEL = sys.argv[2] if len(sys.argv) > 2 else 'BMMC'
C_GRID = [1e-2, 1e-1]
WINDOW = 250_000
NRAND = 20

pr = np.load(NPZ, allow_pickle=True)
W_A, W_R = pr['W_A'], pr['W_R']
pgenes = pr['genes'].astype(str); cts = list(pr['cell_types'].astype(str))
elig = pr['eligible']; link_g, link_p, link_r = pr['link_g'], pr['link_p'], pr['link_r']
peaks = pr['peaks'].astype(str)
jH = cts.index('HSC')

# ---- per-gene connectivity features -------------------------------------------------
n_links = np.bincount(link_g, minlength=len(pgenes)).astype(float)
sum_r   = np.bincount(link_g, weights=link_r.astype(float), minlength=len(pgenes))
mean_r  = np.divide(sum_r, n_links, out=np.zeros_like(sum_r), where=n_links > 0)

ann = pd.read_csv(f'{ROOT}/data/ref/gencode_v38_genes_tss.tsv', sep='\t', header=None,
                  names=['chrom','start','end','strand','tss','ensg','symbol','gene_type'])
ann = ann.drop_duplicates('symbol').set_index('symbol')
span = np.array([float(ann.end[g] - ann.start[g]) if g in ann.index else np.nan for g in pgenes])
tss  = np.array([float(ann.tss[g])   if g in ann.index else np.nan for g in pgenes])
chrom= np.array([ann.chrom[g]        if g in ann.index else ''     for g in pgenes])

parts = pd.Series(peaks).str.split('-', n=2, expand=True)
pk = pd.DataFrame({'chrom': parts[0].values,
                   'mid': ((parts[1].astype(int) + parts[2].astype(int)) // 2).values})
dens = np.full(len(pgenes), np.nan)
for c, sub in pk.groupby('chrom'):
    mids = np.sort(sub.mid.values)
    idx = np.where((chrom == c) & np.isfinite(tss))[0]
    if len(idx) == 0: continue
    lo = np.searchsorted(mids, tss[idx] - WINDOW); hi = np.searchsorted(mids, tss[idx] + WINDOW)
    dens[idx] = hi - lo

hscA, hscR = np.flatnonzero(W_A[:, jH]), np.flatnonzero(W_R[:, jH])
bg = np.flatnonzero(elig)
F = pd.DataFrame({'n_links': n_links, 'mean_r': mean_r, 'span_kb': span/1e3, 'peak_density': dens})
print(f'=== {LABEL} : connectivity profile of the HSC 200-gene arms ===')
rows = []
for nm, sel in (('arm A (chromatin)', hscA), ('arm R (RNA)', hscR), ('eligible background', bg)):
    r = {'set': nm, 'n': len(sel)}
    for c in F.columns:
        v = F[c].values[sel]; r[c] = np.nanmedian(v)
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.1f}'))
print('\nMann-Whitney, arm A vs arm R (and vs background):')
for c in F.columns:
    a, rr, b = F[c].values[hscA], F[c].values[hscR], F[c].values[bg]
    a, rr, b = a[np.isfinite(a)], rr[np.isfinite(rr)], b[np.isfinite(b)]
    print(f'  {c:14s} A vs R p={mannwhitneyu(a, rr).pvalue:9.3g} | '
          f'A vs background p={mannwhitneyu(a, b).pvalue:9.3g}')
rho = spearmanr(W_A[bg, jH], n_links[bg])
print(f'\nSpearman(arm A score, n_links) over eligible genes: rho={rho.statistic:+.3f} p={rho.pvalue:.3g}')
rhoR = spearmanr(W_R[bg, jH], n_links[bg])
print(f'Spearman(arm R score, n_links) over eligible genes: rho={rhoR.statistic:+.3f} p={rhoR.pvalue:.3g}')

# ---- does connectivity alone predict? ------------------------------------------------
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
def cols_of(gidx):
    return np.array([EX[pgenes[i]] for i in gidx if pgenes[i] in EX])
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

K = len(hscA)
def top_elig(score, k):
    s = np.where(elig, score, -np.inf)
    return np.argsort(-s)[:k]
ARMS = {'arm A (chromatin)': cols_of(hscA), 'arm R (RNA)': cols_of(hscR),
        'n_links only':      cols_of(top_elig(n_links, K)),
        'peak density only': cols_of(top_elig(np.nan_to_num(dens, nan=-1), K)),
        'gene span only':    cols_of(top_elig(np.nan_to_num(span, nan=-1), K)),
        'sum_r only':        cols_of(top_elig(sum_r, K))}
# link-count-matched random draws
rng = np.random.default_rng(23)
q = pd.qcut(pd.Series(n_links[bg]), 10, labels=False, duplicates='drop').to_numpy()
qmap = {g: d for g, d in zip(bg, q)}
need = pd.Series([qmap[g] for g in hscA if g in qmap]).value_counts().to_dict()
for i in range(NRAND):
    pick = []
    for d, k in need.items():
        pool = np.array([g for g in bg if qmap[g] == d and g not in set(hscA)])
        pick.append(rng.choice(pool, min(k, len(pool)), replace=False))
    ARMS[f'linkcount-rand #{i}'] = cols_of(np.concatenate(pick))
res = {n: run(s) for n, s in ARMS.items() if len(s) > 5}

print(f'\n=== does connectivity ALONE predict induction response? (K={K}) ===')
print(f'{"arm":26s}{"n":>4s}' + ''.join(f'{f"CV C={C:g}":>12s}{"xwave":>9s}' for C in C_GRID))
for n in ARMS:
    if n.startswith('linkcount-rand') or n not in res: continue
    print(f'  {n:24s}{len(ARMS[n]):4d}' + ''.join(f'{res[n][C][0].mean():12.3f}{res[n][C][1]:9.3f}' for C in C_GRID))
rk = [n for n in res if n.startswith('linkcount-rand')]
print(f'  {"link-count-matched random":24s}{K:4d}' +
      ''.join(f'{np.mean([res[k][C][0].mean() for k in rk]):12.3f}'
              f'{np.mean([res[k][C][1] for k in rk]):9.3f}' for C in C_GRID) + '   (mean of 20)')
print('\narm A vs each connectivity-only arm and vs the matched random draws:')
for k2 in ['n_links only', 'peak density only', 'gene span only', 'sum_r only', 'arm R (RNA)']:
    if k2 not in res: continue
    for C in C_GRID:
        d = res['arm A (chromatin)'][C][0] - res[k2][C][0]
        print(f'  C={C:g}  A vs {k2:20s} CV {d.mean():+.4f} p={wilcoxon(d).pvalue:9.3g} '
              f'| xwave {res["arm A (chromatin)"][C][1]-res[k2][C][1]:+.4f}')
for C in C_GRID:
    a = res['arm A (chromatin)'][C][0]
    w = sum(1 for k in rk if (a - res[k][C][0]).mean() > 0 and wilcoxon(a - res[k][C][0]).pvalue < 0.05)
    ms = [res[k][C][0].mean() for k in rk]
    axw = res['arm A (chromatin)'][C][1]; mxw = [res[k][C][1] for k in rk]
    print(f'  C={C:g}  A {a.mean():.3f} vs link-count-matched {np.mean(ms):.3f}+/-{np.std(ms):.3f} '
          f'(beats {w}/{len(rk)}) | xwave {axw:.3f} above {sum(1 for m in mxw if axw>m)}/{len(rk)}')
tag = os.path.basename(NPZ).replace('.npz','')
json.dump({'label': LABEL,
           'profile': {r['set']: {k: (None if pd.isna(v) else float(v)) for k, v in r.items() if k != 'set'}
                       for r in rows},
           'spearman_armA_nlinks': [float(rho.statistic), float(rho.pvalue)],
           'cv': {n: {str(C): float(res[n][C][0].mean()) for C in C_GRID} for n in res},
           'xwave': {n: {str(C): float(res[n][C][1]) for C in C_GRID} for n in res}},
          open(f'{ROOT}/results/confound_{tag}.json','w'), indent=2)
print(f'\nwrote results/confound_{tag}.json')
