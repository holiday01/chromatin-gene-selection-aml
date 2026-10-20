"""Published-method baseline: LSC17 (Ng et al. Nature 2016) at its own 17-gene budget.

Cancer Medicine's Methods Paper type requires comparison against conventional techniques.
LSC17 is the right comparator: a leukaemic-stem-cell expression signature built to predict
response to induction therapy in AML from bulk RNA -- same endpoint, same compartment, same
input as our claim.

!! PROVISIONAL GENE LIST !!  The 17 symbols below are written from memory and were NOT
verified against Ng et al. 2016. All 17 resolve in the Beat AML matrix under the
microarray-era symbols the signature was defined with (KIAA0125, C19orf77, NGFRAP1, GPR56),
which is consistent but is not verification. Check against the Nature supplementary
(doi:10.1038/nature20598) before this number goes in a manuscript.

The published score is a weighted sum with gene-specific coefficients that are only in that
supplementary, so this script does NOT reproduce the published score. It compares LSC17 as a
GENE SET through the identical model and folds as every other arm, which is the comparison
the paper's thesis actually calls for: same classifier, same patients, same folds, only the
gene-selection rule differs. Add the weighted score as a separate arm once the coefficients
are in hand.
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
NPZ = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/data/programs/programs.npz'
C_GRID = [1e-2, 1e-1]
LSC17 = ['DNMT3B','ZBTB46','NYNRIN','ARHGAP22','LAPTM4B','MMRN1','DPYSL3','KIAA0125','CDK6',
         'CPXM1','SOCS2','C19orf77','EMP1','NGFRAP1','CD34','AKR1C3','GPR56']

pr = np.load(NPZ, allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
cts = list(pr['cell_types'].astype(str))
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr.drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
S = spec.set_index('dbgap_rnaseq_sample')
lab, coh = S.induction_response.to_dict(), S.cohort.to_dict()
yall = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
cw = np.array([str(coh.get(s)) for s in cols])
okm = yall >= 0
tr, te = okm & (cw == 'Waves1+2'), okm & (cw == 'Waves3+4')

missing = [g for g in LSC17 if g not in expr.index]
print(f'LSC17: {len(LSC17)-len(missing)}/17 symbols found' + (f', MISSING {missing}' if missing else ''))
lsc = [g for g in LSC17 if g in expr.index]

# the HSC program's own top-17, by each arm's score -- LSC17's budget, our selection rules
jH = cts.index('HSC')
gi = {g: i for i, g in enumerate(pgenes)}
def top_k_in_program(W, k):
    s = W[:, jH]
    idx = np.argsort(-s)[:k]
    return [pgenes[i] for i in idx if s[i] > 0]
hscA17, hscR17 = top_k_in_program(W_A, 17), top_k_in_program(W_R, 17)
hscA200 = [g for g in top_k_in_program(W_A, 200) if g in expr.index]
print(f'HSC arm A top-17: {hscA17}')
print(f'HSC arm R top-17: {hscR17}')
print(f'overlap of LSC17 with HSC arm A 200-gene program: '
      f'{len(set(lsc) & set(hscA200))}/{len(lsc)}')

def mat(names):
    names = [g for g in names if g in expr.index]
    return expr.loc[names, cols].to_numpy(dtype=np.float64).T, names

var_full = expr.loc[:, cols].to_numpy(dtype=np.float64).T.var(0)
dec = pd.qcut(pd.Series(var_full), 10, labels=False, duplicates='drop').to_numpy()
allg = np.array(expr.index)
gpos = {g: i for i, g in enumerate(allg)}
rng = np.random.default_rng(7)
def matched_random(names, n):
    sel = np.array([gpos[g] for g in names if g in gpos])
    need = pd.Series(dec[sel]).value_counts().to_dict()
    out = []
    for _ in range(n):
        pick = []
        for d, k in need.items():
            pool = np.setdiff1d(np.where(dec == d)[0], sel)
            pick.append(rng.choice(pool, min(k, len(pool)), replace=False))
        out.append(list(allg[np.concatenate(pick)]))
    return out

ARMS = {'LSC17 (published gene set)': lsc,
        'HSC arm A top-17 (chromatin)': hscA17,
        'HSC arm R top-17 (RNA)': hscR17,
        'HSC arm A 200 (ours, full)': hscA200}
for i, m in enumerate(matched_random(lsc, 20)): ARMS[f'rand17 #{i}'] = m

X0 = expr.loc[:, cols]
y = yall[okm]
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0).split(np.zeros((okm.sum(), 1)), y))
res = {}
for name, names in ARMS.items():
    Mw, used = mat(names)
    M = Mw[okm]
    cvs = {C: [] for C in C_GRID}
    for a, b in folds:
        mu, sd = M[a].mean(0), M[a].std(0) + 1e-8
        Xt, Xv = (M[a] - mu) / sd, (M[b] - mu) / sd
        for C in C_GRID:
            cvs[C].append(roc_auc_score(y[b], LogisticRegression(C=C, max_iter=5000)
                                        .fit(Xt, y[a]).decision_function(Xv)))
    mu, sd = Mw[tr].mean(0), Mw[tr].std(0) + 1e-8
    Xt, Xv = (Mw[tr] - mu) / sd, (Mw[te] - mu) / sd
    xw = {C: roc_auc_score(yall[te], LogisticRegression(C=C, max_iter=5000)
                           .fit(Xt, yall[tr]).decision_function(Xv)) for C in C_GRID}
    res[name] = ({C: np.array(v) for C, v in cvs.items()}, xw, len(used))

print(f'\n{"arm":32s}{"n":>4s}' + ''.join(f'{f"CV C={C:g}":>12s}{"xwave":>9s}' for C in C_GRID))
for n in ARMS:
    if n.startswith('rand17'): continue
    cv, xw, k = res[n]
    print(f'  {n:30s}{k:4d}' + ''.join(f'{cv[C].mean():12.3f}{xw[C]:9.3f}' for C in C_GRID))
rk = [n for n in ARMS if n.startswith('rand17')]
cvm = {C: np.mean([res[k][0][C].mean() for k in rk]) for C in C_GRID}
cvs_ = {C: np.std([res[k][0][C].mean() for k in rk]) for C in C_GRID}
print(f'  {"variance-matched random 17":30s}{17:4d}' +
      ''.join(f'{cvm[C]:12.3f}{np.mean([res[k][1][C] for k in rk]):9.3f}' for C in C_GRID)
      + '   (mean of 20)')

print('\npaired comparisons (100 folds):')
def line(k1, k2):
    for C in C_GRID:
        d = res[k1][0][C] - res[k2][0][C]
        print(f'  C={C:g}  {k1} vs {k2}\n        CV {d.mean():+.4f} p={wilcoxon(d).pvalue:9.3g} '
              f'wins {int((d>0).sum())}/100 | xwave {res[k1][1][C]-res[k2][1][C]:+.4f}')
line('HSC arm A top-17 (chromatin)', 'LSC17 (published gene set)')
line('HSC arm A top-17 (chromatin)', 'HSC arm R top-17 (RNA)')
line('HSC arm A 200 (ours, full)', 'LSC17 (published gene set)')
print('\nLSC17 vs its variance-matched random 17-gene draws:')
for C in C_GRID:
    a = res['LSC17 (published gene set)'][0][C]
    w = sum(1 for k in rk if (a - res[k][0][C]).mean() > 0 and wilcoxon(a - res[k][0][C]).pvalue < 0.05)
    print(f'  C={C:g}  LSC17 {a.mean():.3f} vs {cvm[C]:.3f}+/-{cvs_[C]:.3f} (beats {w}/20)')
json.dump({'note': 'PROVISIONAL LSC17 gene list, unverified against Ng 2016; gene set only, '
                   'published weights not applied',
           'lsc17_used': lsc, 'hsc_A_top17': hscA17, 'hsc_R_top17': hscR17,
           'cv': {n: {str(C): float(v.mean()) for C, v in res[n][0].items()} for n in res},
           'xwave': {n: {str(C): float(v) for C, v in res[n][1].items()} for n in res}},
          open(f'{ROOT}/results/baseline_lsc17.json', 'w'), indent=2)
print('\nwrote results/baseline_lsc17.json')
