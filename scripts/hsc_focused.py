"""The one result that was stable in both splits and both penalties: the HSC chromatin
program. 200 genes, selected by the accessibility of their linked peaks in HSCs of a
healthy marrow, predicting whether an AML patient achieves complete remission on
induction chemotherapy.

Controls, all at 200 genes: the HSC RNA program (same cells, same program, expression
instead of accessibility), 20 variance-matched random draws, and the other 21 chromatin
programs - so "HSC" has to beat its own siblings, not just noise.
"""
import os, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
from multiprocessing import Pool
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
C_GRID = [1e-2, 1e-1]
NRAND = 20

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
cts = list(pr['cell_types'].astype(str))
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
Xall = expr[cols].to_numpy(dtype=np.float64).T
gi = {g: i for i, g in enumerate(pgenes)}
rows = np.array([gi[g] for g in expr.index])
WA, WR = W_A[rows], W_R[rows]
jH = cts.index('HSC')
hscA, hscR = np.flatnonzero(WA[:, jH]), np.flatnonzero(WR[:, jH])
S = spec.set_index('dbgap_rnaseq_sample')
lab = S.induction_response.to_dict(); coh = S.cohort.to_dict()
yall = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
cw = np.array([str(coh.get(s)) for s in cols])
okm = yall >= 0
print(f'HSC chromatin program {len(hscA)} genes, HSC RNA program {len(hscR)}, '
      f'overlap {len(set(hscA)&set(hscR))}')
print('genes unique to the chromatin version:',
      ', '.join(sorted(expr.index[np.setdiff1d(hscA, hscR)])[:25]), '...')

var = Xall[okm].var(0)
dec10 = pd.qcut(pd.Series(var), 10, labels=False, duplicates='drop').to_numpy()
rng = np.random.default_rng(23)
need = pd.Series(dec10[hscA]).value_counts().to_dict()
rand = []
for _ in range(NRAND):
    pick = []
    for d, k in need.items():
        pool = np.setdiff1d(np.where(dec10 == d)[0], hscA)
        pick.append(rng.choice(pool, min(k, len(pool)), replace=False))
    rand.append(np.concatenate(pick))

ARMS = {'HSC chromatin': hscA, 'HSC RNA': hscR, 'all genes': np.arange(Xall.shape[1])}
for j, c in enumerate(cts):
    if c != 'HSC': ARMS[f'chrom: {c}'] = np.flatnonzero(WA[:, j])
for i, r in enumerate(rand): ARMS[f'var-matched random #{i}'] = r

X, y = Xall[okm], yall[okm]
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0).split(X, y))
tr = (yall >= 0) & (cw == 'Waves1+2'); te = (yall >= 0) & (cw == 'Waves3+4')

def run(item):
    name, sel = item
    M, Mw = X[:, sel], Xall[:, sel]
    cvs = {C: [] for C in C_GRID}
    for a, b in folds:
        mu, sd = M[a].mean(0), M[a].std(0) + 1e-8
        Xt, Xv = (M[a] - mu) / sd, (M[b] - mu) / sd
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=5000).fit(Xt, y[a])
            cvs[C].append(roc_auc_score(y[b], clf.decision_function(Xv)))
    mu, sd = Mw[tr].mean(0), Mw[tr].std(0) + 1e-8
    Xt, Xv = (Mw[tr] - mu) / sd, (Mw[te] - mu) / sd
    xw = {C: LogisticRegression(C=C, max_iter=5000).fit(Xt, yall[tr]).decision_function(Xv)
          for C in C_GRID}
    return name, {C: np.array(v) for C, v in cvs.items()}, xw

if __name__ == '__main__':
    with Pool(22) as p:
        out = p.map(run, list(ARMS.items()))
    cv = {n: a for n, a, _ in out}; xw = {n: b for n, _, b in out}
    yte = yall[te]
    boot = [b for b in (rng.choice(len(yte), len(yte), replace=True) for _ in range(2000))
            if len(set(yte[b])) == 2]
    print(f'\n{"arm":28s}' + ''.join(f'{f"CV C={C:g}":>12s}{f"xwave":>22s}' for C in C_GRID))
    for n in ['HSC chromatin', 'HSC RNA', 'all genes']:
        line = f'{n:28s}'
        for C in C_GRID:
            a = roc_auc_score(yte, xw[n][C])
            bs = np.array([roc_auc_score(yte[b], xw[n][C][b]) for b in boot])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            line += f'{cv[n][C].mean():12.3f}{a:10.3f} [{lo:.2f},{hi:.2f}]'
        print(line)
    print('\nHSC chromatin vs HSC RNA:')
    for C in C_GRID:
        d = cv['HSC chromatin'][C] - cv['HSC RNA'][C]
        s1, s2 = xw['HSC chromatin'][C], xw['HSC RNA'][C]
        dw = np.array([roc_auc_score(yte[b], s1[b]) - roc_auc_score(yte[b], s2[b]) for b in boot])
        lo, hi = np.percentile(dw, [2.5, 97.5])
        pw = 2 * min((dw <= 0).mean(), (dw >= 0).mean())
        print(f'  C={C:g}  CV {d.mean():+.4f} p={wilcoxon(d).pvalue:.3g} wins {int((d>0).sum())}/100'
              f'  |  xwave {dw.mean():+.4f} [{lo:+.3f},{hi:+.3f}] p={max(pw,1/len(boot)):.3g}')
    print('\nHSC chromatin vs 20 variance-matched random 200-gene sets:')
    rk = [k for k in ARMS if k.startswith('var-matched')]
    for C in C_GRID:
        a = cv['HSC chromatin'][C]
        ms = [cv[k][C].mean() for k in rk]
        w = sum(1 for k in rk if wilcoxon(a - cv[k][C]).pvalue < 0.05 and (a - cv[k][C]).mean() > 0)
        axw = roc_auc_score(yte, xw['HSC chromatin'][C])
        mxw = [roc_auc_score(yte, xw[k][C]) for k in rk]
        print(f'  C={C:g}  CV {a.mean():.3f} vs {np.mean(ms):.3f}+/-{np.std(ms):.3f} '
              f'(beats {w}/{NRAND} at p<0.05)  |  xwave {axw:.3f} vs '
              f'{np.mean(mxw):.3f}+/-{np.std(mxw):.3f} (above {sum(1 for m in mxw if axw>m)}/{NRAND})')
    print('\nHSC chromatin vs the other 21 chromatin programs (rank of HSC):')
    pk = [k for k in ARMS if k.startswith('chrom: ')]
    for C in C_GRID:
        a = cv['HSC chromatin'][C].mean(); ms = sorted((cv[k][C].mean() for k in pk), reverse=True)
        axw = roc_auc_score(yte, xw['HSC chromatin'][C])
        mxw = sorted((roc_auc_score(yte, xw[k][C]) for k in pk), reverse=True)
        print(f'  C={C:g}  CV HSC {a:.3f} -> rank {sum(1 for m in ms if m>a)+1}/22 '
              f'(best other {ms[0]:.3f})  |  xwave HSC {axw:.3f} -> '
              f'rank {sum(1 for m in mxw if m>axw)+1}/22 (best other {mxw[0]:.3f})')
    json.dump({'cv': {n: {str(C): float(v.mean()) for C, v in d.items()} for n, d in cv.items()},
               'xwave': {n: {str(C): float(roc_auc_score(yte, s)) for C, s in d.items()}
                         for n, d in xw.items()}},
              open(f'{ROOT}/results/hsc_focused.json', 'w'), indent=2)
    print('\nwrote results/hsc_focused.json')
