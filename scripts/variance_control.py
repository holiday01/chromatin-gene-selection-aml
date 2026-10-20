"""Is the chromatin gene set just the high-variance genes?
peak-to-gene linking needs expression variance to reach r>=0.30, so the A subset is
enriched for variable genes, and variable genes predict better. Controls, all 1620 genes:
  - top-1620 by variance across the 346 specimens
  - top-1620 by mean expression
  - variance-MATCHED random draws: sample from the same variance deciles as A
  - A genes that are NOT in the R subset (chromatin-unique)
"""
import os, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, mannwhitneyu
from multiprocessing import Pool
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
N_MATCH = 10

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
Xg = expr[cols].to_numpy(dtype=np.float64).T
gi = {g: i for i, g in enumerate(pgenes)}
rows = np.array([gi[g] for g in expr.index])
inA = np.where((W_A[rows] > 0).any(1))[0]
inR = np.where((W_R[rows] > 0).any(1))[0]
nA = len(inA)
lab = dict(zip(spec.dbgap_rnaseq_sample, spec.induction_response))
y = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
ok = y >= 0; y = y[ok]; X = Xg[ok]

var = X.var(0); mean = X.mean(0)
print(f'A subset {nA} genes')
print(f'variance percentile of A genes: median {np.mean(var[inA][:,None] > var[None,:])*100:.1f}th '
      f'(all genes = 50th)')
print(f'  A genes in the top-1620 by variance: {len(set(inA) & set(np.argsort(-var)[:nA]))}/{nA}')
print(f'  A genes also in R subset:            {len(set(inA) & set(inR))}/{nA}')

# variance-matched random draws: same decile composition as A
dec = pd.qcut(pd.Series(var), 10, labels=False, duplicates='drop').to_numpy()
need = pd.Series(dec[inA]).value_counts().to_dict()
rng = np.random.default_rng(1)
matched = []
for j in range(N_MATCH):
    pick = []
    for d, k in need.items():
        pool = np.setdiff1d(np.where(dec == d)[0], inA)       # exclude A itself
        pick.append(rng.choice(pool, min(k, len(pool)), replace=False))
    matched.append(np.concatenate(pick))
print(f'variance-matched draws: {[len(m) for m in matched[:3]]}... (A has {nA})')

A_only = np.setdiff1d(inA, inR)
ARMS = {'A subset (chromatin)': inA,
        f'A-only (not in R), n={len(A_only)}': A_only,
        'top-1620 by variance': np.argsort(-var)[:nA],
        'top-1620 by mean expr': np.argsort(-mean)[:nA]}
for j, m in enumerate(matched):
    ARMS[f'var-matched random #{j}'] = m
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0).split(X, y))

def run(item):
    name, sel = item
    M = X[:, sel]
    out = {C: [] for C in C_GRID}
    for tr, te in folds:
        mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-8
        Xt, Xv = (M[tr] - mu) / sd, (M[te] - mu) / sd
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=5000).fit(Xt, y[tr])
            out[C].append(roc_auc_score(y[te], clf.decision_function(Xv)))
    return name, M.shape[1], {C: np.array(v) for C, v in out.items()}

if __name__ == '__main__':
    with Pool(min(14, len(ARMS))) as p:
        res = dict((n, (k, s)) for n, k, s in p.map(run, list(ARMS.items())))
    print('\nmean test AUROC over the same 100 folds:')
    print(f'  {"arm":34s}{"p":>6s}' + ''.join(f'{f"C={C:g}":>12s}' for C in C_GRID))
    for name in ARMS:
        nf, sc = res[name]
        print(f'  {name:34s}{nf:6d}' + ''.join(f'{sc[C].mean():12.3f}' for C in C_GRID))
    print('\nA subset vs the variance-matched random draws:')
    mk = [k for k in ARMS if k.startswith('var-matched')]
    for C in C_GRID:
        a = res['A subset (chromatin)'][1][C]
        ms = [res[k][1][C].mean() for k in mk]
        w = sum(1 for k in mk if wilcoxon(a - res[k][1][C]).pvalue < 0.05
                and (a - res[k][1][C]).mean() > 0)
        print(f'  C={C:<6g} A {a.mean():.3f} | var-matched {np.mean(ms):.3f} +/- {np.std(ms):.3f} | '
              f'diff {a.mean()-np.mean(ms):+.4f} | A beats draw (p<0.05) in {w}/{len(mk)}')
    print('\npaired Wilcoxon:')
    for k2 in ['top-1620 by variance', 'top-1620 by mean expr', f'A-only (not in R), n={len(A_only)}']:
        for C in C_GRID:
            d = res['A subset (chromatin)'][1][C] - res[k2][1][C]
            print(f'  C={C:<6g} A vs {k2:34s} {d.mean():+.4f}  p={wilcoxon(d).pvalue:.3g}  '
                  f'wins {int((d>0).sum())}/100')
        print()
    json.dump({n: {str(C): float(v.mean()) for C, v in sc.items()} for n, (nf, sc) in res.items()},
              open(f'{ROOT}/results/variance_control.json', 'w'), indent=2)
    print('wrote results/variance_control.json')
