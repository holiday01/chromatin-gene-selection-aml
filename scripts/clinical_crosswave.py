"""External check on the clinical endpoint: train on Waves 1+2, test on Waves 3+4.
Section 8 used random CV folds, which can leak the cohort/batch structure. This is one
fixed split, so differences get a paired bootstrap over test specimens rather than a
Wilcoxon over folds. The program config is left at the analysis default
(window 250 kb, r>=0.30, top_m=200) - NOT the best config from the sweep.
"""
import os, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
NBOOT = 2000

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
X = expr[cols].to_numpy(dtype=np.float64).T
gi = {g: i for i, g in enumerate(pgenes)}
rows = np.array([gi[g] for g in expr.index])
inA = np.where((W_A[rows] > 0).any(1))[0]
inR = np.where((W_R[rows] > 0).any(1))[0]
rw = np.abs(W_R[rows]).sum(1)
nA = len(inA)
S = spec.set_index('dbgap_rnaseq_sample')
lab = S.induction_response.to_dict(); coh = S.cohort.to_dict()
y = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
cw = np.array([str(coh.get(s)) for s in cols])
print('cohort values:', pd.Series(cw).value_counts().to_dict())
tr = (y >= 0) & np.isin(cw, ['Waves1+2'])
te = (y >= 0) & np.isin(cw, ['Waves3+4'])
print(f'train {tr.sum()} (CR {int((y[tr]==0).sum())}/Ref {int((y[tr]==1).sum())})  '
      f'test {te.sum()} (CR {int((y[te]==0).sum())}/Ref {int((y[te]==1).sum())})')
if te.sum() < 30:
    raise SystemExit('cohort labels did not match - inspect the printed value counts')

var = X[tr].var(0)
dec = pd.qcut(pd.Series(var), 10, labels=False, duplicates='drop').to_numpy()
rng = np.random.default_rng(3)
need = pd.Series(dec[inA]).value_counts().to_dict()
matched = []
for j in range(10):
    pick = []
    for d, k in need.items():
        pool = np.setdiff1d(np.where(dec == d)[0], inA)
        pick.append(rng.choice(pool, min(k, len(pool)), replace=False))
    matched.append(np.concatenate(pick))

ARMS = {'all genes': np.arange(X.shape[1]),
        'A subset (chromatin)': inA,
        'R subset (RNA)': inR,
        f'R top-{nA} (size-matched)': np.argsort(-rw)[:nA],
        f'top-{nA} by variance': np.argsort(-var)[:nA]}
for j, m in enumerate(matched):
    ARMS[f'var-matched random #{j}'] = m

scores = {}            # name -> {C: decision values on the test set}
for name, sel in ARMS.items():
    M = X[:, sel]
    mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-8
    Xt, Xv = (M[tr] - mu) / sd, (M[te] - mu) / sd
    scores[name] = {}
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000).fit(Xt, y[tr])
        scores[name][C] = clf.decision_function(Xv)

yte = y[te]
boot = [rng.choice(len(yte), len(yte), replace=True) for _ in range(NBOOT)]
boot = [b for b in boot if len(set(yte[b])) == 2]
print(f'\ncross-wave test AUROC (n={len(yte)}), 95% bootstrap CI over {len(boot)} resamples:')
print(f'  {"arm":30s}{"p":>6s}' + ''.join(f'{f"C={C:g}":>20s}' for C in C_GRID))
auc = {}
for name in ARMS:
    line = f'  {name:30s}{len(ARMS[name]):6d}'
    auc[name] = {}
    for C in C_GRID:
        s = scores[name][C]
        a = roc_auc_score(yte, s)
        bs = np.array([roc_auc_score(yte[b], s[b]) for b in boot])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        auc[name][C] = (a, lo, hi)
        line += f'{a:8.3f} [{lo:.2f},{hi:.2f}]'
    print(line, flush=True)

print('\npaired bootstrap on the difference (same resamples for both arms):')
def pb(k1, k2):
    for C in C_GRID:
        s1, s2 = scores[k1][C], scores[k2][C]
        d = np.array([roc_auc_score(yte[b], s1[b]) - roc_auc_score(yte[b], s2[b]) for b in boot])
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        print(f'  C={C:<6g} {k1:24s} vs {k2:26s} {d.mean():+.4f} '
              f'[{lo:+.3f},{hi:+.3f}]  p={max(p,1/len(boot)):.3g}')
    print()
for k2 in ['all genes', f'R top-{nA} (size-matched)', f'top-{nA} by variance']:
    pb('A subset (chromatin)', k2)
mk = [k for k in ARMS if k.startswith('var-matched')]
print('A subset vs the variance-matched random draws:')
for C in C_GRID:
    a = auc['A subset (chromatin)'][C][0]
    ms = [auc[k][C][0] for k in mk]
    print(f'  C={C:<6g} A {a:.3f} | random {np.mean(ms):.3f} +/- {np.std(ms):.3f} '
          f'(range {min(ms):.3f}-{max(ms):.3f}) | A above {sum(1 for m in ms if a>m)}/10')
json.dump({n: {str(C): list(map(float, v)) for C, v in d.items()} for n, d in auc.items()},
          open(f'{ROOT}/results/clinical_crosswave.json', 'w'), indent=2)
print('\nwrote results/clinical_crosswave.json')
