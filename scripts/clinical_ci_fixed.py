"""Does chromatin-based gene SELECTION beat a size-matched random selection?
Fixes three bugs in clinical_endpoint_ci.py:
  (1) weights were applied before per-column z-scoring, which cancels them exactly;
      here we z-score first, then weight, so the weight acts as a per-gene prior on
      coefficient scale.
  (2) C was chosen by TRAINING AUROC, which always picks the weakest penalty;
      here nothing is tuned - every arm is reported at every C, so arms are matched.
  (3) no size-matched random control, so "1620 chromatin genes beat 11319" could
      just be "1620 genes beat 11319 at n=254"; here 10 random draws of the same size.
"""
import os, sys, json
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
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]
N_RAND = 10

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
link_g, link_r = pr['link_g'], pr['link_r']
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
Xg = expr[cols].to_numpy(dtype=np.float64).T
gi = {g: i for i, g in enumerate(pgenes)}
rows = np.array([gi[g] for g in expr.index])
WA, WR = W_A[rows], W_R[rows]
inA, inR = np.where((WA > 0).any(1))[0], np.where((WR > 0).any(1))[0]
lw_full = np.zeros(len(pgenes), np.float64); np.add.at(lw_full, link_g, link_r)
lw = lw_full[rows]
rw = np.abs(W_R[rows]).sum(1)
nA = len(inA)
# size-matched RNA selection: top-nA genes by RNA program weight
rank_R = np.argsort(-rw)[:nA]

lab = dict(zip(spec.dbgap_rnaseq_sample, spec.induction_response))
y = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
ok = y >= 0; y = y[ok]; X = Xg[ok]
print(f'labelled specimens {len(y)}  CR {int((y==0).sum())}  Refractory {int((y==1).sum())}', flush=True)
print(f'A subset {nA} genes | R subset {len(inR)} | all {X.shape[1]} | '
      f'genes with any chromatin link {int((lw>0).sum())}', flush=True)

rng = np.random.default_rng(0)
ARMS = {'all genes': ('sel', np.arange(X.shape[1])),
        'A subset (chromatin)': ('sel', inA),
        'R subset (RNA)': ('sel', inR),
        f'R top-{nA} (size-matched)': ('sel', rank_R)}
for j in range(N_RAND):
    ARMS[f'random {nA} #{j}'] = ('sel', rng.choice(X.shape[1], nA, replace=False))
ARMS['w: chromatin link'] = ('w', lw)
ARMS['w: RNA weight'] = ('w', rw)
ARMS['w: shuffled link'] = ('w', rng.permutation(lw))

folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0).split(X, y))

def run(item):
    name, (kind, arg) = item
    M = X[:, arg] if kind == 'sel' else X
    out = {C: [] for C in C_GRID}
    for tr, te in folds:
        mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-8
        Xt, Xv = (M[tr] - mu) / sd, (M[te] - mu) / sd
        if kind == 'w':                         # weight AFTER standardising, so it survives
            Xt, Xv = Xt * arg[None, :], Xv * arg[None, :]
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=5000).fit(Xt, y[tr])
            out[C].append(roc_auc_score(y[te], clf.decision_function(Xv)))
    return name, M.shape[1], {C: np.array(v) for C, v in out.items()}

if __name__ == '__main__':
    with Pool(min(17, len(ARMS))) as p:
        res = dict((n, (k, s)) for n, k, s in p.map(run, list(ARMS.items())))
    print(f'\nmean test AUROC over 100 folds, nothing tuned (rows x C):', flush=True)
    hdr = ''.join(f'{f"C={C:g}":>12s}' for C in C_GRID)
    print(f'  {"arm":28s}{"p":>6s}{hdr}')
    rand_keys = [k for k in ARMS if k.startswith('random')]
    for name in [k for k in ARMS if not k.startswith('random')] + rand_keys:
        nf, sc = res[name]
        print(f'  {name:28s}{nf:6d}' + ''.join(f'{sc[C].mean():12.3f}' for C in C_GRID), flush=True)
    # the crux: chromatin selection vs the random draws of identical size
    print(f'\nA subset vs each random {nA}-gene draw, paired over the same 100 folds:')
    for C in C_GRID:
        a = res['A subset (chromatin)'][1][C]
        d = [(res[k][1][C] - a).mean() * -1 for k in rand_keys]
        wins = sum(1 for k in rand_keys if wilcoxon(a - res[k][1][C]).pvalue < 0.05
                   and (a - res[k][1][C]).mean() > 0)
        rm = np.mean([res[k][1][C].mean() for k in rand_keys])
        rs = np.std([res[k][1][C].mean() for k in rand_keys])
        print(f'  C={C:<6g} A {a.mean():.3f} | random mean {rm:.3f} +/- {rs:.3f} | '
              f'A-random {a.mean()-rm:+.4f} | A beats draw (p<0.05) in {wins}/{N_RAND}', flush=True)
    print('\npaired Wilcoxon at each C:')
    def cmp(k1, k2):
        for C in C_GRID:
            d = res[k1][1][C] - res[k2][1][C]
            print(f'  C={C:<6g} {k1:24s} vs {k2:24s} {d.mean():+.4f}  p={wilcoxon(d).pvalue:.3g}  '
                  f'wins {int((d>0).sum())}/100')
    for pair in [('A subset (chromatin)', 'all genes'),
                 ('A subset (chromatin)', f'R top-{nA} (size-matched)'),
                 ('w: chromatin link', 'all genes'),
                 ('w: chromatin link', 'w: shuffled link'),
                 ('w: chromatin link', 'w: RNA weight')]:
        cmp(*pair); print()
    json.dump({n: {str(C): [float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                   for C, v in sc.items()} for n, (nf, sc) in res.items()},
              open(f'{ROOT}/results/clinical_cv_fixed.json', 'w'), indent=2)
    print('wrote results/clinical_cv_fixed.json')
