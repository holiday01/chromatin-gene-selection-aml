"""Which of the 22 cell-type programs carries the chromatin advantage?
The theory is that chromatin accessibility decides which genes should be read as a group.
If that is right, the advantage should not be spread evenly: it should sit in the programs
whose accessibility state actually differs between patients who do and do not achieve CR.
Each program is evaluated on its own, A (top-200 chromatin) against R (top-200 RNA), on the
same 100 folds as section 8. Also writes out the chromatin-unique gene list per program.
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

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
cts = pr['cell_types'].astype(str)
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
X = expr[cols].to_numpy(dtype=np.float64).T
name2col = {g: i for i, g in enumerate(expr.index)}
gi = {g: i for i, g in enumerate(pgenes)}
rows = np.array([gi[g] for g in expr.index])
WA, WR = W_A[rows], W_R[rows]
lab = dict(zip(spec.dbgap_rnaseq_sample, spec.induction_response))
y = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
ok = y >= 0; y = y[ok]; X = X[ok]
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0).split(X, y))
print(f'{len(cts)} programs, {len(y)} labelled specimens', flush=True)

tasks = {}
for j, c in enumerate(cts):
    a = np.flatnonzero(WA[:, j]); r = np.flatnonzero(WR[:, j])
    tasks[f'{j}|A'] = a; tasks[f'{j}|R'] = r

def run(item):
    name, sel = item
    M = X[:, sel]; out = {C: [] for C in C_GRID}
    for tr, te in folds:
        mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-8
        Xt, Xv = (M[tr] - mu) / sd, (M[te] - mu) / sd
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=5000).fit(Xt, y[tr])
            out[C].append(roc_auc_score(y[te], clf.decision_function(Xv)))
    return name, {C: np.array(v) for C, v in out.items()}

if __name__ == '__main__':
    with Pool(22) as p:
        res = dict(p.map(run, list(tasks.items())))
    recs = []
    for j, c in enumerate(cts):
        a_, r_ = res[f'{j}|A'], res[f'{j}|R']
        rec = dict(program=c, nA=len(tasks[f'{j}|A']), nR=len(tasks[f'{j}|R']),
                   overlap=len(set(tasks[f'{j}|A']) & set(tasks[f'{j}|R'])))
        for C in C_GRID:
            d = a_[C] - r_[C]
            rec[f'A_C{C:g}'] = a_[C].mean(); rec[f'R_C{C:g}'] = r_[C].mean()
            rec[f'diff_C{C:g}'] = d.mean(); rec[f'p_C{C:g}'] = wilcoxon(d).pvalue
        recs.append(rec)
    D = pd.DataFrame(recs).sort_values('diff_C0.1', ascending=False)
    print('\nper-program clinical AUROC, chromatin (A) vs RNA (R) genes, 100 folds:')
    print(f'  {"program":26s}{"nA":>4s}{"ovl":>5s}' +
          ''.join(f'{f"A C={C:g}":>10s}{f"R":>8s}{f"diff":>9s}{f"p":>10s}' for C in C_GRID))
    for _, r in D.iterrows():
        line = f'  {r.program[:26]:26s}{int(r.nA):4d}{int(r.overlap):5d}'
        for C in C_GRID:
            line += (f'{r[f"A_C{C:g}"]:10.3f}{r[f"R_C{C:g}"]:8.3f}'
                     f'{r[f"diff_C{C:g}"]:+9.4f}{r[f"p_C{C:g}"]:10.3g}')
        print(line)
    for C in C_GRID:
        d = D[f'diff_C{C:g}']
        sig = ((d > 0) & (D[f'p_C{C:g}'] < 0.05)).sum()
        print(f'\nC={C:g}: chromatin wins in {int((d>0).sum())}/{len(D)} programs, '
              f'significant in {int(sig)}; median diff {d.median():+.4f}; '
              f'best single program A={D[f"A_C{C:g}"].max():.3f}')
    D.to_csv(f'{ROOT}/results/per_program.csv', index=False)

    # chromatin-unique genes per program, for interpretation
    out = []
    for j, c in enumerate(cts):
        a = set(pgenes[rows][np.flatnonzero(WA[:, j])])
        r = set(pgenes[rows][np.flatnonzero(WR[:, j])])
        for g in sorted(a - r):
            out.append((c, g, float(WA[gi[g] if False else np.flatnonzero(pgenes[rows] == g)[0], j])))
    U = pd.DataFrame(out, columns=['program', 'gene', 'chromatin_score'])
    U.to_csv(f'{ROOT}/results/chromatin_unique_genes.csv', index=False)
    print(f'\nchromatin-unique genes: {U.gene.nunique()} distinct, {len(U)} program-gene pairs')
    print('wrote results/per_program.csv and results/chromatin_unique_genes.csv')
