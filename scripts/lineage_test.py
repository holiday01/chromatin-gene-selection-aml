"""Two things the per-program table cannot establish on its own.

(1) The best single program (A = 0.789 for G/M prog) is the maximum of 44 noisy estimates,
    so it is biased upward. Checked here on the held-out Waves3+4 split instead.
(2) The lineage pattern - chromatin wins in the myeloid/erythroid/stem compartments that
    AML arises from and loses in mature lymphoid ones - is ONE hypothesis, not 22. Tested
    as a single pre-grouped comparison, and also by permuting the lineage labels.
"""
import os, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from multiprocessing import Pool
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
C_GRID = [1e-2, 1e-1]
NBOOT = 2000

# the grouping is fixed by haematopoietic lineage, decided before looking at the numbers
# only in the sense that it is the AML cell-of-origin compartment vs everything mature lymphoid
MYELOID = {'HSC', 'G/M prog', 'ID2-hi myeloid prog', 'Lymph prog', 'MK/E prog',
           'Proerythroblast', 'Erythroblast', 'Normoblast', 'CD14+ Mono', 'CD16+ Mono'}
LYMPHOID = {'CD4+ T naive', 'CD4+ T activated', 'CD8+ T', 'CD8+ T naive', 'NK', 'ILC',
            'B1 B', 'Transitional B', 'Naive CD20+ B', 'Plasma cell'}

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
cts = pr['cell_types'].astype(str)
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
X = expr[cols].to_numpy(dtype=np.float64).T
gi = {g: i for i, g in enumerate(pgenes)}
rows = np.array([gi[g] for g in expr.index])
WA, WR = W_A[rows], W_R[rows]
S = spec.set_index('dbgap_rnaseq_sample')
lab = S.induction_response.to_dict(); coh = S.cohort.to_dict()
y = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
cw = np.array([str(coh.get(s)) for s in cols])
tr = (y >= 0) & (cw == 'Waves1+2'); te = (y >= 0) & (cw == 'Waves3+4')
print(f'train {tr.sum()}  test {te.sum()}', flush=True)

tasks = {}
for j in range(len(cts)):
    tasks[f'{j}|A'] = np.flatnonzero(WA[:, j])
    tasks[f'{j}|R'] = np.flatnonzero(WR[:, j])

def run(item):
    name, sel = item
    M = X[:, sel]
    mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-8
    Xt, Xv = (M[tr] - mu) / sd, (M[te] - mu) / sd
    return name, {C: LogisticRegression(C=C, max_iter=5000).fit(Xt, y[tr]).decision_function(Xv)
                  for C in C_GRID}

if __name__ == '__main__':
    with Pool(22) as p:
        dec = dict(p.map(run, list(tasks.items())))
    yte = y[te]
    rng = np.random.default_rng(11)
    boot = [b for b in (rng.choice(len(yte), len(yte), replace=True) for _ in range(NBOOT))
            if len(set(yte[b])) == 2]
    recs = []
    for j, c in enumerate(cts):
        rec = dict(program=c, lineage=('myeloid' if c in MYELOID else
                                       'lymphoid' if c in LYMPHOID else 'other'))
        for C in C_GRID:
            a = roc_auc_score(yte, dec[f'{j}|A'][C]); r_ = roc_auc_score(yte, dec[f'{j}|R'][C])
            rec[f'A_C{C:g}'] = a; rec[f'R_C{C:g}'] = r_; rec[f'diff_C{C:g}'] = a - r_
        recs.append(rec)
    D = pd.DataFrame(recs).sort_values('diff_C0.1', ascending=False)
    print('\ncross-wave (Waves3+4, n=96) per-program AUROC:')
    print(f'  {"program":24s}{"lineage":10s}' +
          ''.join(f'{f"A C={C:g}":>10s}{"R":>8s}{"diff":>9s}' for C in C_GRID))
    for _, r in D.iterrows():
        print(f'  {r.program[:24]:24s}{r.lineage:10s}' +
              ''.join(f'{r[f"A_C{C:g}"]:10.3f}{r[f"R_C{C:g}"]:8.3f}{r[f"diff_C{C:g}"]:+9.4f}'
                      for C in C_GRID))

    print('\n(1) does the best program from the CV table hold up out of sample?')
    cvbest = pd.read_csv(f'{ROOT}/results/per_program.csv')
    for C in C_GRID:
        top = cvbest.sort_values(f'A_C{C:g}', ascending=False).iloc[0]
        row = D[D.program == top.program].iloc[0]
        s = dec[f'{list(cts).index(top.program)}|A'][C]
        bs = np.array([roc_auc_score(yte[b], s[b]) for b in boot])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'  C={C:g}  CV best = {top.program} (CV A {top[f"A_C{C:g}"]:.3f}) -> '
              f'cross-wave A {row[f"A_C{C:g}"]:.3f} [{lo:.2f},{hi:.2f}]')

    print('\n(2) is the lineage pattern itself reproducible? (one test, not 22)')
    for C in C_GRID:
        for frame, what in ((cvbest, '100-fold CV'), (D, 'cross-wave')):
            f2 = frame.copy()
            if 'lineage' not in f2.columns:
                f2['lineage'] = ['myeloid' if c in MYELOID else
                                 'lymphoid' if c in LYMPHOID else 'other' for c in f2.program]
            m = f2.loc[f2.lineage == 'myeloid', f'diff_C{C:g}']
            l = f2.loc[f2.lineage == 'lymphoid', f'diff_C{C:g}']
            obs = m.mean() - l.mean()
            allv = np.concatenate([m.values, l.values])
            nm = len(m)
            perm = np.array([(lambda p: p[:nm].mean() - p[nm:].mean())(rng.permutation(allv))
                             for _ in range(20000)])
            pp = (np.abs(perm) >= abs(obs)).mean()
            print(f'  C={C:g} {what:12s} myeloid {m.mean():+.4f} (n={nm}) vs '
                  f'lymphoid {l.mean():+.4f} (n={len(l)}) | gap {obs:+.4f} | '
                  f'permutation p={max(pp,1/20000):.4g} | MWU p={mannwhitneyu(m,l).pvalue:.3g}')
    D.to_csv(f'{ROOT}/results/lineage_crosswave.csv', index=False)
    print('\nwrote results/lineage_crosswave.csv')
