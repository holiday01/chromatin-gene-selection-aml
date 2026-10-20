"""Run the settled clinical-endpoint analysis against ANY program file, so the healthy-BMMC
reference and the AML reference go through identical code.

Usage: eval_programs.py <programs.npz> <label>
Reports: the subset comparison with variance-matched random controls (100 paired folds),
the HSC-program comparison, the rank of HSC among that reference's chromatin programs,
the per-program lineage split, and the cross-wave external split for all of it.
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
NPZ, LABEL = sys.argv[1], sys.argv[2]
ROOT = '/home/holiday01/2026_cm'
C_GRID = [1e-2, 1e-1]
NRAND = 20
# myeloid = the compartment AML arises from; names differ between the two references
MYE = {'HSC','G/M prog','ID2-hi myeloid prog','Lymph prog','MK/E prog','Proerythroblast',
       'Erythroblast','Normoblast','CD14+ Mono','CD16+ Mono',
       'GMP','Progenitor','CLP','Monocytes','CD16.Monocytes','Early.Erythrocyte',
       'Late.Erythrocyte','Early.Basophil'}
LYM = {'CD4+ T naive','CD4+ T activated','CD8+ T','CD8+ T naive','NK','ILC','B1 B',
       'Transitional B','Naive CD20+ B','Plasma cell',
       'CD4.Naive','CD4.Memory','CD8.Naive','CD8.Memory','CD8.Effector','B.Cell','Pre.B.Cell'}

pr = np.load(NPZ, allow_pickle=True)
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
rw = np.abs(WR).sum(1)
inA, inR = np.where((WA > 0).any(1))[0], np.where((WR > 0).any(1))[0]
nA = len(inA)
S = spec.set_index('dbgap_rnaseq_sample')
lab = S.induction_response.to_dict(); coh = S.cohort.to_dict()
yall = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
cw = np.array([str(coh.get(s)) for s in cols])
okm = yall >= 0
X, y = Xall[okm], yall[okm]
tr = (yall >= 0) & (cw == 'Waves1+2'); te = (yall >= 0) & (cw == 'Waves3+4')
print(f'=== {LABEL} ===')
print(f'{len(cts)} programs | A subset {nA} genes | R subset {len(inR)} | '
      f'all {X.shape[1]} | labelled {len(y)}')
jH = cts.index('HSC')
hscA, hscR = np.flatnonzero(WA[:, jH]), np.flatnonzero(WR[:, jH])
print(f'HSC program: A {len(hscA)} genes, R {len(hscR)}, overlap {len(set(hscA)&set(hscR))}')

var = X.var(0)
dec = pd.qcut(pd.Series(var), 10, labels=False, duplicates='drop').to_numpy()
rng = np.random.default_rng(31)
def matched(sel, n):
    need = pd.Series(dec[sel]).value_counts().to_dict()
    out = []
    for _ in range(n):
        pick = []
        for d, k in need.items():
            pool = np.setdiff1d(np.where(dec == d)[0], sel)
            pick.append(rng.choice(pool, min(k, len(pool)), replace=False))
        out.append(np.concatenate(pick))
    return out

ARMS = {'all genes': np.arange(X.shape[1]), 'A subset': inA, 'R subset': inR,
        'R size-matched': np.argsort(-rw)[:nA],
        f'top-{nA} by variance': np.argsort(-var)[:nA],
        'HSC chromatin': hscA, 'HSC RNA': hscR}
for i, m in enumerate(matched(inA, 10)): ARMS[f'sub-rand #{i}'] = m
for i, m in enumerate(matched(hscA, NRAND)): ARMS[f'hsc-rand #{i}'] = m
for j, c in enumerate(cts):
    ARMS[f'prog-A: {c}'] = np.flatnonzero(WA[:, j])
    ARMS[f'prog-R: {c}'] = np.flatnonzero(WR[:, j])
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0).split(X, y))

def run(item):
    name, sel = item
    M, Mw = X[:, sel], Xall[:, sel]
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
    return name, {C: np.array(v) for C, v in cvs.items()}, xw

if __name__ == '__main__':
    with Pool(22) as p:
        out = p.map(run, list(ARMS.items()))
    cv = {n: a for n, a, _ in out}; xw = {n: b for n, _, b in out}
    print(f'\n{"arm":24s}' + ''.join(f'{f"CV C={C:g}":>12s}{"xwave":>9s}' for C in C_GRID))
    for n in ['all genes', 'A subset', 'R subset', 'R size-matched',
              f'top-{nA} by variance', 'HSC chromatin', 'HSC RNA']:
        print(f'  {n:22s}' + ''.join(f'{cv[n][C].mean():12.3f}{xw[n][C]:9.3f}' for C in C_GRID))
    def line(k1, k2, tag):
        for C in C_GRID:
            d = cv[k1][C] - cv[k2][C]
            print(f'  C={C:g} {tag:34s} CV {d.mean():+.4f} p={wilcoxon(d).pvalue:9.3g} '
                  f'wins {int((d>0).sum()):3d}/100 | xwave {xw[k1][C]-xw[k2][C]:+.4f}')
    print('\npaired comparisons:')
    line('A subset', 'all genes', 'A subset vs all genes')
    line('A subset', 'R size-matched', 'A subset vs R size-matched')
    line('A subset', f'top-{nA} by variance', 'A subset vs top-by-variance')
    line('HSC chromatin', 'HSC RNA', 'HSC chromatin vs HSC RNA')
    line('HSC chromatin', 'all genes', 'HSC chromatin vs all genes')
    print('\nvs variance-matched random draws:')
    for key, pref, n in (('A subset', 'sub-rand', 10), ('HSC chromatin', 'hsc-rand', NRAND)):
        rk = [k for k in ARMS if k.startswith(pref)]
        for C in C_GRID:
            a = cv[key][C]; ms = [cv[k][C].mean() for k in rk]
            w = sum(1 for k in rk if wilcoxon(a - cv[k][C]).pvalue < 0.05
                    and (a - cv[k][C]).mean() > 0)
            axw = xw[key][C]; mxw = [xw[k][C] for k in rk]
            print(f'  C={C:g} {key:16s} CV {a.mean():.3f} vs {np.mean(ms):.3f}+/-{np.std(ms):.3f} '
                  f'(beats {w}/{n}) | xwave {axw:.3f} vs {np.mean(mxw):.3f}+/-{np.std(mxw):.3f} '
                  f'(above {sum(1 for m in mxw if axw>m)}/{n})')
    print('\nHSC rank among this reference\'s chromatin programs:')
    pk = [f'prog-A: {c}' for c in cts]
    for C in C_GRID:
        a = cv['HSC chromatin'][C].mean(); ms = sorted((cv[k][C].mean() for k in pk), reverse=True)
        axw = xw['HSC chromatin'][C]; mxw = sorted((xw[k][C] for k in pk), reverse=True)
        print(f'  C={C:g}  CV {a:.3f} rank {ms.index(a)+1}/{len(cts)} (best {ms[0]:.3f}) | '
              f'xwave {axw:.3f} rank {mxw.index(axw)+1}/{len(cts)} (best {mxw[0]:.3f})')
    print('\nper-program A-minus-R, by lineage:')
    recs = []
    for c in cts:
        r = dict(program=c, lineage='myeloid' if c in MYE else 'lymphoid' if c in LYM else 'other')
        for C in C_GRID:
            r[f'cv_{C:g}'] = (cv[f'prog-A: {c}'][C] - cv[f'prog-R: {c}'][C]).mean()
            r[f'p_{C:g}'] = wilcoxon(cv[f'prog-A: {c}'][C] - cv[f'prog-R: {c}'][C]).pvalue
            r[f'xw_{C:g}'] = xw[f'prog-A: {c}'][C] - xw[f'prog-R: {c}'][C]
        recs.append(r)
    D = pd.DataFrame(recs).sort_values(f'cv_{C_GRID[-1]:g}', ascending=False)
    print(D.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    for C in C_GRID:
        m = D.loc[D.lineage == 'myeloid', f'cv_{C:g}']; l = D.loc[D.lineage == 'lymphoid', f'cv_{C:g}']
        mx = D.loc[D.lineage == 'myeloid', f'xw_{C:g}']; lx = D.loc[D.lineage == 'lymphoid', f'xw_{C:g}']
        allv = np.concatenate([m.values, l.values]); nm = len(m)
        perm = np.array([(lambda q: q[:nm].mean() - q[nm:].mean())(rng.permutation(allv))
                         for _ in range(20000)])
        obs = m.mean() - l.mean()
        print(f'  C={C:g}  CV myeloid {m.mean():+.4f} (n={nm}) vs lymphoid {l.mean():+.4f} '
              f'(n={len(l)}) gap {obs:+.4f} perm p={max((np.abs(perm)>=abs(obs)).mean(),1/20000):.4g}'
              f' | xwave gap {mx.mean()-lx.mean():+.4f}')
    tag = os.path.basename(NPZ).replace('.npz', '')
    D.to_csv(f'{ROOT}/results/eval_{tag}_per_program.csv', index=False)
    json.dump({'cv': {n: {str(C): float(v.mean()) for C, v in d.items()} for n, d in cv.items()},
               'xwave': {n: {str(C): float(v) for C, v in d.items()} for n, d in xw.items()}},
              open(f'{ROOT}/results/eval_{tag}.json', 'w'), indent=2)
    print(f'\nwrote results/eval_{tag}.json and results/eval_{tag}_per_program.csv')
