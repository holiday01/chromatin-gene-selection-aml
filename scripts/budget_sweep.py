"""At what gene budget does the chromatin selection rule start to beat the RNA rule?

baseline_lsc17.py found arm A LOSES to arm R at a 17-gene budget (-0.054 / -0.021) while
S17 found it WINS at 200 (+0.032 / +0.046). The claim therefore depends on the budget, and
the paper has to say where the crossover is instead of reporting one lucky k.

Same patients, same programs, same folds, same classifier as everywhere else -- only k and
the selection rule change. Run for all three reference constructions.
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
C_GRID = [1e-2, 1e-1]
KS = [10, 17, 25, 40, 60, 100, 150, 200]
REFS = {'BMMC': 'programs', 'AML-mal': 'programs_aml_malignant_v2', 'AML-all': 'programs_aml_all_v2'}

expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr.drop_duplicates('display_label').set_index('display_label')
S = spec.set_index('dbgap_rnaseq_sample')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
lab, coh = S.induction_response.to_dict(), S.cohort.to_dict()
yall = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
cw = np.array([str(coh.get(s)) for s in cols])
okm = yall >= 0; y = yall[okm]
tr, te = okm & (cw == 'Waves1+2'), okm & (cw == 'Waves3+4')
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
             .split(np.zeros((okm.sum(), 1)), y))
EX = {g: i for i, g in enumerate(expr.index)}
XFULL = expr.loc[:, cols].to_numpy(dtype=np.float64).T

def score(sel):
    Mw = XFULL[:, sel]; M = Mw[okm]
    out = {}
    for C in C_GRID:
        cv = []
        for a, b in folds:
            mu, sd = M[a].mean(0), M[a].std(0) + 1e-8
            cv.append(roc_auc_score(y[b], LogisticRegression(C=C, max_iter=5000)
                      .fit((M[a]-mu)/sd, y[a]).decision_function((M[b]-mu)/sd)))
        mu, sd = Mw[tr].mean(0), Mw[tr].std(0) + 1e-8
        xw = roc_auc_score(yall[te], LogisticRegression(C=C, max_iter=5000)
                           .fit((Mw[tr]-mu)/sd, yall[tr]).decision_function((Mw[te]-mu)/sd))
        out[C] = (np.array(cv), xw)
    return out

def job(t):
    refname, npz, k = t
    pr = np.load(f'{ROOT}/data/programs/{npz}.npz', allow_pickle=True)
    pgenes = pr['genes'].astype(str); cts = list(pr['cell_types'].astype(str))
    jH = cts.index('HSC')
    rows = {}
    for arm, W in (('A', pr['W_A']), ('R', pr['W_R'])):
        s = W[:, jH]
        idx = [i for i in np.argsort(-s)[:k] if s[i] > 0]
        sel = np.array([EX[pgenes[i]] for i in idx if pgenes[i] in EX])
        rows[arm] = score(sel)
    rec = dict(reference=refname, k=k, n_A=len(rows['A'][C_GRID[0]][0]) and k)
    for C in C_GRID:
        dA, xA = rows['A'][C]; dR, xR = rows['R'][C]
        d = dA - dR
        rec[f'A_cv_{C:g}'] = dA.mean(); rec[f'R_cv_{C:g}'] = dR.mean()
        rec[f'd_cv_{C:g}'] = d.mean(); rec[f'p_{C:g}'] = wilcoxon(d).pvalue
        rec[f'd_xw_{C:g}'] = xA - xR
    return rec

if __name__ == '__main__':
    tasks = [(r, n, k) for r, n in REFS.items() for k in KS]
    with Pool(12) as p:
        recs = p.map(job, tasks)
    D = pd.DataFrame(recs)
    pd.set_option('display.width', 200)
    for r in REFS:
        print(f'\n=== {r} : HSC arm A minus arm R, by gene budget ===')
        sub = D[D.reference == r].sort_values('k')
        print(sub[['k'] + [c for c in D.columns if c.startswith(('A_cv','R_cv','d_cv','p_','d_xw'))]]
              .to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print('\n=== crossover: smallest k at which arm A beats arm R in all 4 cells ===')
    for r in REFS:
        sub = D[D.reference == r].sort_values('k')
        ok = sub[[f'd_cv_{C:g}' for C in C_GRID] + [f'd_xw_{C:g}' for C in C_GRID]].gt(0).all(1)
        ks = sub.k[ok].tolist()
        print(f'  {r:8s} all-4-positive at k = {ks if ks else "never"}')
    D.to_csv(f'{ROOT}/results/budget_sweep.csv', index=False)
    print('\nwrote results/budget_sweep.csv')
