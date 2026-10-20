"""Second endpoint: does the A-vs-R result hold for overall survival?

Exactly the design of eval_programs.py (same references, same programs, same 200-gene
budget, same variance-matched controls, same cross-wave split) with the induction-response
logistic model replaced by a ridge-penalised Cox model and AUROC by Harrell's C.

The theory sentence says chromatin readability decides whether a program carries
patient-level PROGNOSTIC information. Induction response is one instance of that. If OS
agrees, the claim stands as written; if it does not, the claim narrows to induction
response, which is itself a result worth reporting.

Usage: eval_os.py <programs.npz> <label>
"""
import os, sys, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from scipy.stats import wilcoxon
from multiprocessing import Pool
warnings.filterwarnings('ignore')
NPZ, LABEL = sys.argv[1], sys.argv[2]
ROOT = '/home/holiday01/2026_cm'
ALPHAS = [float(x) for x in (os.environ.get('OS_ALPHAS') or '1.0,0.1').split(',')]
# Only the 200-gene program arms are run. CoxPH builds and inverts a dense p x p Hessian,
# so a 1620- or 2041-gene arm is ~1000x slower per fit and starves the pool. This is also
# the right scope: S17 already deleted the subset-level claims (chromatin subset as a better
# overall predictor, HSC as best program, HSC beating all genes), so the only claim that
# needs a second endpoint is the one that survived -- HSC arm A vs arm R at a fixed
# 200-gene budget, plus the per-program lineage split and the variance-matched controls.
PMAX = int(os.environ.get('OS_PMAX', 250))
NRAND = 20
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
tmap = pd.to_numeric(S.overallSurvival, errors='coerce')
vmap = S.vitalStatus.astype(str).str.lower()
cmap = S.cohort.to_dict()
time_all = np.array([tmap.get(s, np.nan) for s in cols], float)
vit_all  = np.array([vmap.get(s, 'unknown') for s in cols])
ev_all   = vit_all == 'dead'
cw       = np.array([str(cmap.get(s)) for s in cols])
okm = np.isfinite(time_all) & (time_all > 0) & np.isin(vit_all, ['dead', 'alive'])
T, E = time_all[okm], ev_all[okm]
X = Xall[okm]
tr = okm & (cw == 'Waves1+2'); te = okm & (cw == 'Waves3+4')
print(f'=== {LABEL} | endpoint = overall survival ===')
print(f'{len(cts)} programs | A subset {nA} | R subset {len(inR)} | all {X.shape[1]} genes')
print(f'CV set {okm.sum()} (events {int(E.sum())}, censored {int((~E).sum())}, '
      f'median OS {np.median(T):.0f} d) | xwave train {tr.sum()} (ev {int(ev_all[tr].sum())}) '
      f'-> test {te.sum()} (ev {int(ev_all[te].sum())})')
print(f'alphas {ALPHAS} | PMAX {PMAX}')
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

ARMS = {'A subset': inA, 'R subset': inR, 'R size-matched': np.argsort(-rw)[:nA],
        f'top-{nA} by variance': np.argsort(-var)[:nA],
        'HSC chromatin': hscA, 'HSC RNA': hscR}
for i, m in enumerate(matched(inA, 10)): ARMS[f'sub-rand #{i}'] = m
for i, m in enumerate(matched(hscA, NRAND)): ARMS[f'hsc-rand #{i}'] = m
for j, c in enumerate(cts):
    ARMS[f'prog-A: {c}'] = np.flatnonzero(WA[:, j])
    ARMS[f'prog-R: {c}'] = np.flatnonzero(WR[:, j])
ARMS = {k: v for k, v in ARMS.items() if len(v) <= PMAX and len(v) > 0}
# stratify the folds on the event indicator so every fold has deaths to rank
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0).split(X, E))

def cidx(e, t, p):
    return concordance_index_censored(np.asarray(e, bool), t, p)[0]

def run(item):
    name, sel = item
    t0 = __import__('time').time()
    M, Mw = X[:, sel], Xall[:, sel]
    cvs = {a: [] for a in ALPHAS}
    for ia, ib in folds:
        mu, sd = M[ia].mean(0), M[ia].std(0) + 1e-8
        Xt, Xv = (M[ia] - mu) / sd, (M[ib] - mu) / sd
        ys = Surv.from_arrays(E[ia], T[ia])
        for a in ALPHAS:
            try:
                p = CoxPHSurvivalAnalysis(alpha=a).fit(Xt, ys).predict(Xv)
                cvs[a].append(cidx(E[ib], T[ib], p))
            except Exception:
                cvs[a].append(np.nan)
    mu, sd = Mw[tr].mean(0), Mw[tr].std(0) + 1e-8
    Xt, Xv = (Mw[tr] - mu) / sd, (Mw[te] - mu) / sd
    ys = Surv.from_arrays(ev_all[tr], time_all[tr])
    xw = {}
    for a in ALPHAS:
        try:
            p = CoxPHSurvivalAnalysis(alpha=a).fit(Xt, ys).predict(Xv)
            xw[a] = cidx(ev_all[te], time_all[te], p)
        except Exception:
            xw[a] = np.nan
    print(f'  [{name}] p={len(sel)} {__import__("time").time()-t0:.1f}s', flush=True)
    return name, {a: np.array(v) for a, v in cvs.items()}, xw

if __name__ == '__main__':
    with Pool(22) as p:
        out = p.map(run, list(ARMS.items()))
    cv = {n: a for n, a, _ in out}; xw = {n: b for n, _, b in out}
    print(f'\n{"arm":24s}' + ''.join(f'{f"CV a={a:g}":>12s}{"xwave":>9s}' for a in ALPHAS))
    for n in ['A subset', 'R subset', 'R size-matched', f'top-{nA} by variance',
              'HSC chromatin', 'HSC RNA']:
        if n in cv:
            print(f'  {n:22s}' + ''.join(f'{np.nanmean(cv[n][a]):12.3f}{xw[n][a]:9.3f}' for a in ALPHAS))
    def line(k1, k2, tag):
        if k1 not in cv or k2 not in cv: return
        for a in ALPHAS:
            d = cv[k1][a] - cv[k2][a]; d = d[np.isfinite(d)]
            pv = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
            print(f'  a={a:g} {tag:34s} CV {d.mean():+.4f} p={pv:9.3g} '
                  f'wins {int((d>0).sum()):3d}/{len(d)} | xwave {xw[k1][a]-xw[k2][a]:+.4f}')
    print('\npaired comparisons:')
    line('HSC chromatin', 'HSC RNA', 'HSC chromatin vs HSC RNA')
    line('A subset', 'R size-matched', 'A subset vs R size-matched')
    line('A subset', f'top-{nA} by variance', 'A subset vs top-by-variance')
    print('\nvs variance-matched random draws:')
    for key, pref, n in (('A subset', 'sub-rand', 10), ('HSC chromatin', 'hsc-rand', NRAND)):
        rk = [k for k in ARMS if k.startswith(pref)]
        if key not in cv or not rk: continue
        for a in ALPHAS:
            v = cv[key][a]; ms = [np.nanmean(cv[k][a]) for k in rk]
            w = sum(1 for k in rk if np.nanmean(v - cv[k][a]) > 0
                    and wilcoxon((v - cv[k][a])[np.isfinite(v - cv[k][a])]).pvalue < 0.05)
            axw = xw[key][a]; mxw = [xw[k][a] for k in rk]
            print(f'  a={a:g} {key:16s} CV {np.nanmean(v):.3f} vs {np.mean(ms):.3f}+/-{np.std(ms):.3f} '
                  f'(beats {w}/{len(rk)}) | xwave {axw:.3f} vs {np.mean(mxw):.3f}+/-{np.std(mxw):.3f} '
                  f'(above {sum(1 for m in mxw if axw>m)}/{len(rk)})')
    print('\nHSC rank among this reference\'s chromatin programs:')
    pk = [f'prog-A: {c}' for c in cts if f'prog-A: {c}' in cv]
    for a in ALPHAS:
        v = np.nanmean(cv['HSC chromatin'][a])
        ms = sorted((np.nanmean(cv[k][a]) for k in pk), reverse=True)
        axw = xw['HSC chromatin'][a]; mxw = sorted((xw[k][a] for k in pk), reverse=True)
        print(f'  a={a:g}  CV {v:.3f} rank {ms.index(v)+1}/{len(pk)} (best {ms[0]:.3f}) | '
              f'xwave {axw:.3f} rank {mxw.index(axw)+1}/{len(pk)} (best {mxw[0]:.3f})')
    print('\nper-program A-minus-R, by lineage:')
    recs = []
    for c in cts:
        ka, kr = f'prog-A: {c}', f'prog-R: {c}'
        if ka not in cv or kr not in cv: continue
        r = dict(program=c, lineage='myeloid' if c in MYE else 'lymphoid' if c in LYM else 'other')
        for a in ALPHAS:
            d = cv[ka][a] - cv[kr][a]; d = d[np.isfinite(d)]
            r[f'cv_{a:g}'] = d.mean()
            r[f'p_{a:g}'] = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
            r[f'xw_{a:g}'] = xw[ka][a] - xw[kr][a]
        recs.append(r)
    D = pd.DataFrame(recs).sort_values(f'cv_{ALPHAS[-1]:g}', ascending=False)
    print(D.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    for a in ALPHAS:
        m = D.loc[D.lineage == 'myeloid', f'cv_{a:g}']; l = D.loc[D.lineage == 'lymphoid', f'cv_{a:g}']
        mx = D.loc[D.lineage == 'myeloid', f'xw_{a:g}']; lx = D.loc[D.lineage == 'lymphoid', f'xw_{a:g}']
        allv = np.concatenate([m.values, l.values]); nm = len(m)
        perm = np.array([(lambda q: q[:nm].mean() - q[nm:].mean())(rng.permutation(allv))
                         for _ in range(20000)])
        obs = m.mean() - l.mean()
        print(f'  a={a:g}  CV myeloid {m.mean():+.4f} (n={nm}) vs lymphoid {l.mean():+.4f} '
              f'(n={len(l)}) gap {obs:+.4f} perm p={max((np.abs(perm)>=abs(obs)).mean(),1/20000):.4g}'
              f' | xwave gap {mx.mean()-lx.mean():+.4f}')
    tag = os.path.basename(NPZ).replace('.npz', '')
    D.to_csv(f'{ROOT}/results/os_{tag}_per_program.csv', index=False)
    json.dump({'endpoint': 'overall_survival', 'alphas': ALPHAS,
               'n': int(okm.sum()), 'events': int(E.sum()),
               'cv': {n: {str(a): float(np.nanmean(v)) for a, v in d.items()} for n, d in cv.items()},
               'xwave': {n: {str(a): float(v) for a, v in d.items()} for n, d in xw.items()}},
              open(f'{ROOT}/results/os_{tag}.json', 'w'), indent=2)
    print(f'\nwrote results/os_{tag}.json and results/os_{tag}_per_program.csv')
