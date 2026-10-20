"""The comparison that matters: a chromatin-selected gene set vs an EXPERIMENTALLY DERIVED
LSC gene set at the same budget.

Extended Data Table 1 of Ng et al. 2016 (doi:10.1038/nature20598) lists 104 genes
differentially expressed between 138 LSC+ and 89 LSC- sorted fractions from 78 AML
patients, validated by xenotransplantation. That set is ~104 genes, and S19.2 found our
chromatin rule only beats the RNA rule at a budget of >=100 -- so this is a budget-matched
contest between a computational selection rule and an experimental one, on the same
patients, the same programs, the same folds and the same classifier.

Input: a text file with one gene symbol per line (blank lines and lines starting with # are
ignored; a comma- or tab-separated single line also works). Default path:
  data/ref/published_signatures/ng2016_lsc104.txt

Beat AML uses microarray-era symbols, so aliases are mapped before matching and every
unmatched symbol is reported rather than silently dropped -- a silent drop would shrink the
published set and flatter us.

Usage: baseline_lsc104.py [genelist.txt] [programs.npz]
"""
import os, sys, json, re
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
LIST = sys.argv[1] if len(sys.argv) > 1 else f'{ROOT}/data/ref/published_signatures/ng2016_lsc104.txt'
NPZ  = sys.argv[2] if len(sys.argv) > 2 else f'{ROOT}/data/programs/programs.npz'
C_GRID = [1e-2, 1e-1]
NRAND = 20
# new symbol -> the symbol Beat AML's annotation uses
ALIAS = {'LOC284422':'C19orf77','SMIM24':'C19orf77','FAM30A':'KIAA0125','SMIM24':'C19orf77','BEX3':'NGFRAP1','ADGRG1':'GPR56',
         'MACROH2A1':'H2AFY','H2AZ1':'H2AFZ','ATP5F1B':'ATP5B','SEPTIN6':'SEPT6',
         'CAVIN1':'PTRF','CAVIN2':'SDPR','PLPP3':'PPAP2B','CARMIL1':'LRRC16A',
         'DGLUCY':'C14orf159','MINDY2':'FAM63B','TENT5A':'FAM46A','EOLA1':'CXorf40A'}

if not os.path.exists(LIST):
    sys.exit(f'gene list not found: {LIST}\n'
             'Save Extended Data Table 1 of Ng et al. 2016 as one gene symbol per line.')
raw = open(LIST).read()
toks = [t.strip().upper() for t in re.split(r'[\s,;]+', raw) if t.strip() and not t.startswith('#')]
toks = list(dict.fromkeys(toks))
print(f'read {len(toks)} unique symbols from {os.path.basename(LIST)}')

expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr.drop_duplicates('display_label').set_index('display_label')
idx_up = {g.upper(): g for g in expr.index}
mapped, missing = [], []
for t in toks:
    g = idx_up.get(t) or idx_up.get(ALIAS.get(t, '').upper())
    (mapped if g else missing).append(g or t)
mapped = list(dict.fromkeys(mapped))
print(f'matched {len(mapped)}/{len(toks)} in the Beat AML matrix')
if missing:
    print(f'UNMATCHED ({len(missing)}) -- check these before quoting any number: {missing}')

cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
S = spec.set_index('dbgap_rnaseq_sample')
lab, coh = S.induction_response.to_dict(), S.cohort.to_dict()
yall = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
cw = np.array([str(coh.get(s)) for s in cols])
okm = yall >= 0; y = yall[okm]
tr, te = okm & (cw == 'Waves1+2'), okm & (cw == 'Waves3+4')
XFULL = expr.loc[:, cols].to_numpy(dtype=np.float64).T
EX = {g: i for i, g in enumerate(expr.index)}
folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
             .split(np.zeros((okm.sum(), 1)), y))

def score(sel):
    Mw = XFULL[:, sel]; M = Mw[okm]; out = {}
    for C in C_GRID:
        cv = []
        for a, b in folds:
            mu, sd = M[a].mean(0), M[a].std(0) + 1e-8
            cv.append(roc_auc_score(y[b], LogisticRegression(C=C, max_iter=5000)
                      .fit((M[a]-mu)/sd, y[a]).decision_function((M[b]-mu)/sd)))
        mu, sd = Mw[tr].mean(0), Mw[tr].std(0) + 1e-8
        out[C] = (np.array(cv), roc_auc_score(yall[te], LogisticRegression(C=C, max_iter=5000)
                  .fit((Mw[tr]-mu)/sd, yall[tr]).decision_function((Mw[te]-mu)/sd)))
    return out

K = len(mapped)                      # match our budget to the published set's size exactly
pr = np.load(NPZ, allow_pickle=True)
pgenes = pr['genes'].astype(str); cts = list(pr['cell_types'].astype(str))
jH = cts.index('HSC')
def top_k(W, k):
    s = W[:, jH]
    return [EX[pgenes[i]] for i in np.argsort(-s)[:k] if s[i] > 0 and pgenes[i] in EX]
pub = np.array([EX[g] for g in mapped])
ARMS = {f'Ng2016 LSC genes (published, experimental)': pub,
        f'HSC arm A top-{K} (chromatin)': np.array(top_k(pr['W_A'], K)),
        f'HSC arm R top-{K} (RNA)':       np.array(top_k(pr['W_R'], K))}
var = XFULL.var(0)
dec = pd.qcut(pd.Series(var), 10, labels=False, duplicates='drop').to_numpy()
rng = np.random.default_rng(11)
need = pd.Series(dec[pub]).value_counts().to_dict()
for i in range(NRAND):
    pick = []
    for d, k in need.items():
        pool = np.setdiff1d(np.where(dec == d)[0], pub)
        pick.append(rng.choice(pool, min(k, len(pool)), replace=False))
    ARMS[f'var-matched random #{i}'] = np.concatenate(pick)
ARMS[f'top-{K} by variance'] = np.argsort(-var)[:K]

res = {n: score(s) for n, s in ARMS.items()}
print(f'\nbudget matched at K={K} genes\n')
print(f'{"arm":40s}' + ''.join(f'{f"CV C={C:g}":>12s}{"xwave":>9s}' for C in C_GRID))
for n in ARMS:
    if n.startswith('var-matched'): continue
    print(f'  {n:38s}' + ''.join(f'{res[n][C][0].mean():12.3f}{res[n][C][1]:9.3f}' for C in C_GRID))
rk = [n for n in ARMS if n.startswith('var-matched')]
print(f'  {"var-matched random (mean of 20)":38s}' +
      ''.join(f'{np.mean([res[k][C][0].mean() for k in rk]):12.3f}'
              f'{np.mean([res[k][C][1] for k in rk]):9.3f}' for C in C_GRID))
print('\npaired comparisons (100 folds):')
pubk = 'Ng2016 LSC genes (published, experimental)'
for k1 in (f'HSC arm A top-{K} (chromatin)', f'HSC arm R top-{K} (RNA)'):
    for C in C_GRID:
        d = res[k1][C][0] - res[pubk][C][0]
        print(f'  C={C:g}  {k1} vs published\n'
              f'        CV {d.mean():+.4f} p={wilcoxon(d).pvalue:9.3g} wins {int((d>0).sum())}/100'
              f' | xwave {res[k1][C][1]-res[pubk][C][1]:+.4f}')
for C in C_GRID:
    d = res[f'HSC arm A top-{K} (chromatin)'][C][0] - res[f'HSC arm R top-{K} (RNA)'][C][0]
    print(f'  C={C:g}  arm A vs arm R at K={K}: CV {d.mean():+.4f} p={wilcoxon(d).pvalue:9.3g}'
          f' | xwave {res[f"HSC arm A top-{K} (chromatin)"][C][1]-res[f"HSC arm R top-{K} (RNA)"][C][1]:+.4f}')
print(f'\noverlap: published set vs HSC arm A top-{K}: '
      f'{len(set(pub) & set(ARMS[f"HSC arm A top-{K} (chromatin)"]))}/{K}'
      f' | vs arm R: {len(set(pub) & set(ARMS[f"HSC arm R top-{K} (RNA)"]))}/{K}')
json.dump({'source': 'Ng et al. Nature 2016 Extended Data Table 1', 'K': K,
           'n_input': len(toks), 'unmatched': missing, 'genes_used': mapped,
           'cv': {n: {str(C): float(res[n][C][0].mean()) for C in C_GRID} for n in res},
           'xwave': {n: {str(C): float(res[n][C][1]) for C in C_GRID} for n in res}},
          open(f'{ROOT}/results/baseline_lsc104.json', 'w'), indent=2)
print('\nwrote results/baseline_lsc104.json')
