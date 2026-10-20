"""Is the clinical-endpoint advantage of chromatin-informed features real?
Repeated stratified CV over all labelled Initial-Diagnosis specimens, paired across folds,
plus the cross-wave split as an external check. Also fixes the drug-family mapping.
"""
import numpy as np, pandas as pd, warnings, json, re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
warnings.filterwarnings('ignore')
ROOT='/home/holiday01/2026_cm'

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
link_g, link_r = pr['link_g'], pr['link_r']
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')

expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
Xg = expr[cols].to_numpy(dtype=np.float32).T
gi = {g:i for i,g in enumerate(pgenes)}; rows = np.array([gi[g] for g in expr.index])
WA, WR = W_A[rows], W_R[rows]
inA, inR = (WA>0).any(1), (WR>0).any(1)
lw = np.zeros(len(pgenes), np.float32); np.add.at(lw, link_g, link_r); lw = lw[rows]
# a matched RNA weighting: gene-level expression specificity magnitude, same scale treatment
rw = np.abs(W_R[rows]).sum(1)

def comp(Z): return (Z/np.clip(np.abs(Z).sum(1,keepdims=True),1e-8,None)).astype(np.float32)
REPS = {
 'all genes'                : Xg,
 'A gene subset'            : Xg[:, inA],
 'R gene subset'            : Xg[:, inR],
 'chromatin-weighted genes' : Xg * lw[None,:],
 'RNA-weighted genes'       : Xg * rw[None,:],
 'shuffled-link weights'    : Xg * np.random.default_rng(0).permutation(lw)[None,:],
}
lab = dict(zip(spec.dbgap_rnaseq_sample, spec.induction_response))
y = np.array([{'CR':0,'Refractory':1}.get(lab.get(s), -1) for s in cols])
ok = y >= 0; y = y[ok]
coh = np.array([spec.set_index('dbgap_rnaseq_sample').cohort.get(s) for s in cols])[ok]
print(f'labelled specimens {ok.sum()}  CR {int((y==0).sum())}  Refractory {int((y==1).sum())}')

C_GRID = np.logspace(-3, 0, 4)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
scores = {k: [] for k in REPS}
for name, M in REPS.items():
    M = M[ok]
    for tr, te in cv.split(M, y):
        mu, sd = M[tr].mean(0), M[tr].std(0)+1e-8
        Xt, Xv = (M[tr]-mu)/sd, (M[te]-mu)/sd
        best, bs = None, -1
        for C in C_GRID:                                  # tiny inner grid on the training fold
            clf = LogisticRegression(C=C, max_iter=2000).fit(Xt, y[tr])
            s = roc_auc_score(y[tr], clf.decision_function(Xt))
            if s > bs: bs, best = s, clf
        scores[name].append(roc_auc_score(y[te], best.decision_function(Xv)))
    a = np.array(scores[name]); lo, hi = np.percentile(a, [2.5, 97.5])
    print(f'  {name:26s} AUROC mean {a.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (100 folds, {M.shape[1]} features)')

print('\npaired across the same 100 folds (Wilcoxon):')
def cmp(k1,k2):
    d = np.array(scores[k1]) - np.array(scores[k2])
    print(f'  {k1:26s} vs {k2:24s} mean diff {d.mean():+.4f}  p={wilcoxon(d).pvalue:.3g}  wins {int((d>0).sum())}/{len(d)}')
cmp('chromatin-weighted genes','all genes')
cmp('chromatin-weighted genes','RNA-weighted genes')
cmp('chromatin-weighted genes','shuffled-link weights')
cmp('A gene subset','R gene subset')
cmp('A gene subset','all genes')

json.dump({k: {'mean': float(np.mean(v)), 'ci': [float(np.percentile(v,2.5)), float(np.percentile(v,97.5))]}
           for k,v in scores.items()}, open(f'{ROOT}/results/clinical_cv.json','w'), indent=2)

# ---- fix the drug-family mapping ----
fam = pd.read_excel(f'{ROOT}/data/raw/beataml2/beataml_drug_families.xlsx')
def norm(s): return re.sub(r'[^a-z0-9]+','', re.sub(r'\([^)]*\)',' ', str(s).lower()))
fmap = {}
for f_, syn in zip(fam.family.astype(str), fam.synonym.astype(str)):
    for part in re.split(r'[;,/]| and ', syn):
        k = norm(part)
        if k: fmap[k] = f_
P = pd.read_csv(f'{ROOT}/results/probe_per_drug.csv', index_col=0)
def lookup(d):
    cands = [norm(d)] + [norm(p) for p in re.split(r'[;/]', d)]
    for c in cands:
        if c in fmap: return fmap[c]
    for k, v in fmap.items():                            # substring fallback
        if k and (k in cands[0] or cands[0] in k): return v
    return 'unassigned'
P['family'] = [lookup(d) for d in P.index]
print(f"\ndrug-family mapping: {int((P.family!='unassigned').sum())}/{len(P)} drugs assigned, "
      f"{P.family.nunique()-1} families")
P['A_minus_R'] = P['A gene subset'] - P['R gene subset']
fb = P.groupby('family').agg(n=('A_minus_R','size'), med=('A_minus_R','median'),
                             wins=('A_minus_R', lambda s: int((s>0).sum())))
print(fb[fb.n>=4].sort_values('med', ascending=False).to_string())
P.to_csv(f'{ROOT}/results/probe_per_drug.csv')
