"""Keep testing after the negative result. Four cheap, mechanistically motivated probes.

P1 clinical endpoint      : does chromatin help where the claim actually lives (pre-treatment
                            induction response) rather than on ex vivo AUC?
P2 drug-family breakdown  : is there a drug class where chromatin wiring wins?
P3 complementarity        : does A add anything ON TOP of R (A+R vs R)?
P4 chromatin weighting    : weight genes by linked-peak strength instead of selecting them.
"""
import numpy as np, pandas as pd, warnings, json
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, wilcoxon
warnings.filterwarnings('ignore')
ROOT='/home/holiday01/2026_cm'

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
link_g, link_r = pr['link_g'], pr['link_r']
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
auc  = pd.read_csv(f'{ROOT}/data/raw/beataml2/beataml_probit_curve_fits_v4_dbgap.txt', sep='\t',
                   usecols=['dbgap_rnaseq_sample','inhibitor','auc','type'])

expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
Xg = expr[cols].to_numpy(dtype=np.float32).T
gi = {g:i for i,g in enumerate(pgenes)}; rows = np.array([gi[g] for g in expr.index])
WA, WR = W_A[rows], W_R[rows]
inA, inR = (WA>0).any(1), (WR>0).any(1)
# per-gene total chromatin link strength, for P4
lw = np.zeros(len(pgenes), dtype=np.float32); np.add.at(lw, link_g, link_r); lw = lw[rows]

def comp(Z): return (Z/np.clip(np.abs(Z).sum(1,keepdims=True),1e-8,None)).astype(np.float32)
ZA, ZR = comp(Xg@WA), comp(Xg@WR)
REPS = {
 'all genes'        : Xg,
 'A programs'       : ZA,
 'R programs'       : ZR,
 'A+R programs'     : np.hstack([ZA, ZR]),
 'A gene subset'    : Xg[:, inA],
 'R gene subset'    : Xg[:, inR],
 'chromatin-weighted genes': Xg * lw[None, :],
}
sid = {s:i for i,s in enumerate(cols)}
coh = dict(zip(spec.dbgap_rnaseq_sample, spec.cohort))
tr_m = np.array([coh.get(s)=='Waves1+2' for s in cols]); te_m = np.array([coh.get(s)=='Waves3+4' for s in cols])

# ---------------- P1 clinical endpoint ----------------
lab = dict(zip(spec.dbgap_rnaseq_sample, spec.induction_response))
y = np.array([{'CR':0,'Refractory':1}.get(lab.get(s), np.nan) for s in cols], dtype=float)
ok = ~np.isnan(y)
print('P1 clinical induction response (1 = Refractory)')
print(f'  train CR/Ref {int(((y==0)&tr_m&ok).sum())}/{int(((y==1)&tr_m&ok).sum())} | '
      f'test CR/Ref {int(((y==0)&te_m&ok).sum())}/{int(((y==1)&te_m&ok).sum())}')
p1 = {}
for name, M in REPS.items():
    mu, sd = M[tr_m&ok].mean(0), M[tr_m&ok].std(0)+1e-8; Ms = (M-mu)/sd
    clf = LogisticRegressionCV(Cs=np.logspace(-3,2,8), cv=5, max_iter=5000, scoring='roc_auc')
    clf.fit(Ms[tr_m&ok], y[tr_m&ok])
    a = roc_auc_score(y[te_m&ok], clf.predict_proba(Ms[te_m&ok])[:,1])
    p1[name] = a; print(f'  {name:26s} test AUROC {a:.3f}  (features {M.shape[1]})')

# ---------------- P2/P3/P4 per-drug ex vivo ----------------
a = auc[(auc.type=='single-agent') & auc.auc.notna() & auc.dbgap_rnaseq_sample.isin(set(cols))]
per = {}
for name, M in REPS.items():
    mu, sd = M[tr_m].mean(0), M[tr_m].std(0)+1e-8; Ms = (M-mu)/sd
    rec = {}
    for drug, g in a.groupby('inhibitor'):
        g = g.drop_duplicates('dbgap_rnaseq_sample')
        idx = np.array([sid[s] for s in g.dbgap_rnaseq_sample]); yy = g.auc.to_numpy(float)
        itr = np.isin(idx, np.flatnonzero(tr_m)); ite = np.isin(idx, np.flatnonzero(te_m))
        if itr.sum()<80 or ite.sum()<40: continue
        m = RidgeCV(alphas=np.logspace(1,5,9)).fit(Ms[idx[itr]], yy[itr])
        rec[drug] = spearmanr(m.predict(Ms[idx[ite]]), yy[ite]).statistic
    per[name] = pd.Series(rec)
P = pd.DataFrame(per)
print(f'\nP3 complementarity and overall (n={len(P)} drugs, per-drug Spearman medians)')
for k in REPS: print(f'  {k:26s} {P[k].median():.3f}')
for k1,k2 in [('A+R programs','R programs'), ('A+R programs','A programs'),
              ('chromatin-weighted genes','all genes'), ('A gene subset','R gene subset')]:
    d = P[k1]-P[k2]; print(f'  {k1:26s} vs {k2:22s} median {d.median():+.4f} p={wilcoxon(P[k1],P[k2]).pvalue:.3g} wins {int((d>0).sum())}/{len(d)}')

fam = pd.read_excel(f'{ROOT}/data/raw/beataml2/beataml_drug_families.xlsx')
fmap = dict(zip(fam.synonym.astype(str), fam.family.astype(str)))
P['family'] = [fmap.get(d, 'unassigned') for d in P.index]
P['A_minus_R'] = P['A gene subset'] - P['R gene subset']
print('\nP2 drug families where the chromatin wiring wins (>=5 drugs)')
fb = P.groupby('family').agg(n=('A_minus_R','size'), med=('A_minus_R','median'),
                             wins=('A_minus_R', lambda s: int((s>0).sum())))
fb = fb[fb.n>=5].sort_values('med', ascending=False)
print(fb.to_string())
P.to_csv(f'{ROOT}/results/probe_per_drug.csv')
json.dump({'P1_clinical_auroc': p1, 'P3_medians': {k: float(P[k].median()) for k in REPS}},
          open(f'{ROOT}/results/probe_summary.json','w'), indent=2)
print('\nwrote results/probe_per_drug.csv and probe_summary.json')
