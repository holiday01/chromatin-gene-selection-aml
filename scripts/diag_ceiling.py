"""Diagnostic: is there ANY patient-specific signal in bulk expression for this endpoint?
If ridge on all 11,319 genes cannot beat the lookup floor either, then no choice of
first-layer wiring can, and the bottleneck is the endpoint, not the chromatin question.
"""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
ROOT='/home/holiday01/2026_cm'

expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
auc  = pd.read_csv(f'{ROOT}/data/raw/beataml2/beataml_probit_curve_fits_v4_dbgap.txt', sep='\t',
                   usecols=['dbgap_rnaseq_sample','inhibitor','auc','type'])
pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
keep = set(pr['genes'].astype(str))

expr = expr[expr.display_label.isin(keep)].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
X = expr[cols].to_numpy(dtype=np.float32).T
sid = {s:i for i,s in enumerate(cols)}
coh = dict(zip(spec.dbgap_rnaseq_sample, spec.cohort))
tr_m = np.array([coh.get(s)=='Waves1+2' for s in cols]); te_m = np.array([coh.get(s)=='Waves3+4' for s in cols])
mu, sd = X[tr_m].mean(0), X[tr_m].std(0)+1e-8
Xs = (X-mu)/sd
print(f'genes {X.shape[1]} | train {tr_m.sum()} | test {te_m.sum()}')

a = auc[(auc.type=='single-agent') & auc.auc.notna() & auc.dbgap_rnaseq_sample.isin(set(cols))]
rows=[]
for drug, g in a.groupby('inhibitor'):
    g = g.drop_duplicates('dbgap_rnaseq_sample')
    idx = np.array([sid[s] for s in g.dbgap_rnaseq_sample]); y = g.auc.to_numpy(float)
    itr = np.isin(idx, np.flatnonzero(tr_m)); ite = np.isin(idx, np.flatnonzero(te_m))
    if itr.sum() < 80 or ite.sum() < 40: continue
    Xtr, ytr, Xte, yte = Xs[idx[itr]], y[itr], Xs[idx[ite]], y[ite]
    m = RidgeCV(alphas=np.logspace(1,5,9)).fit(Xtr, ytr)
    p = m.predict(Xte)
    base = np.full_like(yte, ytr.mean())                       # per-drug mean = the floor for AUC
    rows.append({'drug': drug, 'n_tr': int(itr.sum()), 'n_te': int(ite.sum()),
                 'spearman': spearmanr(p, yte).statistic,
                 'rmse_ridge': float(np.sqrt(((p-yte)**2).mean())),
                 'rmse_floor': float(np.sqrt(((base-yte)**2).mean()))})
r = pd.DataFrame(rows)
r['beats_floor'] = r.rmse_ridge < r.rmse_floor
print(f'\nridge on all {X.shape[1]} genes, per-drug AUC, {len(r)} drugs with enough samples:')
print(f'  Spearman   mean {r.spearman.mean():.3f}  median {r.spearman.median():.3f}  frac>0 {(r.spearman>0).mean()*100:.0f}%')
print(f'  beats the per-drug-mean floor on RMSE: {r.beats_floor.mean()*100:.0f}% of drugs')
print(f'  drugs with Spearman > 0.3: {(r.spearman>0.3).sum()} | > 0.2: {(r.spearman>0.2).sum()}')
print('\ntop 8 drugs by Spearman:')
print(r.nlargest(8,'spearman')[['drug','n_tr','n_te','spearman','rmse_ridge','rmse_floor']].to_string(index=False))
r.to_csv(f'{ROOT}/results/diag_ridge_per_drug.csv', index=False)
print(f'\nwrote results/diag_ridge_per_drug.csv')
