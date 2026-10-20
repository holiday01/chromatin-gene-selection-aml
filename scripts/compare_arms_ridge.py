"""The chromatin question asked at equal, strong model capacity: the SAME per-drug ridge,
fed five different representations of the same bulk expression vector.
"""
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr, wilcoxon
warnings.filterwarnings('ignore')
ROOT='/home/holiday01/2026_cm'

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
auc  = pd.read_csv(f'{ROOT}/data/raw/beataml2/beataml_probit_curve_fits_v4_dbgap.txt', sep='\t',
                   usecols=['dbgap_rnaseq_sample','inhibitor','auc','type'])

expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
Xg   = expr[cols].to_numpy(dtype=np.float32).T
gi   = {g:i for i,g in enumerate(pgenes)}
rows = np.array([gi[g] for g in expr.index])
WA, WR = W_A[rows], W_R[rows]
inA = (WA > 0).any(1); inR = (WR > 0).any(1)
print(f'genes {Xg.shape[1]} | in any chromatin program {inA.sum()} | in any RNA program {inR.sum()} '
      f'| overlap {int((inA & inR).sum())}')

def comp(Z):
    return (Z / np.clip(np.abs(Z).sum(1, keepdims=True), 1e-8, None)).astype(np.float32)
REPS = {
 'all genes'              : Xg,
 'A: 22 chromatin programs': comp(Xg @ WA),
 'R: 22 RNA programs'      : comp(Xg @ WR),
 'A: chromatin gene subset': Xg[:, inA],
 'R: RNA gene subset'      : Xg[:, inR],
}
sid = {s:i for i,s in enumerate(cols)}
coh = dict(zip(spec.dbgap_rnaseq_sample, spec.cohort))
tr_m = np.array([coh.get(s)=='Waves1+2' for s in cols]); te_m = np.array([coh.get(s)=='Waves3+4' for s in cols])
a = auc[(auc.type=='single-agent') & auc.auc.notna() & auc.dbgap_rnaseq_sample.isin(set(cols))]

out = {}
for name, M in REPS.items():
    mu, sd = M[tr_m].mean(0), M[tr_m].std(0)+1e-8
    Ms = (M-mu)/sd
    rec = []
    for drug, g in a.groupby('inhibitor'):
        g = g.drop_duplicates('dbgap_rnaseq_sample')
        idx = np.array([sid[s] for s in g.dbgap_rnaseq_sample]); y = g.auc.to_numpy(float)
        itr = np.isin(idx, np.flatnonzero(tr_m)); ite = np.isin(idx, np.flatnonzero(te_m))
        if itr.sum() < 80 or ite.sum() < 40: continue
        m = RidgeCV(alphas=np.logspace(1,5,9)).fit(Ms[idx[itr]], y[itr])
        p = m.predict(Ms[idx[ite]])
        rec.append({'drug': drug, 'spearman': spearmanr(p, y[ite]).statistic,
                    'rmse': float(np.sqrt(((p-y[ite])**2).mean())),
                    'rmse_floor': float(np.sqrt(((y[itr].mean()-y[ite])**2).mean()))})
    d = pd.DataFrame(rec); out[name] = d
    print(f'  {name:26s} n={len(d):3d}  Spearman median {d.spearman.median():.3f} mean {d.spearman.mean():.3f} '
          f'| frac>0 {(d.spearman>0).mean()*100:3.0f}%  | beats floor {(d.rmse<d.rmse_floor).mean()*100:3.0f}%  | feat {M.shape[1]}')

print('\npaired comparisons over the same drugs (Wilcoxon on per-drug Spearman):')
def cmp(k1, k2):
    m = out[k1].merge(out[k2], on='drug', suffixes=('_1','_2'))
    s = wilcoxon(m.spearman_1, m.spearman_2)
    d = (m.spearman_1 - m.spearman_2)
    print(f'  {k1:26s} vs {k2:26s}  median diff {d.median():+.4f}  mean {d.mean():+.4f}  p={s.pvalue:.3g}  n={len(m)}  wins {int((d>0).sum())}/{len(m)}')
cmp('A: 22 chromatin programs', 'R: 22 RNA programs')
cmp('A: chromatin gene subset', 'R: RNA gene subset')
cmp('all genes', 'A: 22 chromatin programs')
cmp('all genes', 'A: chromatin gene subset')
for k, d in out.items(): d.to_csv(f"{ROOT}/results/ridge_{k.replace(':','').replace(' ','_')}.csv", index=False)
print('\nwrote per-representation per-drug tables to results/')
