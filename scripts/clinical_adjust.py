"""The confound that would sink the HSC result: refractory patients carry more blasts, so a
stem/progenitor gene program scored on bulk RNA-seq could just be a blast-fraction proxy.

Nested CV: inside each training fold the gene model is fitted and its score is read out on
train and test, then a second-stage logistic on [standard clinical covariates, gene score]
is fitted on train only. So the gene score is always out-of-sample when it is evaluated.
Covariates: age, sex, % blasts in marrow, WBC, FLT3-ITD, NPM1.
"""
import os, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, spearmanr
warnings.filterwarnings('ignore')
ROOT = '/home/holiday01/2026_cm'
C_GRID = [1e-2, 1e-1]

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
W_A, W_R, pgenes = pr['W_A'], pr['W_R'], pr['genes'].astype(str)
cts = list(pr['cell_types'].astype(str)); jH = cts.index('HSC')
expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr[expr.display_label.isin(set(pgenes))].drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
X = expr[cols].to_numpy(dtype=np.float64).T
gi = {g: i for i, g in enumerate(pgenes)}
rows = np.array([gi[g] for g in expr.index])
hscA = np.flatnonzero(W_A[rows][:, jH]); hscR = np.flatnonzero(W_R[rows][:, jH])

cl = pd.read_excel(f'{ROOT}/data/raw/beataml2/beataml_wv1to4_clinical.xlsx')
cl = cl.drop_duplicates('dbgap_rnaseq_sample').set_index('dbgap_rnaseq_sample')
def num(s): return pd.to_numeric(s, errors='coerce')
C = pd.DataFrame(index=cols)
C['age'] = num(cl.reindex(cols).ageAtDiagnosis)
C['male'] = (cl.reindex(cols).consensus_sex == 'Male').astype(float)
C['blast_bm'] = num(cl.reindex(cols)['%.Blasts.in.BM'])
C['wbc'] = np.log1p(num(cl.reindex(cols).wbcCount))
C['flt3'] = (cl.reindex(cols)['FLT3-ITD'].astype(str).str.lower() == 'positive').astype(float)
C['npm1'] = (cl.reindex(cols)['NPM1'].astype(str).str.lower() == 'positive').astype(float)
lab = dict(zip(spec.dbgap_rnaseq_sample, spec.induction_response))
y = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])
print('covariate coverage among labelled specimens:')
for c_ in C.columns:
    print(f'  {c_:10s} {int(C.loc[np.array(cols)[y>=0], c_].notna().sum())}/{int((y>=0).sum())}')
ok = (y >= 0) & C.notna().all(1).to_numpy()
print(f'complete cases: {ok.sum()} (CR {int((y[ok]==0).sum())} / Ref {int((y[ok]==1).sum())})')
Xo, Co, yo = X[ok], C[ok.tolist()].to_numpy(dtype=float), y[ok]

print('\nis the HSC chromatin score just a blast-fraction readout?')
for nm, sel in (('chromatin', hscA), ('RNA', hscR)):
    M = (Xo[:, sel] - Xo[:, sel].mean(0)) / (Xo[:, sel].std(0) + 1e-8)
    s = LogisticRegression(C=0.01, max_iter=5000).fit(M, yo).decision_function(M)  # in-sample, for correlation only
    r = spearmanr(s, Co[:, 2])
    print(f'  HSC {nm:9s} score vs % blasts in marrow: rho {r.statistic:+.3f}  p={r.pvalue:.3g}')
print(f'  % blasts alone vs outcome: AUROC '
      f'{roc_auc_score(yo, Co[:, 2]):.3f}')

folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0).split(Xo, yo))
def nested(sel, C_gene, with_clin=True, gene=True):
    aucs = []
    for tr, te in folds:
        feats_tr, feats_te = [], []
        if with_clin:
            mu, sd = Co[tr].mean(0), Co[tr].std(0) + 1e-8
            feats_tr.append((Co[tr] - mu) / sd); feats_te.append((Co[te] - mu) / sd)
        if gene:
            M = Xo[:, sel]
            mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-8
            Mt, Mv = (M[tr] - mu) / sd, (M[te] - mu) / sd
            g = LogisticRegression(C=C_gene, max_iter=5000).fit(Mt, yo[tr])
            feats_tr.append(g.decision_function(Mt)[:, None])
            feats_te.append(g.decision_function(Mv)[:, None])
        Ft, Fv = np.hstack(feats_tr), np.hstack(feats_te)
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(Ft, yo[tr])
        aucs.append(roc_auc_score(yo[te], clf.decision_function(Fv)))
    return np.array(aucs)

base = nested(None, None, with_clin=True, gene=False)
print(f'\nnested CV, {len(folds)} folds, complete cases only:')
print(f'  clinical covariates only                 {base.mean():.3f}')
res = {'clinical only': base}
for C_gene in C_GRID:
    for nm, sel in (('HSC chromatin', hscA), ('HSC RNA', hscR)):
        g = nested(sel, C_gene, with_clin=False)
        cg = nested(sel, C_gene, with_clin=True)
        res[f'{nm} alone C={C_gene:g}'] = g
        res[f'clinical + {nm} C={C_gene:g}'] = cg
        d = cg - base
        print(f'  {nm} alone (C={C_gene:g})               {g.mean():.3f}')
        print(f'  clinical + {nm} (C={C_gene:g})          {cg.mean():.3f}   '
              f'gain over clinical {d.mean():+.4f}  p={wilcoxon(d).pvalue:.3g}  '
              f'wins {int((d>0).sum())}/{len(d)}')
print('\ndoes chromatin still beat RNA after the clinical covariates are in the model?')
for C_gene in C_GRID:
    d = res[f'clinical + HSC chromatin C={C_gene:g}'] - res[f'clinical + HSC RNA C={C_gene:g}']
    print(f'  C={C_gene:g}  {d.mean():+.4f}  p={wilcoxon(d).pvalue:.3g}  wins {int((d>0).sum())}/{len(d)}')
json.dump({k: float(v.mean()) for k, v in res.items()},
          open(f'{ROOT}/results/clinical_adjust.json', 'w'), indent=2)
print('\nwrote results/clinical_adjust.json')
