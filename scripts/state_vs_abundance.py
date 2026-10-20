"""Does arm R index how MANY stem cells are present, while arm A indexes what STATE they are in?

Expression-specificity selection picks marker genes -- high in the cell type, low elsewhere --
whose bulk signal tracks the abundance of that compartment. Linked-peak selection picks genes
that are regulatorily engaged in the compartment, whose signal need not track its size.

If that distinction is real, the arm R score should align more closely with established
abundance readouts (bone-marrow blast percentage, CD34 expression, the published LSC gene
sets) than the arm A score does, while predicting induction response less well.

Scores are the out-of-fold linear predictors from the same repeated cross-validation used
throughout, averaged per patient, so nothing is read in-sample.
"""
import os, json
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v]='1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
warnings.filterwarnings('ignore')
ROOT='/home/holiday01/2026_cm'; C=0.01; NSPLIT,NREP=5,20
expr=pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec=pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr=expr.drop_duplicates('display_label').set_index('display_label')
cols=[c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
S=spec.set_index('dbgap_rnaseq_sample'); lab=S.induction_response.to_dict()
yall=np.array([{'CR':0,'Refractory':1}.get(lab.get(s),-1) for s in cols])
XF=expr.loc[:,cols].to_numpy(dtype=np.float64).T; EX={g:i for i,g in enumerate(expr.index)}
okm=yall>=0; y=yall[okm]; idx=np.flatnonzero(okm)
pr=np.load(f'{ROOT}/data/programs/programs.npz',allow_pickle=True)
pg=pr['genes'].astype(str); cts=list(pr['cell_types'].astype(str)); jH=cts.index('HSC')
A=np.array([EX[pg[i]] for i in np.flatnonzero(pr['W_A'][:,jH]) if pg[i] in EX])
R=np.array([EX[pg[i]] for i in np.flatnonzero(pr['W_R'][:,jH]) if pg[i] in EX])
folds=list(RepeatedStratifiedKFold(n_splits=NSPLIT,n_repeats=NREP,random_state=0).split(np.zeros((okm.sum(),1)),y))
def oof(sel):
    M=XF[:,sel][okm]; acc=np.zeros(len(y)); cnt=np.zeros(len(y)); auc=[]
    for tr,te in folds:
        mu,sd=M[tr].mean(0),M[tr].std(0)+1e-8
        m=LogisticRegression(C=C,max_iter=5000).fit((M[tr]-mu)/sd,y[tr])
        d=m.decision_function((M[te]-mu)/sd)
        acc[te]+=(d-d.mean())/(d.std()+1e-12); cnt[te]+=1; auc.append(roc_auc_score(y[te],d))
    return acc/cnt, float(np.mean(auc))
sA,aucA=oof(A); sR,aucR=oof(R)
print(f'out-of-fold scores: arm A AUROC {aucA:.3f} | arm R AUROC {aucR:.3f}\n')
# abundance readouts
cl=pd.read_excel(f'{ROOT}/data/raw/beataml2/beataml_wv1to4_clinical.xlsx').drop_duplicates('dbgap_rnaseq_sample').set_index('dbgap_rnaseq_sample')
blast=pd.to_numeric(cl.reindex(cols)['%.Blasts.in.BM'],errors='coerce').to_numpy()[okm]
pb=pd.to_numeric(cl.reindex(cols)['%.Blasts.in.PB'],errors='coerce').to_numpy()[okm] if '%.Blasts.in.PB' in cl.columns else None
def zmean(names):
    sel=np.array([EX[g] for g in names if g in EX])
    M=XF[:,sel][okm]; return ((M-M.mean(0))/(M.std(0)+1e-8)).mean(1)
lsc104=[l.strip() for l in open(f'{ROOT}/data/ref/published_signatures/ng2016_lsc104_resolved.txt') if l.strip()]
l17=[{'LOC284422':'C19orf77'}.get(l.strip(),l.strip()) for l in open(f'{ROOT}/data/ref/published_signatures/ng2016_lsc17.txt') if l.strip()]
READ={'BM blast %':blast,'CD34 expression':zmean(['CD34']),
      'Ng2016 LSC104 mean z':zmean(lsc104),'LSC17 mean z':zmean(l17),
      'arm R gene-set mean z':zmean([pg[i] for i in np.flatnonzero(pr['W_R'][:,jH]) if pg[i] in EX]),
      'arm A gene-set mean z':zmean([pg[i] for i in np.flatnonzero(pr['W_A'][:,jH]) if pg[i] in EX])}
if pb is not None: READ['PB blast %']=pb
print(f'{"abundance readout":26s}{"rho vs arm A":>14s}{"rho vs arm R":>14s}{"|R|-|A|":>10s}{"p(diff)":>10s}')
rows=[]
for nm,v in READ.items():
    m=np.isfinite(v)
    ra=stats.spearmanr(sA[m],v[m]); rr=stats.spearmanr(sR[m],v[m])
    # Steiger test for dependent correlations sharing one variable
    r12,r13=ra.statistic,rr.statistic; r23=stats.spearmanr(sA[m],sR[m]).statistic; n=m.sum()
    d=(1-r23)*2; det=1-r12**2-r13**2-r23**2+2*r12*r13*r23
    av=(r12+r13)/2; cube=(1-r23)**3
    t=(r12-r13)*np.sqrt((n-1)*(1+r23)/((2*(n-1)/(n-3))*det+av**2*cube)) if det>0 else np.nan
    p=2*stats.t.sf(abs(t),df=n-3) if np.isfinite(t) else np.nan
    print(f'{nm:26s}{r12:+14.3f}{r13:+14.3f}{abs(r13)-abs(r12):+10.3f}{p:10.4f}')
    rows.append(dict(readout=nm,n=int(n),rho_armA=r12,rho_armR=r13,abs_diff=abs(r13)-abs(r12),steiger_p=p))
print(f'\nrho(arm A score, arm R score) = {stats.spearmanr(sA,sR).statistic:+.3f}')
pd.DataFrame(rows).to_csv(f'{ROOT}/results/state_vs_abundance.csv',index=False)
print('wrote results/state_vs_abundance.csv')
