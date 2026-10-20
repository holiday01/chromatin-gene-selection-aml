"""Is the chromatin signal distributed across the gene set, or concentrated in its best genes?

The proposed reading is that the score indexes a compartment-level chromatin configuration
rather than a set of individually informative genes. If so, a random subset of arm A should
perform about as well as the top-ranked genes of arm A at the same size. If instead a few
genes carry the signal, the top-ranked subset should clearly win.

Arms, all at sizes 25/50/100/150 drawn from the arm A 200-gene HSC programme:
  top-k         the k highest-scoring genes of arm A
  random-k      k genes drawn at random from the 200 (20 draws)
  bottom-k      the k lowest-scoring genes of arm A
and, as the external anchor, k genes drawn at random from the eligible pool (20 draws).
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
ROOT='/home/holiday01/2026_cm'; C_GRID=[1e-2,1e-1]; NSPLIT,NREP=5,20; NDRAW=20
SIZES=[25,50,100,150,200]
expr=pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec=pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr=expr.drop_duplicates('display_label').set_index('display_label')
cols=[c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
S=spec.set_index('dbgap_rnaseq_sample'); lab=S.induction_response.to_dict(); coh=S.cohort.to_dict()
yall=np.array([{'CR':0,'Refractory':1}.get(lab.get(s),-1) for s in cols])
cw=np.array([str(coh.get(s)) for s in cols]); okm=yall>=0; y=yall[okm]
tr,te=okm&(cw=='Waves1+2'), okm&(cw=='Waves3+4')
XF=expr.loc[:,cols].to_numpy(dtype=np.float64).T; EX={g:i for i,g in enumerate(expr.index)}
folds=list(RepeatedStratifiedKFold(n_splits=NSPLIT,n_repeats=NREP,random_state=0).split(np.zeros((okm.sum(),1)),y))
def run(sel):
    Mw=XF[:,sel]; M=Mw[okm]; o={}
    for C in C_GRID:
        cv=[]
        for a,b in folds:
            mu,sd=M[a].mean(0),M[a].std(0)+1e-8
            cv.append(roc_auc_score(y[b],LogisticRegression(C=C,max_iter=5000)
                      .fit((M[a]-mu)/sd,y[a]).decision_function((M[b]-mu)/sd)))
        mu,sd=Mw[tr].mean(0),Mw[tr].std(0)+1e-8
        o[C]=(float(np.mean(cv)),roc_auc_score(yall[te],LogisticRegression(C=C,max_iter=5000)
              .fit((Mw[tr]-mu)/sd,yall[tr]).decision_function((Mw[te]-mu)/sd)))
    return o
pr=np.load(f'{ROOT}/data/programs/programs.npz',allow_pickle=True)
pg=pr['genes'].astype(str); cts=list(pr['cell_types'].astype(str)); jH=cts.index('HSC'); el=pr['eligible']
sp_=np.load(f'{ROOT}/data/programs/specificity_bmmc.npz',allow_pickle=True)
Sc=np.zeros((len(pg),len(cts)),np.float32)
np.add.at(Sc,pr['link_g'],pr['link_r'][:,None]*sp_['S_peak'][pr['link_p']])
sA=np.where(el,Sc[:,jH],-np.inf)
order=np.argsort(-sA)[:200]
A200=[EX[pg[i]] for i in order if pg[i] in EX]
pool=np.array([EX[pg[i]] for i in np.flatnonzero(el) if pg[i] in EX])
rng=np.random.default_rng(99); rows=[]
print(f'{"size":>5s}{"arm":>14s}' + ''.join(f'{f"CV C={C:g}":>11s}{"xw":>8s}' for C in C_GRID))
for k in SIZES:
    out={}
    out['top-k']=run(np.array(A200[:k]))
    if k<200: out['bottom-k']=run(np.array(A200[-k:]))
    rs=[run(rng.choice(A200,k,replace=False)) for _ in range(NDRAW if k<200 else 1)]
    rp_=[run(rng.choice(pool,k,replace=False)) for _ in range(NDRAW)]
    for nm,v in out.items():
        print(f'{k:5d}{nm:>14s}'+''.join(f'{v[C][0]:11.3f}{v[C][1]:8.3f}' for C in C_GRID))
        rows.append(dict(size=k,arm=nm,**{f'cv_{C:g}':v[C][0] for C in C_GRID},**{f'xw_{C:g}':v[C][1] for C in C_GRID}))
    for nm,draws in (('random-in-A',rs),('random-in-pool',rp_)):
        m={C:np.mean([d[C][0] for d in draws]) for C in C_GRID}
        s_={C:np.std([d[C][0] for d in draws]) for C in C_GRID}
        mx={C:np.mean([d[C][1] for d in draws]) for C in C_GRID}
        print(f'{k:5d}{nm:>14s}'+''.join(f'{m[C]:11.3f}{mx[C]:8.3f}' for C in C_GRID)
              + f'   (sd {s_[C_GRID[0]]:.3f})')
        rows.append(dict(size=k,arm=nm,**{f'cv_{C:g}':m[C] for C in C_GRID},
                         **{f'xw_{C:g}':mx[C] for C in C_GRID},
                         **{f'sd_{C:g}':s_[C] for C in C_GRID}))
pd.DataFrame(rows).to_csv(f'{ROOT}/results/distributed_test.csv',index=False)
print('\nwrote results/distributed_test.csv')
