"""Bring the last three quoted statistics onto the inference the Methods declares.

Still on the abandoned 100-fold paired Wilcoxon:
  (a) the clinical-covariate increments, standard panel and ELN2017, with the gene score
      entering the second stage out-of-fold within each training set
  (b) link-count-only versus arm A in the three references
  (c) the overall-survival comparisons ("significant at three")

All recomputed as per-repeat mean differences (n=20) with the Nadeau-Bengio correction.
"""
import os, json, sys
for v in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[v] = '1'
import numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from scipy import stats
warnings.filterwarnings('ignore')
ROOT='/home/holiday01/2026_cm'; NSPLIT,NREP=5,20
REFS={'BMMC':'programs','AML-mal':'programs_aml_malignant_v3','AML-all':'programs_aml_all_v3'}
def nb(d, n_tr, n_te):
    """Corrected resampled t-test on the individual fold differences; see
    robustness_check.py for why the per-repeat-mean variant was abandoned."""
    d=np.asarray(d).ravel(); m,v=d.mean(),d.var(ddof=1)
    se=np.sqrt(v*((1.0/len(d))+(n_te/n_tr)))
    t=m/se if se>0 else np.nan
    return m,t,(2*stats.t.sf(abs(t),df=len(d)-1) if np.isfinite(t) else np.nan)

expr=pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec=pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr=expr.drop_duplicates('display_label').set_index('display_label')
cols=[c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
S=spec.set_index('dbgap_rnaseq_sample'); lab=S.induction_response.to_dict()
yall=np.array([{'CR':0,'Refractory':1}.get(lab.get(s),-1) for s in cols])
XF=expr.loc[:,cols].to_numpy(dtype=np.float64).T
EX={g:i for i,g in enumerate(expr.index)}
rows=[]

# ---------- (a) clinical covariates and ELN2017 ------------------------------------
print('=== (a) two-stage clinical models, Nadeau-Bengio ===')
pr=np.load(f'{ROOT}/data/programs/programs.npz',allow_pickle=True)
pg=pr['genes'].astype(str); cts=list(pr['cell_types'].astype(str)); jH=cts.index('HSC')
hscA=np.array([EX[pg[i]] for i in np.flatnonzero(pr['W_A'][:,jH]) if pg[i] in EX])
hscR=np.array([EX[pg[i]] for i in np.flatnonzero(pr['W_R'][:,jH]) if pg[i] in EX])
cl=pd.read_excel(f'{ROOT}/data/raw/beataml2/beataml_wv1to4_clinical.xlsx')
cl=cl.drop_duplicates('dbgap_rnaseq_sample').set_index('dbgap_rnaseq_sample')
num=lambda s: pd.to_numeric(s,errors='coerce')
C=pd.DataFrame(index=cols)
C['age']=num(cl.reindex(cols).ageAtDiagnosis)
C['male']=(cl.reindex(cols).consensus_sex=='Male').astype(float)
C['blast']=num(cl.reindex(cols)['%.Blasts.in.BM'])
C['wbc']=np.log1p(num(cl.reindex(cols).wbcCount))
C['flt3']=(cl.reindex(cols)['FLT3-ITD'].astype(str).str.lower()=='positive').astype(float)
C['npm1']=(cl.reindex(cols)['NPM1'].astype(str).str.lower()=='positive').astype(float)
eln=cl.reindex(cols).ELN2017
ELN3=['Favorable','Intermediate','Adverse']
E=pd.DataFrame({'eln_intermediate':(eln=='Intermediate').astype(float),
                'eln_adverse':(eln=='Adverse').astype(float)},index=cols)
ok=(yall>=0)&C.notna().all(1).to_numpy()&eln.isin(ELN3).to_numpy()
Xo,Co,Eo,yo=XF[ok],C[ok.tolist()].to_numpy(float),E[ok.tolist()].to_numpy(float),yall[ok]
print(f'  complete cases {ok.sum()} (CR {(yo==0).sum()} / Ref {(yo==1).sum()}); '
      f'ELN2017 favourable {int((Eo.sum(1)==0).sum())} / intermediate {int(Eo[:,0].sum())} '
      f'/ adverse {int(Eo[:,1].sum())}')
PANELS={'standard panel':Co,'ELN2017':Eo,'standard panel + ELN2017':np.hstack([Co,Eo])}
folds=list(RepeatedStratifiedKFold(n_splits=NSPLIT,n_repeats=NREP,random_state=0).split(Xo,yo))
ntr,nte=ok.sum()*(NSPLIT-1)/NSPLIT, ok.sum()/NSPLIT

def gene_scores(sel,Cg,tr,te,seed):
    """gene score for the second stage: out-of-fold within the training set, so the
    second-stage model never sees a score fitted on the same patient"""
    M=Xo[:,sel]; s_tr=np.empty(len(tr))
    inner=StratifiedKFold(n_splits=NSPLIT,shuffle=True,random_state=seed)
    for itr,ite in inner.split(M[tr],yo[tr]):
        mu,sd=M[tr][itr].mean(0),M[tr][itr].std(0)+1e-8
        s_tr[ite]=LogisticRegression(C=Cg,max_iter=5000).fit((M[tr][itr]-mu)/sd,yo[tr][itr]) \
                  .decision_function((M[tr][ite]-mu)/sd)
    mu,sd=M[tr].mean(0),M[tr].std(0)+1e-8
    s_te=LogisticRegression(C=Cg,max_iter=5000).fit((M[tr]-mu)/sd,yo[tr]) \
         .decision_function((M[te]-mu)/sd)
    return s_tr,s_te

def twostage(sel,Cg,cov=None):
    a=[]
    for i,(tr,te) in enumerate(folds):
        ft,fv=[],[]
        if cov is not None:
            mu,sd=cov[tr].mean(0),cov[tr].std(0)+1e-8
            ft.append((cov[tr]-mu)/sd); fv.append((cov[te]-mu)/sd)
        if sel is not None:
            s_tr,s_te=gene_scores(sel,Cg,tr,te,seed=i)
            ft.append(s_tr[:,None]); fv.append(s_te[:,None])
        clf=LogisticRegression(C=1.0,max_iter=5000).fit(np.hstack(ft),yo[tr])
        a.append(roc_auc_score(yo[te],clf.decision_function(np.hstack(fv))))
    return np.array(a)

ARMS=(('chromatin',hscA),('RNA',hscR))
alone={f'{nm}_{Cg}':twostage(sel,Cg,None) for Cg in (1e-2,1e-1) for nm,sel in ARMS}
for pn,cov in PANELS.items():
    base=twostage(None,None,cov)
    print(f'  {pn} alone {base.mean():.3f}')
    rows.append(dict(analysis='clinical',C=np.nan,comparison=f'{pn} alone',
                     diff=base.mean(),nb_t=np.nan,nb_p=np.nan))
    for Cg in (1e-2,1e-1):
        comb={}
        for nm,sel in ARMS:
            comb[nm]=twostage(sel,Cg,cov)
            m,t,p=nb(comb[nm]-base,ntr,nte)
            print(f'    C={Cg:g} {pn} + {nm:9s} {comb[nm].mean():.3f} '
                  f'(gene alone {alone[f"{nm}_{Cg}"].mean():.3f})  gain {m:+.4f} NB t={t:+.2f} p={p:.4f}')
            rows.append(dict(analysis='clinical',C=Cg,comparison=f'{pn} + {nm} - {pn}',
                             diff=m,nb_t=t,nb_p=p))
        m,t,p=nb(comb['chromatin']-comb['RNA'],ntr,nte)
        print(f'    C={Cg:g} {pn} + chromatin vs + RNA  {m:+.4f} NB t={t:+.2f} p={p:.4f}')
        rows.append(dict(analysis='clinical',C=Cg,comparison=f'{pn} + chromatin - {pn} + RNA',
                         diff=m,nb_t=t,nb_p=p))
for Cg in (1e-2,1e-1):
    for nm,_ in ARMS:
        rows.append(dict(analysis='clinical',C=Cg,comparison=f'{nm} alone',
                         diff=alone[f'{nm}_{Cg}'].mean(),nb_t=np.nan,nb_p=np.nan))
    m,t,p=nb(alone[f'chromatin_{Cg}']-alone[f'RNA_{Cg}'],ntr,nte)
    print(f'  C={Cg:g} chromatin alone vs RNA alone  {m:+.4f} NB t={t:+.2f} p={p:.4f}')
    rows.append(dict(analysis='clinical',C=Cg,comparison='chromatin alone - RNA alone',
                     diff=m,nb_t=t,nb_p=p))

# ---------- (b) link count versus arm A --------------------------------------------
print('\n=== (b) link-count-only vs arm A, Nadeau-Bengio ===')
okm=yall>=0; y=yall[okm]
f2=list(RepeatedStratifiedKFold(n_splits=NSPLIT,n_repeats=NREP,random_state=0).split(np.zeros((okm.sum(),1)),y))
n2tr,n2te=okm.sum()*(NSPLIT-1)/NSPLIT, okm.sum()/NSPLIT
def cvarr(sel,Cg):
    M=XF[:,sel][okm]; a=[]
    for tr,te in f2:
        mu,sd=M[tr].mean(0),M[tr].std(0)+1e-8
        a.append(roc_auc_score(y[te],LogisticRegression(C=Cg,max_iter=5000)
                 .fit((M[tr]-mu)/sd,y[tr]).decision_function((M[te]-mu)/sd)))
    return np.array(a)
for rn,tag in REFS.items():
    z=np.load(f'{ROOT}/data/programs/{tag}.npz',allow_pickle=True)
    g=z['genes'].astype(str); ct=list(z['cell_types'].astype(str)); j=ct.index('HSC'); el=z['eligible']
    SA=z['S_chrom'][:,j] if 'S_chrom' in z.files else None
    if SA is None:
        sp_=np.load(f'{ROOT}/data/programs/specificity_bmmc.npz',allow_pickle=True)
        Sc=np.zeros((len(g),len(ct)),np.float32)
        np.add.at(Sc,z['link_g'],z['link_r'][:,None]*sp_['S_peak'][z['link_p']]); SA=Sc[:,j]
    nl=np.bincount(z['link_g'],minlength=len(g)).astype(float)
    top=lambda s:np.array([EX[g[i]] for i in np.argsort(-np.where(el,np.nan_to_num(s,nan=-np.inf,neginf=-np.inf),-np.inf))[:200] if g[i] in EX])
    A,L=top(SA),top(nl)
    for Cg in (1e-2,1e-1):
        m,t,p=nb(cvarr(A,Cg)-cvarr(L,Cg),n2tr,n2te)
        print(f'  {rn:8s} C={Cg:g}  armA - n_links {m:+.4f} NB t={t:+.2f} p={p:.4f}')
        rows.append(dict(analysis='link_count',reference=rn,C=Cg,comparison='armA - n_links_only',diff=m,nb_t=t,nb_p=p))

# ---------- (c) overall survival ----------------------------------------------------
print('\n=== (c) overall survival, Nadeau-Bengio ===')
t_=pd.to_numeric(S.overallSurvival,errors='coerce'); vs=S.vitalStatus.astype(str).str.lower()
T=np.array([t_.get(s,np.nan) for s in cols],float); V=np.array([vs.get(s,'unknown') for s in cols])
oks=np.isfinite(T)&(T>0)&np.isin(V,['dead','alive']); E=(V=='dead')[oks]; Tt=T[oks]
f3=list(RepeatedStratifiedKFold(n_splits=NSPLIT,n_repeats=NREP,random_state=0).split(np.zeros((oks.sum(),1)),E))
n3tr,n3te=oks.sum()*(NSPLIT-1)/NSPLIT, oks.sum()/NSPLIT
print(f'  evaluable {oks.sum()} (events {E.sum()})')
def cox(sel,al):
    M=XF[:,sel][oks]; a=[]
    for tr,te in f3:
        mu,sd=M[tr].mean(0),M[tr].std(0)+1e-8
        try:
            p_=CoxPHSurvivalAnalysis(alpha=al).fit((M[tr]-mu)/sd,Surv.from_arrays(E[tr],Tt[tr])).predict((M[te]-mu)/sd)
            a.append(concordance_index_censored(E[te],Tt[te],p_)[0])
        except Exception: a.append(np.nan)
    return np.array(a)
for rn,tag in REFS.items():
    z=np.load(f'{ROOT}/data/programs/{tag}.npz',allow_pickle=True)
    g=z['genes'].astype(str); ct=list(z['cell_types'].astype(str)); j=ct.index('HSC')
    A=np.array([EX[g[i]] for i in np.flatnonzero(z['W_A'][:,j]) if g[i] in EX])
    Rr=np.array([EX[g[i]] for i in np.flatnonzero(z['W_R'][:,j]) if g[i] in EX])
    for al in (100.,10.,1.,.1):
        d=cox(A,al)-cox(Rr,al); d=d[np.isfinite(d)]
        if len(d)<NREP*NSPLIT: print(f'  {rn:8s} alpha={al:<5g} incomplete folds'); continue
        m,t,p=nb(d,n3tr,n3te)
        print(f'  {rn:8s} alpha={al:<5g} armA - armR {m:+.4f} NB t={t:+.2f} p={p:.4f}')
        rows.append(dict(analysis='overall_survival',reference=rn,C=al,comparison='armA - armR',diff=m,nb_t=t,nb_p=p))
pd.DataFrame(rows).to_csv(f'{ROOT}/results/recomputed_remaining.csv',index=False)
print('\nwrote results/recomputed_remaining.csv')
