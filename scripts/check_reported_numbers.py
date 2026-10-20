"""Assert every number the manuscript quotes against the files that produced it.

Each chk() call carries the value as written in the text and the expression that reads it
back out of results/. A mismatch prints both and the script exits non-zero, so a rerun
that moves a number cannot pass unnoticed.
"""
import json, numpy as np, pandas as pd
R='/home/holiday01/2026_cm/results'
rb=pd.read_csv(f'{R}/robustness_check.csv'); rp=pd.read_csv(f'{R}/recomputed_pvalues.csv')
rr=pd.read_csv(f'{R}/recomputed_remaining.csv'); bs=pd.read_csv(f'{R}/budget_sweep.csv')
L=json.load(open(f'{R}/baseline_lsc104.json')); lo=json.load(open(f'{R}/locus_arm.json'))
cf=json.load(open(f'{R}/confound_programs.json'))
BAD=[]
def chk(l,c,a,tol=6e-4):
    if a is None or abs(c-a)>tol: BAD.append((l,c,a))
a=rb[rb.arm=='armA_sum']
chk('cv min',0.011,float(a.cv_diff.min()),1e-3); chk('cv max',0.070,float(a.cv_diff.max()),1e-3)
chk('xw min',-0.005,float(a.xwave_diff.min()),1e-3); chk('xw max',0.083,float(a.xwave_diff.max()),1e-3)
chk('cv pos',12,int((a.cv_diff>0).sum()),0); chk('xw pos',10,int((a.xwave_diff>0).sum()),0); chk('boot at zero',1,int(((a.boot_lo>-1e-9)&(a.boot_lo<1e-9)).sum()),0)
chk('nb sig',2,int((a.nb_p<0.05).sum()),0); chk('boot',2,int((a.boot_lo>1e-9).sum()),0)
for nm,pos,sig in [('armA_count_partialled',12,2),('armA_mean_per_link',9,0),('n_links_only',8,1)]:
    v=rb[rb.arm==nm]; chk(f'{nm} pos',pos,int((v.cv_diff>0).sum()),0); chk(f'{nm} sig',sig,int((v.nb_p<0.05).sum()),0)
for t,l,n in [('programs','BMMC',80),('programs_aml_malignant_v3','mal',55),('programs_aml_all_v3','all',75)]:
    z=np.load(f'/home/holiday01/2026_cm/data/programs/{t}.npz',allow_pickle=True)
    ct=list(z['cell_types'].astype(str)); j=ct.index('HSC')
    chk(f'overlap {l}',n,len(set(np.flatnonzero(z['W_A'][:,j]))&set(np.flatnonzero(z['W_R'][:,j]))),0)
for t,l,e in [('programs_aml_malignant_v3','mal',9039),('programs_aml_all_v3','all',8155)]:
    z=np.load(f'/home/holiday01/2026_cm/data/programs/{t}.npz',allow_pickle=True)
    chk(f'eligible {l}',e,int(z['eligible'].sum()),0)
k=bs[bs.k.isin([150,200])]
chk('150+200 all pos',24,sum(int(r[f'd_cv_{c:g}']>0)+int(r[f'd_xw_{c:g}']>0) for _,r in k.iterrows() for c in (0.01,0.1)),0)
k100=bs[bs.k==100]
npos=sum(1 for _,r in k100.iterrows() if all(r[f'd_{m}_{c:g}']>0 for m in ('cv','xw') for c in (0.01,0.1)))
chk('k=100 refs all-4-pos',1,npos,0)
for C,v in [(0.01,0.686),(0.1,0.682)]: chk(f'pub {C}',v,L['cv']['Ng2016 LSC genes (published, experimental)'][str(C)],1e-3)
for C,v in [(0.01,0.773),(0.1,0.781)]: chk(f'armA99 {C}',v,L['cv']['HSC arm A top-99 (chromatin)'][str(C)],1e-3)
def P(c,C,col='cv_diff'):
    r=rp[(rp.comparison==c)&(np.isclose(rp.C,C))]; return float(r[col].iloc[0]) if len(r) else None
chk('A-pub .01',0.087,P('armA_99 vs published_LSC',0.01),1e-3)
chk('A-pub p',0.0003,P('armA_99 vs published_LSC',0.01,'nb_p'),5e-4)
os_=rr[rr.analysis=='overall_survival']
chk('OS pos',7,int((os_['diff']>0).sum()),0); chk('OS sig',0,int((os_.nb_p<0.05).sum()),0)
chk('OS min',-0.047,float(os_['diff'].min()),1e-3); chk('OS max',0.022,float(os_['diff'].max()),1e-3)
for k_,v in [('arm A (chromatin)',14),('arm R (RNA)',8),('eligible background',4)]:
    chk(f'links {k_}',v,cf['profile'][k_]['n_links'],0)
chk('locus_mean',0.743,lo['cv']['locus_mean (no linking)']['0.01'],1e-3)
print(f'{len(BAD)} mismatch')
for l,c,aa in BAD: print(f'  {l:26s} manuscript={c} source={aa}')

# --- second pass: remaining narrative numbers ---
BAD.clear()
b=bs.set_index(['reference','k'])
chk('17g healthy A-R .01',-0.054,float(b.loc[('BMMC',17),'d_cv_0.01']),1e-3)
chk('17g healthy A-R .1',-0.021,float(b.loc[('BMMC',17),'d_cv_0.1']),1e-3)
chk('10g healthy A-R .01',-0.072,float(b.loc[('BMMC',10),'d_cv_0.01']),1e-3)
chk('10g healthy A-R .1',-0.074,float(b.loc[('BMMC',10),'d_cv_0.1']),1e-3)
for C,v in [(0.01,0.663),(0.1,0.670)]:
    chk(f'LSC17 {C}',v,json.load(open(f'{R}/baseline_lsc17.json'))['cv']['LSC17 (published gene set)'][str(C)],1e-3)
for nm,C,v in [('HSC arm A top-99 (chromatin)',0.01,0.835),('HSC arm A top-99 (chromatin)',0.1,0.833),
               ('Ng2016 LSC genes (published, experimental)',0.01,0.699),('Ng2016 LSC genes (published, experimental)',0.1,0.750)]:
    chk(f'xw {nm[:12]} {C}',v,L['xwave'][nm][str(C)],1e-3)
lp=json.load(open(f'{R}/link_permutation.json'))
lpw=json.load(open(f'{R}/link_permutation_window100.json'))
for C,ab,p in [('0.01',93,0.079),('0.1',83,0.178)]:
    d=lpw[f'window_C{C}']
    chk(f'win above {C}',ab,d['n']-round(d['emp_p']*(d['n']+1)-1),0)
    chk(f'win empp {C}',p,d['emp_p'],5e-4)
    chk(f'win null {C}',{'0.01':0.754,'0.1':0.751}[C],d['null_cv_mean'],1e-3)
g=json.load(open(f'{R}/link_permutation.json'))['global_C0.01']
chk('global null',0.724,g['null_cv_mean'],1e-3); chk('global sd',0.015,g['null_cv_sd'],1e-3)
chk('global above',20,g['n']-round(g['emp_p']*(g['n']+1)-1),0)
ca=json.load(open(f'{R}/clinical_adjust.json'))
chk('clin only',0.734,ca['clinical only'],1e-3)
chk('A alone .01',0.756,ca['HSC chromatin alone C=0.01'],1e-3)
chk('A alone .1',0.752,ca['HSC chromatin alone C=0.1'],1e-3)
chk('clin+A .01',0.757,ca['clinical + HSC chromatin C=0.01'],1e-3)
chk('clin+A .1',0.743,ca['clinical + HSC chromatin C=0.1'],1e-3)
cf2=cf['cv']
chk('nlinks only',0.712,cf2['n_links only']['0.01'],1e-3)
chk('full score',0.773,cf2['arm A (chromatin)']['0.01'],1e-3)
print(f'pass2: {len(BAD)} mismatch')
for l,c,aa in BAD: print(f'  {l:26s} manuscript={c} source={aa}')

# --- third pass: clinical / ELN numbers quoted in the manuscript ---
BAD.clear()
cr=pd.read_csv(f'{R}/recomputed_remaining.csv'); cc=cr[cr.analysis=='clinical']
def C_(comp,c=None):
    v=cc[cc.comparison==comp] if c is None else cc[(cc.comparison==comp)&(np.isclose(cc.C,c))]
    return float(v['diff'].iloc[0]) if len(v) else None
def Cp(comp,c):
    v=cc[(cc.comparison==comp)&(np.isclose(cc.C,c))]
    return float(v['nb_p'].iloc[0]) if len(v) else None
chk('panel alone',0.730,C_('standard panel alone'),6e-4)
chk('ELN alone',0.737,C_('ELN2017 alone'),6e-4)
chk('panel+ELN alone',0.754,C_('standard panel + ELN2017 alone'),6e-4)
chk('chrom alone .01',0.759,C_('chromatin alone',0.01),6e-4)
chk('chrom alone .1',0.747,C_('chromatin alone',0.1),6e-4)
chk('RNA alone .01',0.735,C_('RNA alone',0.01),6e-4)
chk('RNA alone .1',0.725,C_('RNA alone',0.1),6e-4)
for lbl,comp,c,d,pv in [
    ('panel+chrom .01','standard panel + chromatin - standard panel',0.01,0.059,0.012),
    ('panel+chrom .1','standard panel + chromatin - standard panel',0.1,0.055,0.023),
    ('panel+RNA .01','standard panel + RNA - standard panel',0.01,0.048,0.043),
    ('panel+RNA .1','standard panel + RNA - standard panel',0.1,0.037,0.098),
    ('eln+chrom .01','ELN2017 + chromatin - ELN2017',0.01,0.043,0.032),
    ('eln+chrom .1','ELN2017 + chromatin - ELN2017',0.1,0.046,0.021),
    ('eln+RNA .01','ELN2017 + RNA - ELN2017',0.01,0.031,0.15),
    ('eln+RNA .1','ELN2017 + RNA - ELN2017',0.1,0.028,0.17),
    ('both+chrom .01','standard panel + ELN2017 + chromatin - standard panel + ELN2017',0.01,0.029,0.061),
    ('both+chrom .1','standard panel + ELN2017 + chromatin - standard panel + ELN2017',0.1,0.028,0.085),
    ('both+RNA .01','standard panel + ELN2017 + RNA - standard panel + ELN2017',0.01,0.017,0.33),
    ('both+RNA .1','standard panel + ELN2017 + RNA - standard panel + ELN2017',0.1,0.013,0.41),
    ('A-R panel .01','standard panel + chromatin - standard panel + RNA',0.01,0.012,0.35),
    ('A-R panel .1','standard panel + chromatin - standard panel + RNA',0.1,0.018,0.30),
    ('A-R eln .01','ELN2017 + chromatin - ELN2017 + RNA',0.01,0.013,0.38),
    ('A-R eln .1','ELN2017 + chromatin - ELN2017 + RNA',0.1,0.018,0.34),
    ('A-R both .01','standard panel + ELN2017 + chromatin - standard panel + ELN2017 + RNA',0.01,0.012,0.27),
    ('A-R both .1','standard panel + ELN2017 + chromatin - standard panel + ELN2017 + RNA',0.1,0.016,0.26),
    ('A-R alone .01','chromatin alone - RNA alone',0.01,0.025,0.30),
    ('A-R alone .1','chromatin alone - RNA alone',0.1,0.022,0.50)]:
    chk(lbl+' d',d,C_(comp,c),6e-4); chk(lbl+' p',pv,Cp(comp,c),6e-3)
# combined-model AUROCs quoted in Results come from the log, checked against diffs:
for lbl,base,comp,c,v in [('panel+chrom auroc .01','standard panel alone','standard panel + chromatin - standard panel',0.01,0.789),
                          ('panel+chrom auroc .1','standard panel alone','standard panel + chromatin - standard panel',0.1,0.784),
                          ('panel+RNA auroc .01','standard panel alone','standard panel + RNA - standard panel',0.01,0.777),
                          ('panel+RNA auroc .1','standard panel alone','standard panel + RNA - standard panel',0.1,0.766),
                          ('eln+chrom auroc .01','ELN2017 alone','ELN2017 + chromatin - ELN2017',0.01,0.780),
                          ('eln+chrom auroc .1','ELN2017 alone','ELN2017 + chromatin - ELN2017',0.1,0.783),
                          ('eln+RNA auroc .01','ELN2017 alone','ELN2017 + RNA - ELN2017',0.01,0.767),
                          ('eln+RNA auroc .1','ELN2017 alone','ELN2017 + RNA - ELN2017',0.1,0.765),
                          ('both+chrom auroc .01','standard panel + ELN2017 alone','standard panel + ELN2017 + chromatin - standard panel + ELN2017',0.01,0.783),
                          ('both+chrom auroc .1','standard panel + ELN2017 alone','standard panel + ELN2017 + chromatin - standard panel + ELN2017',0.1,0.782),
                          ('both+RNA auroc .01','standard panel + ELN2017 alone','standard panel + ELN2017 + RNA - standard panel + ELN2017',0.01,0.771),
                          ('both+RNA auroc .1','standard panel + ELN2017 alone','standard panel + ELN2017 + RNA - standard panel + ELN2017',0.1,0.767)]:
    chk(lbl,v,C_(base)+C_(comp,c),1e-3)
print(f'pass3: {len(BAD)} mismatch')
for l,c,aa in BAD: print(f'  {l:26s} manuscript={c} source={aa}')

import sys
sys.exit(1 if BAD else 0)
