"""Assert every number the manuscript quotes against the files that produced it.

Each chk() call carries the value as written in the text and the expression that reads it
back out of results/. A mismatch prints both and the script exits non-zero, so a rerun
that moves a number cannot pass unnoticed.

Every mismatch found in any section is carried to the exit code. An earlier version cleared
the list between sections and exited on the last one alone, so a mismatch in the first
section printed and then passed. A value that is missing or not finite counts as a
mismatch, because a comparison against NaN is false whatever the expected value.
"""
import json, math, sys, numpy as np, pandas as pd
R='/home/holiday01/2026_cm/results'
rb=pd.read_csv(f'{R}/robustness_check.csv'); rp=pd.read_csv(f'{R}/recomputed_pvalues.csv')
rr=pd.read_csv(f'{R}/recomputed_remaining.csv'); bs=pd.read_csv(f'{R}/budget_sweep.csv')
L=json.load(open(f'{R}/baseline_lsc104.json')); lo=json.load(open(f'{R}/locus_arm.json'))
cf=json.load(open(f'{R}/confound_programs.json'))
ALL=[]; BAD=[]
def chk(l,c,a,tol=6e-4):
    ok = a is not None and math.isfinite(a) and math.isfinite(c) and abs(c-a)<=tol
    if not ok: BAD.append((l,c,a))
def section(name):
    print(f'{name}: {len(BAD)} mismatch')
    for l,c,aa in BAD: print(f'  {l:34s} manuscript={c} source={aa}')
    ALL.extend(BAD); BAD.clear()
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
section('pass1')

# --- second pass: remaining narrative numbers ---
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
cf2=cf['cv']
chk('nlinks only',0.712,cf2['n_links only']['0.01'],1e-3)
chk('full score',0.773,cf2['arm A (chromatin)']['0.01'],1e-3)
section('pass2')

# --- third pass: clinical / ELN numbers quoted in the manuscript ---
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
section('pass3')

# --- fourth pass: the HSC rank claim and the two nested AML constructions ---
def hsc_rank(f, split, C):
    j = json.load(open(f'{R}/{f}'))[split]
    A = {k[8:]: v[C] for k, v in j.items() if k.startswith('prog-A: ')}
    s = sorted(A.items(), key=lambda kv: -kv[1])
    return [i + 1 for i, (k, _) in enumerate(s) if k == 'HSC'][0], len(s)
RANKS = {'eval_programs.json': ((1, 2), (1, 3), 22),
         'eval_programs_aml_malignant_v3.json': ((6, 7), (5, 4), 18),
         'eval_programs_aml_all_v3.json': ((7, 7), (2, 3), 20)}
aml_ranks = []
for f, (cv, xw, n) in RANKS.items():
    for split, exp in (('cv', cv), ('xwave', xw)):
        for C, e in zip(('0.01', '0.1'), exp):
            r, tot = hsc_rank(f, split, C)
            chk(f'{f[:18]} {split} {C} rank', e, r, 0); chk(f'{f[:18]} n', n, tot, 0)
            if 'aml' in f: aml_ranks.append(r)
chk('AML rank range low', 2, min(aml_ranks), 0)
chk('AML rank range high', 7, max(aml_ranks), 0)
zm = np.load('/home/holiday01/2026_cm/data/programs/programs_aml_malignant_v3.npz', allow_pickle=True)
za = np.load('/home/holiday01/2026_cm/data/programs/programs_aml_all_v3.npz', allow_pickle=True)
ctm, cta = set(zm['cell_types'].astype(str)), set(za['cell_types'].astype(str))
chk('shared cell types', 18, len(ctm & cta), 0)
gm, ga = zm['genes'].astype(str), za['genes'].astype(str)
em, ea = set(gm[zm['eligible']]), set(ga[za['eligible']])
chk('shared eligible', 7047, len(em & ea), 0)
chk('pct smaller', 0.86, len(em & ea) / min(len(em), len(ea)), 5e-3)
chk('pct larger', 0.78, len(em & ea) / max(len(em), len(ea)), 5e-3)
jm = list(zm['cell_types'].astype(str)).index('HSC'); ja = list(za['cell_types'].astype(str)).index('HSC')
chk('shared HSC arm A', 104,
    len(set(gm[np.flatnonzero(zm['W_A'][:, jm])]) & set(ga[np.flatnonzero(za['W_A'][:, ja])])), 0)
section('pass4')


# --- fifth pass: cohort flow, the link-count contrast, the zero budget cell and the
#     exploratory cross-programme analysis ---
fl = pd.read_csv(f'{R}/cohort_flow.csv').set_index('step').n
for st, v in [('initial diagnosis records', 580), ('naming an RNA-seq sample', 458),
              ('present in the expression matrix', 450),
              ('with an inhibitor over a complete dose series', 346)]:
    chk(f'flow {st[:22]}', v, int(fl.loc[st]), 0)
ex = pd.read_csv(f'{R}/cohort_flow_excluded.csv').set_index('group')
chk('excluded n', 104, int(ex.loc['excluded', 'n']), 0)
chk('excluded labelled', 75, int(ex.loc['excluded', 'labelled']), 0)
chk('excluded refr rate', 0.307, float(ex.loc['excluded', 'refractory_rate']), 1e-3)
chk('retained refr rate', 0.315, float(ex.loc['retained', 'refractory_rate']), 1e-3)
chk('excluded median age', 62.5, float(ex.loc['excluded', 'median_age']), 0.05)
chk('retained median age', 61, float(ex.loc['retained', 'median_age']), 0.05)

lk = rr[(rr.analysis == 'link_count') & (rr.reference != 'BMMC')]
chk('link count max gap', 0.0132, float(lk['diff'].abs().max()), 6e-4)
chk('link count A ahead', 3, int((lk['diff'] > 0).sum()), 0)
chk('link count R ahead', 1, int((lk['diff'] < 0).sum()), 0)
chk('link count R margin', 0.0002, float(-lk['diff'].min()), 1e-4)
chk('link count none sig', 0, int((lk.nb_p < 0.05).sum()), 0)

b10 = bs[(bs.reference == 'AML-all') & (bs.k == 10)].iloc[0]
cells10 = [float(b10[f'd_{m}_{c:g}']) for m in ('cv', 'xw') for c in (0.01, 0.1)]
chk('AML-all k=10 zero cell', 0.0, min(abs(v) for v in cells10), 1e-12)
chk('AML-all k=10 none pos', 0, sum(v > 0 for v in cells10), 0)
chk('AML-all k=10 neg', 3, sum(v < 0 for v in cells10), 0)

xa = pd.read_csv(f'{R}/cross_reference_aggregate.csv').set_index('cell')
for cell, n, pos in [('BMMC C=0.01', 22, 12), ('BMMC C=0.1', 22, 12),
                     ('AML-mal C=0.01', 18, 13), ('AML-mal C=0.1', 18, 13)]:
    chk(f'xref n {cell}', n, int(xa.loc[cell, 'n']), 0)
    chk(f'xref pos {cell}', pos, int(xa.loc[cell, 'positive']), 0)
for cell, w, sg in [('BMMC C=0.01', 0.775, 0.832), ('AML-mal C=0.01', 0.012, 0.096),
                    ('AML-mal C=0.1', 0.0104, 0.096)]:
    chk(f'xref wilcoxon {cell}', w, float(xa.loc[cell, 'wilcoxon_p']), 1e-3)
    chk(f'xref sign {cell}', sg, float(xa.loc[cell, 'sign_p']), 1e-3)
xp = pd.read_csv(f'{R}/cross_reference_programs.csv')
chk('xref matched', 17, len(xp), 0)
chk('xref all four pos', 4, int(xp.allpos.sum()), 0)
chk('xref stem all pos', 3, int(xp[xp.stem].allpos.sum()), 0)
chk('xref rest all pos', 1, int(xp[~xp.stem].allpos.sum()), 0)
chk('xref stem mean', 0.0412, float(xp[xp.stem].mean4.mean()), 6e-4)
chk('xref rest mean', 0.0089, float(xp[~xp.stem].mean4.mean()), 6e-4)
chk('xref gap', 0.0323, float(xp[xp.stem].mean4.mean() - xp[~xp.stem].mean4.mean()), 6e-4)
from scipy.stats import spearmanr
chk('xref rho .01', -0.147, float(spearmanr(xp.b001, xp.a001).statistic), 1e-3)
chk('xref rho .1', 0.066, float(spearmanr(xp.b01, xp.a01).statistic), 1e-3)
chk('xref sign agree .01', 6, int(((xp.b001 > 0) == (xp.a001 > 0)).sum()), 0)
chk('xref sign agree .1', 7, int(((xp.b01 > 0) == (xp.a01 > 0)).sum()), 0)
em_ = json.load(open(f'{R}/eval_programs_aml_malignant_v3.json'))['cv']
pa = {k[8:]: v['0.01'] for k, v in em_.items() if k.startswith('prog-A: ')}
best = max(pa, key=pa.get)
chk('AML best prog auroc', 0.774, pa[best], 6e-4)
chk('AML HSC auroc', 0.741, pa['HSC'], 6e-4)
chk('AML best is CLP', 1, int(best == 'CLP'), 0)
section('pass5')

print(f'\ntotal: {len(ALL)} mismatch across all sections')
sys.exit(1 if ALL else 0)
