"""Check the numbers the manuscript quotes against the files that produced them.

There are three layers here and they are not equally strong.

Passes 1 to 8 carry the value as written in the text beside the expression that reads it
back out of results/. They catch a result that moves under a rerun. They do not catch an
edit to the manuscript, because the number they compare against is transcribed here.

Pass 10 reads the value out of the sentence that states it and compares that with a source
field, keeping the sign, the exponent and the precision the sentence uses. An edit to the
sentence is therefore checked against the source, and an edit that leaves the sentence
unmatchable fails as a missing claim. This is the layer to extend when a claim must not
drift. bound() is its form for a sentence that states a bound rather than a value.

Pass 9 reports how many decimals in the text are bound by a claim, how many merely appear
as a literal somewhere in this file, and how many are neither. Only the last of those
fails. A literal-only number is not a checked number, since a comment or an unrelated
tolerance satisfies that test, which is why the counts are printed rather than called
coverage. Exemptions are bound to the sentence and the role, not to the value.

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
ex = pd.read_csv(f'{R}/cohort_flow_excluded.csv').set_index('variable')
def EX(var, group):
    return str(ex.loc[var, group])
for var, g, want in [
        ('specimens', 'excluded', '104'), ('specimens', 'retained', '346'),
        ('refractory', 'excluded', '23/75 (30.7%)'), ('refractory', 'retained', '80/254 (31.5%)'),
        ('age at diagnosis, years, median (IQR)', 'excluded', '62.5 (51.0 to 69.5)'),
        ('age at diagnosis, years, median (IQR)', 'retained', '61.0 (44.0 to 70.0)'),
        ('bone-marrow aspirate', 'excluded', '65/104 (62.5%)'),
        ('bone-marrow aspirate', 'retained', '208/346 (60.1%)'),
        ('ELN 2017 adverse', 'excluded', '36/75 (48.0%)'),
        ('ELN 2017 adverse', 'retained', '83/254 (32.7%)'),
        ('collection waves 1 and 2', 'excluded', '61/75 (81.3%)'),
        ('collection waves 1 and 2', 'retained', '150/254 (59.1%)'),
        ('induction recorded as standard chemotherapy', 'retained', '249/254 (98.0%)')]:
    chk(f'cohort {var[:20]} {g}', 1, int(EX(var, g) == want), 0)
rg = pd.read_csv(f'{R}/cohort_regimens.csv').set_index('item').value
chk('distinct first regimens', 60, int(rg.loc['distinct first regimen entries']), 0)
chk('7+3 or 3+7 regimens', 108, int(rg.loc['first entry is 7+3 or 3+7']), 0)

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


# --- sixth pass: the permutation overlap, the reference scope of each control, and the
#     construction of the two expression-variance controls ---
po = pd.read_csv(f'{R}/permutation_overlap.csv').set_index('null')
chk('window overlap median', 87, float(po.loc['window', 'overlap_median']), 0)
chk('global overlap median', 17, float(po.loc['global', 'overlap_median']), 0)
chk('window perms', 100, int(po.loc['window', 'n_permutations']), 0)
chk('global perms', 20, int(po.loc['global', 'n_permutations']), 0)
chk('real distinct peaks', 39589, int(po.loc['window', 'distinct_peaks_real']), 0)
chk('window distinct peaks', 41357, float(po.loc['window', 'distinct_peaks_median']), 1)
chk('global distinct peaks', 30636, float(po.loc['global', 'distinct_peaks_median']), 1)

# the connectivity controls ran in all three references, not the healthy one alone
for f in ('confound_programs.json', 'confound_programs_aml_malignant_v3.json',
          'confound_programs_aml_all_v3.json'):
    cvk = json.load(open(f'{R}/{f}'))['cv']
    for arm in ('peak density only', 'gene span only', 'sum_r only', 'n_links only'):
        chk(f'{f[18:26]} has {arm[:12]}', 1, int(arm in cvk), 0)
    chk(f'{f[18:26]} linkcount draws', 20,
        sum(1 for k in cvk if k.startswith('linkcount-rand')), 0)

# the two expression-variance controls are matched to the published set and are drawn from
# the whole expression matrix, not from the eligible pool
_expr = pd.read_parquet('/home/holiday01/2026_cm/data/interim/expr_norm_346.parquet')
_spec = pd.read_parquet('/home/holiday01/2026_cm/data/interim/specimens_dx.parquet')
_expr = _expr.drop_duplicates('display_label').set_index('display_label')
_cols = [c for c in _expr.columns if c in set(_spec.dbgap_rnaseq_sample)]
_X = _expr.loc[:, _cols].to_numpy(dtype=float).T
_EX = {g: i for i, g in enumerate(_expr.index)}
chk('expression matrix genes', 22843, _X.shape[1], 0)
_z = np.load('/home/holiday01/2026_cm/data/programs/programs.npz', allow_pickle=True)
_pg = _z['genes'].astype(str); _elig = set(_pg[_z['eligible']])
chk('healthy eligible pool', 7778, len(_elig), 0)
_rm = pd.read_csv('/home/holiday01/2026_cm/data/ref/published_signatures/ng2016_resolution_map.csv')
_pub = np.array([_EX[g] for g in dict.fromkeys(_rm.beataml.dropna()) if g in _EX])
chk('published resolved', 99, len(_pub), 0)
_var = _X.var(0)
_dec = pd.qcut(pd.Series(_var), 10, labels=False, duplicates='drop').to_numpy()
_jH = list(_z['cell_types'].astype(str)).index('HSC')
_s = _z['W_A'][:, _jH]
_armA = {_pg[i] for i in np.argsort(-_s)[:99] if _s[i] > 0 and _pg[i] in _EX}
_rng = np.random.default_rng(11)
_need = pd.Series(_dec[_pub]).value_counts().to_dict()
_inv = {i: g for g, i in _EX.items()}
_off, _ov = [], []
for _ in range(20):          # mirrors baseline_lsc104.py exactly, same seed and order
    _pick = []
    for d, k in _need.items():
        _pool = np.setdiff1d(np.flatnonzero(_dec == d), _pub)
        _pick.append(_rng.choice(_pool, min(k, len(_pool)), replace=False))
    _nm = {_inv[j] for j in np.concatenate(_pick)}
    _off.append(len(_nm - _elig)); _ov.append(len(_nm & _armA))
chk('var draws outside pool min', 65, min(_off), 0)
chk('var draws outside pool max', 79, max(_off), 0)
chk('var draws armA overlap min', 0, min(_ov), 0)
chk('var draws armA overlap max', 3, max(_ov), 0)
chk('top99 variance in pool', 16, len({_inv[j] for j in np.argsort(-_var)[:99]} & _elig), 0)
# the subset analysis range, which no assertion covered and which had drifted
dt = pd.read_csv(f'{R}/distributed_test.csv')
gaps = []
for C in ('cv_0.01', 'cv_0.1'):
    for sz in sorted(dt['size'].unique()):
        q = dt[dt['size'] == sz].set_index('arm')
        gaps.append(float(q.loc['random-in-A', C] - q.loc['random-in-pool', C]))
chk('subset gap min', 0.055, min(gaps), 6e-4)
chk('subset gap max', 0.072, max(gaps), 6e-4)
chk('subset gap all positive', 10, sum(g > 0 for g in gaps), 0)
# the figure draws C=0.01 alone, so the legend and the results sentence quote that penalty
_g01 = []
for sz in sorted(dt['size'].unique()):
    q = dt[dt['size'] == sz].set_index('arm')
    _g01.append(float(q.loc['random-in-A', 'cv_0.01'] - q.loc['random-in-pool', 'cv_0.01']))
chk('subset gap min C=0.01', 0.055, min(_g01), 6e-4)
chk('subset gap max C=0.01', 0.068, max(_g01), 6e-4)
# random subsets against the top-ranked subset of the same size
for C, want in (('cv_0.01', 0.023), ('cv_0.1', 0.031)):
    d_ = []
    for sz in sorted(dt['size'].unique()):
        q = dt[dt['size'] == sz].set_index('arm')
        d_.append(abs(float(q.loc['random-in-A', C] - q.loc['top-k', C])))
    chk(f'random within 0.0x of top-k {C}', want, max(d_), 1e-3)
    chk(f'random within the quoted bound {C}', 1, int(max(d_) <= want), 0)
# at 25 genes the top-ranked subset is the weakest at C=0.01 and the strongest at C=0.1
_q25 = dt[dt['size'] == 25].set_index('arm')
chk('25 genes top-k weakest at C=0.01', 1,
    int(_q25.loc['top-k', 'cv_0.01'] < min(_q25.loc['bottom-k', 'cv_0.01'],
                                           _q25.loc['random-in-A', 'cv_0.01'])), 0)
chk('25 genes top-k strongest at C=0.1', 1,
    int(_q25.loc['top-k', 'cv_0.1'] > max(_q25.loc['bottom-k', 'cv_0.1'],
                                          _q25.loc['random-in-A', 'cv_0.1'])), 0)
section('pass6')

# ---- the denominators and counts quoted in the Table 1 note ---------------------------
import re as _re
_spec = pd.read_parquet('/home/holiday01/2026_cm/data/interim/specimens_dx.parquet')
_cl = (pd.read_excel('/home/holiday01/2026_cm/data/raw/beataml2/beataml_wv1to4_clinical.xlsx')
         .drop_duplicates('dbgap_rnaseq_sample').set_index('dbgap_rnaseq_sample'))
_S = _spec.set_index('dbgap_rnaseq_sample'); _i = _S.index
_n = lambda s: pd.to_numeric(s, errors='coerce')
_note = ' '.join(open('/home/holiday01/2026_cm/manuscript/table_notes.py').read().split())
_m = _re.search(r'Age is recorded for (\d+) of the (\d+) specimens, bone-marrow blasts for '
                r'(\d+) and the white cell count for (\d+)', _note)
chk('note parses', 1, int(_m is not None), 0)
if _m:
    chk('note age n', int(_m.group(1)), int(_n(_cl.reindex(_i).ageAtDiagnosis).notna().sum()), 0)
    chk('note cohort n', int(_m.group(2)), len(_i), 0)
    chk('note blast n', int(_m.group(3)), int(_n(_cl.reindex(_i)['%.Blasts.in.BM']).notna().sum()), 0)
    chk('note wbc n', int(_m.group(4)), int(_n(_cl.reindex(_i).wbcCount).notna().sum()), 0)
_m2 = _re.search(r'Kaplan-Meier estimates use the (\d+)\s*specimens with a follow-up time '
                 r'above zero', _note)
chk('note KM parses', 1, int(_m2 is not None), 0)
if _m2:
    chk('note KM n', int(_m2.group(1)), int((_n(_S.overallSurvival) > 0).sum()), 0)
# the note separates the two denominators, so check each against the row it describes
_t1 = pd.read_csv('/home/holiday01/2026_cm/manuscript/Table1_cohort.csv').set_index('Characteristic')
_dead = _S.vitalStatus.astype(str).str.lower().eq('dead')
_dn, _dp = _re.match(r'(\d+) \((\d+)%\)', _t1.loc['Deaths', 'All diagnostic specimens']).groups()
chk('table deaths n', int(_dn), int(_dead.sum()), 0)
chk('table deaths per cent', int(_dp), round(100 * _dead.mean()), 0)
chk('every labelled specimen has follow-up', int(_S.induction_response.notna().sum()),
    int(((_n(_S.overallSurvival) > 0) & _S.induction_response.notna()).sum()), 0)
# the blast fraction the Introduction quotes
_intro = ' '.join(open('/home/holiday01/2026_cm/manuscript/introduction.md').read().split())
_m5 = _re.search(r'a median of (\d+)% bone-marrow blasts in this cohort', _intro)
chk('introduction blast median parses', 1, int(_m5 is not None), 0)
if _m5:
    chk('introduction blast median', int(_m5.group(1)),
        float(_n(_cl.reindex(_i)['%.Blasts.in.BM']).median()), 0.5)
_m3 = _re.search(r'includes the (\d+) specimens whose source label is Complete Response i', _note)
chk('note CRi parses', 1, int(_m3 is not None), 0)
if _m3:
    chk('note CRi n', int(_m3.group(1)),
        int((_cl.reindex(_i)['responseToInductionTx'] == 'Complete Response i').sum()), 0)
_m4 = _re.search(r'of which (\d+) carry a binary\s*induction-response label', _note)
chk('note labelled parses', 1, int(_m4 is not None), 0)
if _m4:
    chk('note labelled n', int(_m4.group(1)), int(_S.induction_response.notna().sum()), 0)
section('pass7')

# ---- eighth pass: the published-set, component, cross-reference and gene-level numbers
#      that no assertion covered before this round -------------------------------------
L17 = json.load(open(f'{R}/baseline_lsc17.json'))
for C, d, pv, xw in ((0.01, 0.087, 0.0003, 0.136), (0.1, 0.099, 0.0008, 0.083)):
    chk(f'A-pub cv {C}', d, P('armA_99 vs published_LSC', C))
    chk(f'A-pub p {C}', pv, P('armA_99 vs published_LSC', C, 'nb_p'))
    chk(f'A-pub xw {C}', xw, P('armA_99 vs published_LSC', C, 'xwave_diff'))
for C, d, pv in ((0.01, 0.064, 0.012), (0.1, 0.058, 0.069)):
    chk(f'R-pub cv {C}', d, P('armR_99 vs published_LSC', C))
    chk(f'R-pub p {C}', pv, P('armR_99 vs published_LSC', C, 'nb_p'))
chk('A-pub boot lo', 0.045, P('armA_99 vs published_LSC', 0.01, 'boot_lo'))
chk('A-pub boot hi', 0.236, P('armA_99 vs published_LSC', 0.01, 'boot_hi'))
for C, v in ((0.01, 0.750), (0.1, 0.740)):
    chk(f'armR99 {C}', v, L['cv']['HSC arm R top-99 (RNA)'][str(C)])
for C, v in ((0.01, 0.711), (0.1, 0.699)):
    chk(f'top99 by variance {C}', v, L['cv']['top-99 by variance'][str(C)])
_vm = [v for k, v in L['cv'].items() if k.startswith('var-matched random')]
chk('var-matched draws', 20, len(_vm), 0)
for C, v in ((0.01, 0.714), (0.1, 0.690)):
    chk(f'var-matched mean {C}', v, float(np.mean([d[str(C)] for d in _vm])))
for nm, key, vals in (('armR17', 'HSC arm R top-17 (RNA)', (0.681, 0.676)),
                      ('armA17', 'HSC arm A top-17 (chromatin)', (0.627, 0.655))):
    for C, v in zip((0.01, 0.1), vals):
        chk(f'{nm} {C}', v, L17['cv'][key][str(C)])
for C, d, pv in ((0.01, -0.035, 0.35), (0.1, -0.015, 0.70)):
    chk(f'A17-LSC17 {C}', d, P('armA_17 vs LSC17', C))
    chk(f'A17-LSC17 p {C}', pv, P('armA_17 vs LSC17', C, 'nb_p'), 6e-3)
chk('locus_mean 0.1', 0.731, lo['cv']['locus_mean (no linking)']['0.1'])
chk('armA200 full 0.1', 0.769, L17['cv']['HSC arm A 200 (ours, full)']['0.1'])
for C, pv, d in ((0.01, 0.13, 0.002), (0.1, 0.21, 0.008)):
    chk(f'A-locus p {C}', pv, P('armA_200 vs locus_mean_200', C, 'nb_p'), 6e-3)
    chk(f'locus-R {C}', d, P('locus_mean_200 vs armR_200', C))
for C, sd in (('0.01', 0.012), ('0.1', 0.019)):
    chk(f'win sd {C}', sd, lpw[f'window_C{C}']['null_cv_sd'])
_lk = rr[(rr.analysis == 'link_count') & (rr.reference == 'BMMC')].set_index('C')
for C, d, pv in ((0.01, 0.062, 0.003), (0.1, 0.068, 0.023)):
    chk(f'A-linkcount {C}', d, float(_lk.loc[C, 'diff']))
    chk(f'A-linkcount p {C}', pv, float(_lk.loc[C, 'nb_p']))
_xa = pd.read_csv(f'{R}/cross_reference_aggregate.csv').set_index('cell')
for cell, mu in (('BMMC C=0.01', -0.001), ('BMMC C=0.1', 0.005),
                 ('AML-mal C=0.01', 0.025), ('AML-mal C=0.1', 0.031)):
    chk(f'xref mean {cell}', mu, float(_xa.loc[cell, 'mean']))
chk('xref wilcoxon BMMC C=0.1', 0.44, float(_xa.loc['BMMC C=0.1', 'wilcoxon_p']), 6e-3)
_xp = pd.read_csv(f'{R}/cross_reference_programs.csv').set_index('bmmc')
chk('CD4 naive healthy', -0.0385, float(_xp.loc['CD4+ T naive', 'b001']))
chk('CD4 naive malignant', 0.0972, float(_xp.loc['CD4+ T naive', 'a001']))
# the two compartment definitions. The means, the gap and the exact P value are read out
# of the text in pass10; what is left here is the count stated in words, which no regular
# expression over the digits can reach, and the tail count behind the P value.
_cp = pd.read_csv(f'{R}/cross_reference_compartment.csv').set_index('definition')
for d_, ap, tail, nass in (('myeloid stem/progenitor', 3, 15, 680),
                           ('progenitor, lymphoid included', 3, 34, 2380)):
    chk(f'{d_[:11]} allpos', ap, int(_cp.loc[d_, 'allpos_stem']), 0)
    chk(f'{d_[:11]} tail count', tail, int(_cp.loc[d_, 'tail_count']), 0)
    chk(f'{d_[:11]} assignments', nass, int(_cp.loc[d_, 'n_assignments']), 0)
    chk(f'{d_[:11]} ties', 1, int(_cp.loc[d_, 'ties']), 0)
_ov = pd.read_csv(f'{R}/enrich_lsc_overlap.csv')
def OV(ref, gs, col):
    r = _ov[(_ov.reference == ref) & (_ov.gene_set == gs)]
    return float(r[col].iloc[0]) if len(r) else None
chk('LSC104 in pool', 73, OV('Ng2016_LSC104', 'armA_HSC', 'n_ref_in_pool'), 0)
chk('LSC17 in pool', 11, OV('Ng2016_LSC17', 'armA_HSC', 'n_ref_in_pool'), 0)
for gs, ov, odds, pv in (('armA_HSC', 12, 7.9, 2.8e-7), ('armR_HSC', 11, 7.1, 2.2e-6)):
    chk(f'LSC104 overlap {gs}', ov, OV('Ng2016_LSC104', gs, 'overlap'), 0)
    chk(f'LSC104 odds {gs}', odds, OV('Ng2016_LSC104', gs, 'odds'), 0.05)
    chk(f'LSC104 p {gs}', pv, OV('Ng2016_LSC104', gs, 'p'), 5e-8)
chk('LSC17 overlap armA', 5, OV('Ng2016_LSC17', 'armA_HSC', 'overlap'), 0)
chk('LSC17 overlap armR', 4, OV('Ng2016_LSC17', 'armR_HSC', 'overlap'), 0)
_E = pd.read_csv(f'{R}/enrich_enrichr.csv')
NSIG = lambda lib, gs: int(((_E.library == lib) & (_E.gene_set == gs) & (_E.q < 0.05)).sum())
chk('ChEA armR sig', 186, NSIG('ChEA_2022', 'armR_HSC'), 0)
chk('ChEA armA sig', 120, NSIG('ChEA_2022', 'armA_HSC'), 0)
chk('GO armA sig', 9, NSIG('GO_Biological_Process_2023', 'armA_HSC'), 0)
chk('GO armR sig', 2, NSIG('GO_Biological_Process_2023', 'armR_HSC'), 0)
chk('A99 shared with published', 11, len(_armA & {_inv[j] for j in _pub}), 0)
# the two scores against each other and against the abundance readouts
_sv = json.load(open(f'{R}/state_vs_abundance_summary.json'))
chk('score correlation', 0.919, _sv['rho_armA_armR'])
chk('blast n', 227, _sv['blast_n'], 0)
chk('blast rho', -0.065, _sv['blast_rho_armA'])
chk('blast rho p', 0.33, _sv['blast_rho_armA_p'], 6e-3)
chk('blast alone auroc', 0.450, _sv['blast_alone_auroc'])
chk('steiger min', 0.11, _sv['steiger_p_min'], 5e-3)
section('pass8')

# ---- tenth pass: claims read out of the manuscript rather than transcribed here --------
# Every assertion above holds a number copied into this file, so an edit to the manuscript
# that leaves the old number here still passes. A claim() instead reads the value from the
# sentence that states it and compares that with a source field, keeping the sign, the
# exponent and the precision the sentence uses, and failing if the sentence no longer
# matches at all.
_SUP = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7',
        '⁸': '8', '⁹': '9', '⁻': '-'}
_SCI = _re.compile(r'^(-?[\d.]+)\s*×\s*10\s*([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)$')


def _mant_exp(s):
    s = s.strip().replace('−', '-').replace('–', '-').replace('+', '').replace(',', '')
    m = _SCI.match(s)
    if m:
        return m.group(1), int(''.join(_SUP.get(c, c) for c in m.group(2)))
    return s, 0


def num(s):
    """A number as the manuscript writes it, Unicode minus and 10-exponents included."""
    mant, exp = _mant_exp(s)
    return float(mant) * 10.0 ** exp


def ulp(s):
    """Half a unit in the last place the manuscript wrote, so the quoted rounding passes
    and any other digit at that place does not."""
    mant, exp = _mant_exp(s)
    nd = len(mant.split('.')[1]) if '.' in mant else 0
    return 0.51 * 10.0 ** (exp - nd)


_TXT = {}


def mtext(f):
    if f not in _TXT:
        _TXT[f] = ' '.join(open(f'/home/holiday01/2026_cm/manuscript/{f}').read().split())
    return _TXT[f]


COVERED = {}


def claim(f, pattern, *sources, label=None):
    """Read the pattern's groups from the manuscript and compare them with the sources."""
    lab = label or (pattern[:30] + '…')
    hits = list(_re.finditer(pattern, mtext(f)))
    if len(hits) != 1:
        BAD.append((f'{lab} [matched {len(hits)}x]', 1, float(len(hits)))); return
    m = hits[0]
    COVERED.setdefault(f, []).append((m.start(), m.end()))
    got = m.groups()
    if len(got) != len(sources):
        BAD.append((f'{lab} [groups]', float(len(sources)), float(len(got)))); return
    for i, (g, src) in enumerate(zip(got, sources)):
        chk(f'{lab} [{i + 1}]', num(g), None if src is None else float(src), ulp(g))


def bound(f, pattern, source, label=None):
    """A sentence that states a bound rather than a value. The source must lie under the
    figure quoted and close enough to it that the bound is not vacuous."""
    lab = label or (pattern[:30] + '…')
    hits = list(_re.finditer(pattern, mtext(f)))
    if len(hits) != 1:
        BAD.append((f'{lab} [matched {len(hits)}x]', 1, float(len(hits)))); return
    m = hits[0]
    COVERED.setdefault(f, []).append((m.start(), m.end()))
    q = num(m.group(1))
    chk(f'{lab} [holds]', 1.0, float(source <= q), 0)
    chk(f'{lab} [tight]', 1.0, float(source > q - 10 * ulp(m.group(1))), 0)


_N = r'([+−-]?\d*\.?\d+)'                      # a number as the text may sign it
_SCIN = r'([\d.]+\s*×\s*10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)'     # a P value in scientific notation
_S = json.load(open(f'{R}/state_vs_abundance_summary.json'))
_CP = pd.read_csv(f'{R}/cross_reference_compartment.csv').set_index('definition')
_XA = pd.read_csv(f'{R}/cross_reference_aggregate.csv').set_index('cell')
_OVL = pd.read_csv(f'{R}/enrich_lsc_overlap.csv').set_index(['reference', 'gene_set'])
_OS = rr[rr.analysis == 'overall_survival']
_SP = pd.read_parquet('/home/holiday01/2026_cm/data/interim/specimens_dx.parquet')
_RESP = _SP.induction_response

# the cohort, as the abstract states it
claim('abstract_v3.md',
      r'analysed ' + _N + r' diagnostic Beat AML specimens, ' + _N +
      r' with documented induction\s*response \(' + _N + r' complete remission, ' + _N +
      r' refractory\)',
      len(_SP), _RESP.notna().sum(), (_RESP == 'CR').sum(), (_RESP == 'Refractory').sum(),
      label='abstract cohort')
# the two ranges and the two counts of the headline comparison
_a = rb[rb.arm == 'armA_sum']
claim('abstract_v3.md', r'by \+' + _N + r' to \+' + _N + r' in the area under',
      _a.cv_diff.min(), _a.cv_diff.max(), label='abstract cv range')
claim('abstract_v3.md', r'in ten of twelve held-out comparisons, from ' + _N + r' to \+' + _N,
      _a.xwave_diff.min(), _a.xwave_diff.max(), label='abstract xwave range')
claim('abstract_v3.md',
      r'arm A reached ' + _N + r' and arm R ' + _N + r' against its ' + _N,
      L['cv']['HSC arm A top-99 (chromatin)']['0.01'],
      L['cv']['HSC arm R top-99 (RNA)']['0.01'],
      L['cv']['Ng2016 LSC genes (published, experimental)']['0.01'], label='abstract 99-gene')
claim('abstract_v3.md',
      r'arm A added \+' + _N + r' AUROC \(\*P\* = ' + _N + r'\)',
      C_('standard panel + chromatin - standard panel', 0.01),
      Cp('standard panel + chromatin - standard panel', 0.01), label='abstract panel')
claim('abstract_v3.md',
      r'ELN 2017 risk category \+' + _N + r' \(\*P\* = ' + _N + r'\)',
      C_('ELN2017 + chromatin - ELN2017', 0.01),
      Cp('ELN2017 + chromatin - ELN2017', 0.01), label='abstract ELN')

# the blast readouts, the sentence this round corrected
claim('results.md',
      r'blast percentage at Spearman ρ =\s*' + _N + r' \(\*P\* = ' + _N + r'\) in the ' + _N +
      r' specimens with a recorded percentage',
      _S['blast_rho_armA'], _S['blast_rho_armA_p'], _S['blast_n'], label='blast rho')
claim('results.md',
      r'higher values scored as refractory, separated the two response groups at AUROC ' + _N,
      _S['blast_alone_auroc'], label='blast AUROC')
claim('results.md', r'the two scores correlated at Spearman ρ = ' + _N, _S['rho_armA_armR'],
      label='score correlation')
claim('results.md', r'all\s*Steiger \*P\* ≥ ' + _N, _S['steiger_p_min'], label='steiger min')

# the compartment comparison, whose P values are exact tail counts
for _lbl, _d in (('myeloid stem/progenitor', 'narrow'),
                 ('progenitor, lymphoid included', 'broad')):
    _r = _CP.loc[_lbl]
    if _d == 'narrow':
        claim('results.md',
              r'programmes led by \+' + _N + r' against \+' + _N +
              r' for the remaining fourteen, a gap of \+' + _N +
              r'\s*\(label assignments enumerated, one-sided \*P\* = ' + _N + r'\)',
              _r.mean_stem, _r.mean_rest, _r.gap, _r.p_one_sided, label='compartment narrow')
    else:
        claim('results.md',
              r'the figures are \+' + _N + r' against \+' + _N + r', a gap of \+' + _N +
              r'\s*\(one-sided \*P\* = ' + _N + r'\)',
              _r.mean_stem, _r.mean_rest, _r.gap, _r.p_one_sided, label='compartment broad')

# the published-set comparison
claim('results.md',
      r'exceeding the published set by ' + _N + r' and ' + _N + r' \(\*P\* =\s*' + _N +
      r' and ' + _N + r'\) and by ' + _N + r' and ' + _N,
      P('armA_99 vs published_LSC', 0.01), P('armA_99 vs published_LSC', 0.1),
      P('armA_99 vs published_LSC', 0.01, 'nb_p'), P('armA_99 vs published_LSC', 0.1, 'nb_p'),
      P('armA_99 vs published_LSC', 0.01, 'xwave_diff'),
      P('armA_99 vs published_LSC', 0.1, 'xwave_diff'), label='armA vs published')
claim('results.md',
      r'expression specificity reached ' + _N + r' and ' + _N + r',\s*exceeding the published '
      r'set by ' + _N + r' and ' + _N + r' \(\*P\* = ' + _N + r' and ' + _N + r'\)',
      L['cv']['HSC arm R top-99 (RNA)']['0.01'], L['cv']['HSC arm R top-99 (RNA)']['0.1'],
      P('armR_99 vs published_LSC', 0.01), P('armR_99 vs published_LSC', 0.1),
      P('armR_99 vs published_LSC', 0.01, 'nb_p'), P('armR_99 vs published_LSC', 0.1, 'nb_p'),
      label='armR vs published')
claim('results.md', r'bootstrap interval was \+' + _N + r' to \+' + _N,
      P('armA_99 vs published_LSC', 0.01, 'boot_lo'),
      P('armA_99 vs published_LSC', 0.01, 'boot_hi'), label='armA boot interval')

# the overlap with the published gene list, including the exponents
claim('results.md',
      r'with ' + _N + r' of ' + _N + r' for the\s*accessibility rule \(odds ratio ' + _N +
      r', \*P\* = ' + _SCIN + r'\) and ' + _N + r' of ' + _N + r' for the expression\s*rule '
      r'\(odds ratio ' + _N + r', \*P\* = ' + _SCIN + r'\)',
      _OVL.loc[('Ng2016_LSC104', 'armA_HSC'), 'overlap'],
      _OVL.loc[('Ng2016_LSC104', 'armA_HSC'), 'n_ref_in_pool'],
      _OVL.loc[('Ng2016_LSC104', 'armA_HSC'), 'odds'],
      _OVL.loc[('Ng2016_LSC104', 'armA_HSC'), 'p'],
      _OVL.loc[('Ng2016_LSC104', 'armR_HSC'), 'overlap'],
      _OVL.loc[('Ng2016_LSC104', 'armR_HSC'), 'n_ref_in_pool'],
      _OVL.loc[('Ng2016_LSC104', 'armR_HSC'), 'odds'],
      _OVL.loc[('Ng2016_LSC104', 'armR_HSC'), 'p'], label='LSC104 overlap')
claim('results.md', r'ChEA terms at FDR < 0.05 \(' + _N + r' versus ' + _N + r'\)',
      int(((_E.library == 'ChEA_2022') & (_E.gene_set == 'armR_HSC') & (_E.q < 0.05)).sum()),
      int(((_E.library == 'ChEA_2022') & (_E.gene_set == 'armA_HSC') & (_E.q < 0.05)).sum()),
      label='ChEA counts')

# overall survival
claim('results.md',
      r'exceeded the\s*expression rule in ' + _N + r' of 12 cross-validated comparisons of '
      r"Harrell's concordance index,\s*with differences between " + _N + r' and \+' + _N,
      int((_OS['diff'] > 0).sum()), _OS['diff'].min(), _OS['diff'].max(), label='survival')
# The compartment comparison is described in two places in the Methods, so both have to
# agree with the artefact, which records an exhaustive enumeration rather than a sample.
# A stale sampling count survived one round in the first of them while the second was
# already correct.
_mt = mtext('methods.md')
_sent = [s for s in _re.split(r'(?<=[.;]) ', _mt) if 'grouped and compared with the remaining' in s]
chk('compartment sentence found', 1, float(len(_sent)), 0)
if _sent:
    chk('compartment described as enumerated', 1.0, float('enumerat' in _sent[0]), 0)
    chk('compartment not described as sampled', 0.0,
        float(bool(_re.search(r'draws|permut', _sent[0]))), 0)
claim('methods.md',
      r'matched\s*programmes, ' + _N + r' for the three-programme definition and ' + _N +
      r' for the\s*four-programme definition',
      _CP.loc['myeloid stem/progenitor', 'n_assignments'],
      _CP.loc['progenitor, lymphoid included', 'n_assignments'], label='enumeration counts')

# The two subsets of the labelled cohort are quoted in the Methods, the Results and the
# Figure 1 legend and drawn in the figure. They have one producer, so all four are read
# from it rather than from a count written into the figure script.
_CC = json.load(open(f'{R}/clinical_cohort.json'))
chk('subset labelled', _CC['labelled'], float(_RESP.notna().sum()), 0)
chk('subset waves sum', _CC['labelled'],
    float(_CC['wave_train'] + _CC['wave_test'] + _CC['wave_both']), 0)
claim('methods.md',
      r'waves 1–2 as training \(n = ' + _N + r'; ' + _N + r' complete remission, ' + _N +
      r' refractory\) and waves 3–4 as test \(n = ' + _N + r'; ' + _N +
      r' complete remission, ' + _N + r' refractory\)',
      _CC['wave_train'], _CC['wave_train_cr'], _CC['wave_train_refractory'],
      _CC['wave_test'], _CC['wave_test_cr'], _CC['wave_test_refractory'],
      label='methods wave split')
claim('methods.md',
      r'Clinical covariate models used the ' + _N + r' patients with complete age',
      _CC['clinical_complete_cases'], label='methods clinical n')
claim('methods.md',
      r'not reassigned here \(' + _N + r' complete remission, ' + _N + r' refractory; ' +
      _N + r' favourable, ' + _N + r' intermediate, ' + _N + r' adverse\)',
      _CC['clinical_cr'], _CC['clinical_refractory'], _CC['clinical_eln_favourable'],
      _CC['clinical_eln_intermediate'], _CC['clinical_eln_adverse'],
      label='methods clinical composition')
claim('results.md',
      r'In the ' + _N + r' patients with complete covariate data and an unambiguous ELN '
      r'2017 category \(' + _N + r' complete remission, ' + _N + r' refractory\)',
      _CC['clinical_complete_cases'], _CC['clinical_cr'], _CC['clinical_refractory'],
      label='results clinical n')
claim('figure_legends.md',
      r'uses the ' + _N + r' training and ' + _N + r' test specimens defined by collection '
      r'wave, with the ' + _N + r' specimens annotated to both wave groups excluded from '
      r'that split, and the clinical models use the ' + _N + r' patients',
      _CC['wave_train'], _CC['wave_test'], _CC['wave_both'],
      _CC['clinical_complete_cases'], label='legend subsets')
# the count comes from a second script, so the two inclusion masks are compared line by
# line; a covariate added to one of them and not the other would otherwise pass unseen
def _mask(path):
    src = open(f'/home/holiday01/2026_cm/scripts/{path}').read()
    return [''.join(l.split()) for l in src.splitlines()
            if l.lstrip().startswith(("C['", 'eln=', 'eln =', 'ok=', 'ok ='))]
chk('clinical mask shared', 1.0,
    float(_mask('clinical_cohort_counts.py') == _mask('recompute_remaining.py')), 0)
chk('clinical mask non-empty', 8, float(len(_mask('clinical_cohort_counts.py'))), 0)

_FIG = open('/home/holiday01/2026_cm/scripts/make_flow_figure.py').read()
chk('figure reads the subset sizes', 1.0, float('clinical_cohort.json' in _FIG), 0)
chk('figure carries no written-in count', 0.0,
    float(bool(_re.search(r'(?<![\d.])(580|458|450|346|254|194|150|96)(?![\d.])', _FIG))), 0)

# the cover letter repeats the headline numbers, so it is read against the same sources
claim('cover_letter.md',
      r'are ' + _N + r' diagnostic Beat AML specimens, ' + _N + r' of them with a',
      len(_SP), _RESP.notna().sum(), label='letter cohort')
claim('cover_letter.md', r'by \+' + _N + r' to\s*\+' + _N + r' in the area under',
      _a.cv_diff.min(), _a.cv_diff.max(), label='letter cv range')
claim('cover_letter.md',
      r'accessibility rule reached AUROC ' + _N + r' and the\s*expression rule ' + _N +
      r', against ' + _N + r' for the published set',
      L['cv']['HSC arm A top-99 (chromatin)']['0.01'],
      L['cv']['HSC arm R top-99 (RNA)']['0.01'],
      L['cv']['Ng2016 LSC genes (published, experimental)']['0.01'], label='letter 99-gene')
claim('cover_letter.md',
      r'an increment of \+' + _N + r' \(\*P\* = ' + _N + r'\), and\s*added to the ELN 2017 '
      r'risk category \+' + _N + r' \(\*P\* = ' + _N + r'\)',
      C_('standard panel + chromatin - standard panel', 0.01),
      Cp('standard panel + chromatin - standard panel', 0.01),
      C_('ELN2017 + chromatin - ELN2017', 0.01),
      Cp('ELN2017 + chromatin - ELN2017', 0.01), label='letter clinical')

# the two sentences that state how close the link-count arm comes, as a bound
_lkx = rr[(rr.analysis == 'link_count') & (rr.reference != 'BMMC')]
for _f, _pat in (('results.md', r'the two were within ' + _N + r' of each other'),
                 ('discussion.md', r'within ' + _N + r' of it in both leukaemia')):
    bound(_f, _pat, float(_lkx['diff'].abs().max()), label=f'{_f[:4]} link-count bound')
section('pass10')

# ---- ninth pass: which decimals in the text are checked, and which are only present ----
# A decimal is bound if a claim() in pass10 read it out of the sentence that states it.
# Otherwise this pass can only ask whether the value appears as a literal somewhere in this
# file, which a comment or an unrelated tolerance also satisfies, so a literal-only number
# is reported as weakly covered rather than treated as checked. Uncovered decimals fail.
_MD = ('abstract_v3.md', 'introduction.md', 'methods.md', 'results.md', 'discussion.md',
       'figure_legends.md', 'supplementary_methods.md', 'cover_letter.md')
# An exemption is bound to the sentence that uses the number and to the role it plays
# there, not to the value, so that a result which happens to equal a penalty or a
# threshold is not waved through.
_EXEMPT = (
    r'C (?:∈|in) \{0\.001, 0\.01, 0\.1, 1\}',              # the penalty grid
    r'Python [\d.]+ \((?:[A-Za-z-]+ [\d.]+(?:, )?)+\)',     # the library versions
    r'(?:\*P\*|P) [<≥] 0\.05', r'FDR < 0\.05',             # significance thresholds
    r'false discovery rate below 0\.25',
    r'window 100, 250 or 500 kb, threshold 0\.20, 0\.30 or 0\.40',
    r'correlation threshold of 0\.30',
    # the two cross-modal discard percentages are printed by the reference build and are
    # not re-derivable without rebuilding the reference, which would move every result
    r'a further 8\.5% and 9\.1%')
_self = open(__file__).read()   # the running copy, so a comment cannot be added unseen
_lits = [float(x) for x in _re.findall(r'(?<![\w.])\d+\.\d+', _self)]
_bound = _weak = 0
_uncovered = []
_exempt_spans = {_f: [(_m.start(), _m.end()) for _p in _EXEMPT
                      for _m in _re.finditer(_p, mtext(_f))] for _f in _MD}
for _f in _MD:
    _t = mtext(_f)
    for _m in _re.finditer(r'(?<![\w.])(\d+\.\d+)', _t):
        _s = _m.group(1)
        if any(_a <= _m.start() < _b for _a, _b in COVERED.get(_f, ())):
            _bound += 1
            continue
        if any(_a <= _m.start() < _b for _a, _b in _exempt_spans[_f]):
            continue
        _nd = len(_s.split('.')[1])
        if any(abs(round(_v, _nd) - float(_s)) < 1e-12 for _v in _lits):
            _weak += 1
            continue
        _uncovered.append(f'{_f}:{_s}')
chk('decimals with no assertion at all', 0, len(_uncovered), 0)
print(f'  {_bound} decimals read out of the text by a claim, {_weak} present only as a '
      f'literal, {len(_uncovered)} neither')
if _uncovered:
    print('  uncovered:', ', '.join(_uncovered))
section('pass9')

print(f'\ntotal: {len(ALL)} mismatch across all sections')
sys.exit(1 if ALL else 0)
