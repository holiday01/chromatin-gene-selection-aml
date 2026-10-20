"""Pull every number the Results section quotes, from the corrected (v2) outputs, so the
rewrite can be checked against one table instead of a dozen files."""
import json, numpy as np, pandas as pd
R='/home/holiday01/2026_cm/results'
def J(f):
    try: return json.load(open(f'{R}/{f}'))
    except Exception: return None
def C(f, **k):
    try: return pd.read_csv(f'{R}/{f}', **k)
    except Exception: return None
out=[]
def row(sec,item,val): out.append(dict(section=sec,item=item,value=val))

rb=C('robustness_check.csv')
if rb is not None:
    a=rb[rb.arm=='armA_sum']
    row('R1','CV diff range',f'{a.cv_diff.min():+.4f} to {a.cv_diff.max():+.4f}')
    row('R1','xwave diff range',f'{a.xwave_diff.min():+.4f} to {a.xwave_diff.max():+.4f}')
    row('R1','CV positive',f'{int((a.cv_diff>0).sum())}/12')
    row('R1','xwave positive',f'{int((a.xwave_diff>0).sum())}/12')
    row('R1','NB p<0.05',f'{int((a.nb_p<0.05).sum())}/12')
    row('R1','bootstrap excludes 0',f'{int((a.boot_lo>0).sum())}/12')
    for nm in ['armA_count_partialled','armA_mean_per_link','n_links_only']:
        v=rb[rb.arm==nm]
        row('R5',f'{nm} CV positive / NB sig',f'{int((v.cv_diff>0).sum())}/12, {int((v.nb_p<0.05).sum())}/12')
for t,l in [('programs','BMMC'),('programs_aml_malignant_v2','AML-mal'),('programs_aml_all_v2','AML-all')]:
    z=np.load(f'/home/holiday01/2026_cm/data/programs/{t}.npz',allow_pickle=True)
    ct=list(z['cell_types'].astype(str)); j=ct.index('HSC')
    A=set(np.flatnonzero(z['W_A'][:,j])); B=set(np.flatnonzero(z['W_R'][:,j]))
    row('R1',f'{l} arm overlap',f'{len(A&B)}/200'); row('R1',f'{l} programs',len(ct))
    row('R1',f'{l} eligible',f"{int(z['eligible'].sum()):,}")
bs=C('budget_sweep.csv')
if bs is not None:
    for ref in bs.reference.unique():
        s=bs[bs.reference==ref]
        okk=[int(k) for k,g in s.groupby('k') if all(g[c].iloc[0]>0 for c in ['d_cv_0.01','d_cv_0.1','d_xw_0.01','d_xw_0.1'])]
        row('R2',f'{ref} all-4-positive at k',okk)
    b=bs[bs.reference=='BMMC']
    for k in (10,17):
        r=b[b.k==k]
        if len(r): row('R2',f'BMMC k={k} cv',f"{r['d_cv_0.01'].iloc[0]:+.4f} / {r['d_cv_0.1'].iloc[0]:+.4f}")
L=J('baseline_lsc104.json')
if L:
    for k in L['cv']:
        if not k.startswith('var-matched'):
            row('R3',f'{k} cv',f"{L['cv'][k]['0.01']:.3f} / {L['cv'][k]['0.1']:.3f}")
            row('R3',f'{k} xwave',f"{L['xwave'][k]['0.01']:.3f} / {L['xwave'][k]['0.1']:.3f}")
rp=C('recomputed_pvalues.csv')
if rp is not None:
    for _,r in rp.iterrows():
        row('R3/R5',f"{r.comparison} C={r.C:g}",f"{r.cv_diff:+.4f} p={r.nb_p:.4f} xw={r.xwave_diff:+.4f} CI[{r.boot_lo:+.3f},{r.boot_hi:+.3f}]")
rr=C('recomputed_remaining.csv')
if rr is not None:
    for _,r in rr[rr.analysis=='clinical'].iterrows():
        row('R4',f"{r.comparison} C={r.C:g}",f"{r['diff']:+.4f} p={r.nb_p:.4f}")
    os_=rr[rr.analysis=='overall_survival']
    if len(os_):
        row('R6','OS positive',f"{int((os_['diff']>0).sum())}/12")
        for ref in os_.reference.unique():
            v=os_[os_.reference==ref]
            row('R6',f'OS {ref}',f"{v['diff'].min():+.4f} to {v['diff'].max():+.4f}, sig {int((v.nb_p<0.05).sum())}/4")
lo=J('locus_arm.json'); dt=C('distributed_test.csv')
if lo:
    for k in lo['cv']: row('R5',f'locus {k}',f"{lo['cv'][k]['0.01']:.3f} / {lo['cv'][k]['0.1']:.3f}")
if dt is not None:
    for sz in sorted(dt['size'].unique()):
        s=dt[dt['size']==sz]
        row('R5',f'subset k={sz}', ' | '.join(f"{r.arm} {r['cv_0.01']:.3f}" for _,r in s.iterrows() if not r.arm.startswith('var')))
D=pd.DataFrame(out)
D.to_csv(f'{R}/v2_numbers.csv',index=False)
pd.set_option('display.max_rows',300,'display.width',200,'display.max_colwidth',90)
print(D.to_string(index=False))
