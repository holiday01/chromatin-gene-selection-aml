"""Pull every number the Results section quotes into one table, so the text can be
checked against a single file instead of a dozen.

check_reported_numbers.py asserts the values the manuscript actually carries; this script
just lists what the outputs hold."""
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
    row('R1','bootstrap excludes 0',f'{int((a.boot_lo>1e-9).sum())}/12')
    row('R1','bootstrap lower bound at 0',f'{int(((a.boot_lo>-1e-9)&(a.boot_lo<1e-9)).sum())}/12')
    for nm in ['armA_count_partialled','armA_mean_per_link','n_links_only']:
        v=rb[rb.arm==nm]
        row('R5',f'{nm} CV positive / NB sig',f'{int((v.cv_diff>0).sum())}/12, {int((v.nb_p<0.05).sum())}/12')
for t,l in [('programs','BMMC'),('programs_aml_malignant_v3','AML-mal'),('programs_aml_all_v3','AML-all')]:
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
xa=C('cross_reference_aggregate.csv'); xp=C('cross_reference_programs.csv')
if xa is not None:
    for _,r in xa.iterrows():
        row('R-prog',f'{r.cell} sweep',
            f"positive {int(r.positive)}/{int(r.n)}, mean {r['mean']:+.4f}, "
            f"sign p={r.sign_p:.3g}, Wilcoxon p={r.wilcoxon_p:.3g}")
if xp is not None:
    from scipy.stats import spearmanr
    row('R-prog','matched programmes',f'{len(xp)}')
    row('R-prog','rank agreement',
        f'Spearman rho {spearmanr(xp.b001,xp.a001).statistic:+.3f} (C=0.01), '
        f'{spearmanr(xp.b01,xp.a01).statistic:+.3f} (C=0.1)')
    row('R-prog','sign agreement',
        f"{int(((xp.b001>0)==(xp.a001>0)).sum())}/{len(xp)} (C=0.01), "
        f"{int(((xp.b01>0)==(xp.a01>0)).sum())}/{len(xp)} (C=0.1)")
    row('R-prog','stem/progenitor vs rest',
        f"{xp[xp.stem].mean4.mean():+.4f} (n={int(xp.stem.sum())}) vs "
        f"{xp[~xp.stem].mean4.mean():+.4f} (n={int((~xp.stem).sum())}), "
        f"gap {xp[xp.stem].mean4.mean()-xp[~xp.stem].mean4.mean():+.4f}")
    row('R-prog','positive in all four CV cells',
        f"{int(xp.allpos.sum())}/{len(xp)}; stem {int(xp[xp.stem].allpos.sum())}/"
        f"{int(xp.stem.sum())}, rest {int(xp[~xp.stem].allpos.sum())}/{int((~xp.stem).sum())}")
cfl=C('cohort_flow.csv'); cex=C('cohort_flow_excluded.csv')
if cfl is not None:
    row('M1','inclusion flow',' -> '.join(str(int(n)) for n in cfl.n))
if cex is not None:
    e=cex.set_index('variable')
    for v in e.index:
        row('M1',f'excluded vs retained, {v}', f"{e.loc[v,'excluded']} vs {e.loc[v,'retained']}")
crg=C('cohort_regimens.csv')
if crg is not None:
    r_=crg.set_index('item').value
    row('M1','recorded induction regimens',
        f"{int(r_.loc['distinct first regimen entries'])} distinct first entries, "
        f"{int(r_.loc['first entry is 7+3 or 3+7'])} a cytarabine and anthracycline induction")
D=pd.DataFrame(out)
D.to_csv(f'{R}/reported_numbers.csv',index=False)
pd.set_option('display.max_rows',300,'display.width',200,'display.max_colwidth',90)
print(D.to_string(index=False))
