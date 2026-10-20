"""Table 1 (cohort) and Table 2 (enrichment, arm A beside arm R)."""
import numpy as np, pandas as pd, json
from scipy import stats
ROOT='/home/holiday01/2026_cm'
spec=pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
cl=pd.read_excel(f'{ROOT}/data/raw/beataml2/beataml_wv1to4_clinical.xlsx').drop_duplicates('dbgap_rnaseq_sample').set_index('dbgap_rnaseq_sample')
S=spec.set_index('dbgap_rnaseq_sample')
lab=S.induction_response
num=lambda s: pd.to_numeric(s,errors='coerce')
idx=S.index
D=pd.DataFrame(index=idx)
D['resp']=lab
D['age']=num(cl.reindex(idx).ageAtDiagnosis)
D['sex']=cl.reindex(idx).consensus_sex
D['blast_bm']=num(cl.reindex(idx)['%.Blasts.in.BM'])
D['wbc']=num(cl.reindex(idx).wbcCount)
D['flt3']=cl.reindex(idx)['FLT3-ITD'].astype(str).str.lower()
D['npm1']=cl.reindex(idx)['NPM1'].astype(str).str.lower()
D['wave']=S.cohort
D['os']=num(S.overallSurvival); D['dead']=S.vitalStatus.astype(str).str.lower().eq('dead')
groups=[('All diagnostic specimens',D),
        ('Complete remission',D[D.resp=='CR']),
        ('Refractory',D[D.resp=='Refractory'])]
def cont(g,c,dec=1):
    v=g[c].dropna()
    return f'{v.median():.{dec}f} ({v.quantile(.25):.{dec}f}–{v.quantile(.75):.{dec}f})' if len(v) else '—'
def cat(g,c,val):
    v=g[c].dropna(); n=(v==val).sum()
    return f'{n} ({100*n/len(v):.0f}%)' if len(v) else '—'
rows=[]
def add(label,fn): rows.append(dict(Characteristic=label,**{nm:fn(g) for nm,g in groups}))
add('n',lambda g: str(len(g)))
add('Age at diagnosis, years',lambda g: cont(g,'age'))
add('Female',lambda g: cat(g,'sex','Female'))
add('Bone-marrow blasts, %',lambda g: cont(g,'blast_bm'))
add('White cell count, ×10⁹/L',lambda g: cont(g,'wbc'))
add('FLT3-ITD positive',lambda g: cat(g,'flt3','positive'))
add('NPM1 positive',lambda g: cat(g,'npm1','positive'))
add('Waves 1–2 / 3–4 / both',lambda g: ' / '.join(str(int((g.wave==w).sum())) for w in ['Waves1+2','Waves3+4','Both']))
add('Deaths',lambda g: f'{int(g.dead.sum())} ({100*g.dead.mean():.0f}%)')
add('Median overall survival, days',lambda g: cont(g,'os',0))
T1=pd.DataFrame(rows)
# p values, CR vs refractory
cr,rf=D[D.resp=='CR'],D[D.resp=='Refractory']
pv={'Age at diagnosis, years':stats.mannwhitneyu(cr.age.dropna(),rf.age.dropna()).pvalue,
    'Bone-marrow blasts, %':stats.mannwhitneyu(cr.blast_bm.dropna(),rf.blast_bm.dropna()).pvalue,
    'White cell count, ×10⁹/L':stats.mannwhitneyu(cr.wbc.dropna(),rf.wbc.dropna()).pvalue,
    'Median overall survival, days':stats.mannwhitneyu(cr.os.dropna(),rf.os.dropna()).pvalue}
for c,v in [('Female','sex'),('FLT3-ITD positive','flt3'),('NPM1 positive','npm1')]:
    tgt={'sex':'Female','flt3':'positive','npm1':'positive'}[v]
    tab=[[(cr[v]==tgt).sum(),(cr[v].notna()&(cr[v]!=tgt)).sum()],
         [(rf[v]==tgt).sum(),(rf[v].notna()&(rf[v]!=tgt)).sum()]]
    pv[c]=stats.fisher_exact(tab)[1]
T1['P (CR vs refractory)']=[('' if r not in pv else (f'{pv[r]:.3f}' if pv[r]>=0.001 else '<0.001')) for r in T1.Characteristic]
T1.to_csv(f'{ROOT}/manuscript/Table1_cohort.csv',index=False)
print('=== Table 1 ===');print(T1.to_string(index=False))

# ---- Table 2 ----
E=pd.read_csv(f'{ROOT}/results/enrich_enrichr.csv'); O=pd.read_csv(f'{ROOT}/results/enrich_lsc_overlap.csv')
r2=[]
for lib,n in [('TRRUST_Transcription_Factors_2019',4),('ChEA_2022',4),('GO_Biological_Process_2023',3)]:
    for s_ in ['armA_HSC','armR_HSC']:
        g=E[(E.library==lib)&(E.gene_set==s_)].sort_values('p').head(n)
        for _,x in g.iterrows():
            r2.append(dict(Library=lib.replace('_2019','').replace('_2022','').replace('_2023','').replace('_',' '),
                           Arm='A (chromatin)' if s_=='armA_HSC' else 'R (expression)',
                           Term=x.term[:58], Overlap=int(x.n_overlap),
                           OddsRatio=round(x.odds,1) if np.isfinite(x.odds) else np.inf,
                           P=f'{x.p:.2g}', FDR=f'{x.q:.2g}'))
for _,x in O[O.gene_set.isin(['armA_HSC','armR_HSC'])].iterrows():
    r2.append(dict(Library=f'Overlap with {x.reference}',
                   Arm='A (chromatin)' if x.gene_set=='armA_HSC' else 'R (expression)',
                   Term=f'{int(x.overlap)} of {int(x.n_ref_in_pool)} in eligible pool (expected {x.expected})',
                   Overlap=int(x.overlap), OddsRatio=x.odds, P=f'{x.p:.2g}', FDR=''))
T2=pd.DataFrame(r2); T2.to_csv(f'{ROOT}/manuscript/Table2_enrichment.csv',index=False)
sig=E[E.q<0.05].groupby(['library','gene_set']).size().unstack(fill_value=0)
print('\n=== Table 2: significant terms at FDR<0.05 ===');print(sig[['armA_HSC','armR_HSC']].to_string())
print(f'\nwrote Table1_cohort.csv and Table2_enrichment.csv ({len(T2)} rows)')
