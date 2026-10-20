"""The inclusion steps from the Beat AML clinical table to the 346 analysed specimens, and
what the drug-assay criterion removed.

The criterion is inherited from the wider project and does not serve the induction-response
endpoint, so the cohort is narrower than the set of diagnostic specimens with expression
data. This script writes the intermediate denominators and compares the 104 specimens it
excluded with the 346 retained, so the effect on the study population can be read off
rather than asserted.
"""
import numpy as np, pandas as pd
ROOT = '/home/holiday01/2026_cm'
RAW = f'{ROOT}/data/raw/beataml2'

hdr = open(f'{RAW}/beataml_waves1to4_norm_exp_dbgap.txt').readline().rstrip('\n').split('\t')
expr_samples = set(hdr[4:])
cl = pd.read_excel(f'{RAW}/beataml_wv1to4_clinical.xlsx')
dx = cl[cl.diseaseStageAtSpecimenCollection == 'Initial Diagnosis']
named = dx.dropna(subset=['dbgap_rnaseq_sample'])
inmat = named[named.dbgap_rnaseq_sample.isin(expr_samples)].copy()
used = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')

def norm_resp(x):
    if not isinstance(x, str): return np.nan
    s = x.strip().lower()
    if s.startswith('complete response'): return 'CR'
    if s.startswith('refractory'): return 'Refractory'
    return np.nan
inmat['induction_response'] = inmat.responseToInductionTx.map(norm_resp)

steps = [('initial diagnosis records', len(dx)),
         ('naming an RNA-seq sample', len(named)),
         ('present in the expression matrix', len(inmat)),
         ('with an inhibitor over a complete dose series', len(used))]
print('inclusion flow')
for lbl, n in steps: print(f'  {lbl:46s} {n:5d}')

excl = inmat[~inmat.dbgap_rnaseq_sample.isin(set(used.dbgap_rnaseq_sample))]
print(f'\nexcluded by the drug-assay criterion: {len(excl)}')

def desc(d, col):
    lab = d[col]
    n = int(lab.notna().sum())
    ref = int((lab == 'Refractory').sum())
    return dict(n=len(d), labelled=n, refractory=ref,
                refractory_rate=ref / n if n else np.nan,
                median_age=float(d.ageAtDiagnosis.median()),
                pct_female=float((d.consensus_sex == 'Female').mean()),
                pct_marrow=float((d.specimenType == 'Bone Marrow Aspirate').mean()))

rows = [dict(group='retained', **desc(used, 'induction_response')),
        dict(group='excluded', **desc(excl, 'induction_response'))]
T = pd.DataFrame(rows)
print()
print(T.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

F = pd.DataFrame(steps, columns=['step', 'n'])
F.to_csv(f'{ROOT}/results/cohort_flow.csv', index=False)
T.to_csv(f'{ROOT}/results/cohort_flow_excluded.csv', index=False)
print('\nwrote results/cohort_flow.csv and results/cohort_flow_excluded.csv')
