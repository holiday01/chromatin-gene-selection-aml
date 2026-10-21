"""The sizes of the two subsets of the labelled cohort, so that the figure and the
Methods read them from a file rather than carrying them as constants.

The clinical subset reproduces the inclusion mask of the clinical models in
recompute_remaining.py (the labelled specimens with every covariate recorded and an
unambiguous ELN 2017 category); nothing is fitted here, only counted. The wave split is
read from the collection-wave annotation of the specimen table.
"""
import json
import numpy as np, pandas as pd

ROOT = '/home/holiday01/2026_cm'
ELN3 = ['Favorable', 'Intermediate', 'Adverse']

expr = pd.read_parquet(f'{ROOT}/data/interim/expr_norm_346.parquet')
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
expr = expr.drop_duplicates('display_label').set_index('display_label')
cols = [c for c in expr.columns if c in set(spec.dbgap_rnaseq_sample)]
S = spec.set_index('dbgap_rnaseq_sample'); lab = S.induction_response.to_dict()
yall = np.array([{'CR': 0, 'Refractory': 1}.get(lab.get(s), -1) for s in cols])

cl = pd.read_excel(f'{ROOT}/data/raw/beataml2/beataml_wv1to4_clinical.xlsx')
cl = cl.drop_duplicates('dbgap_rnaseq_sample').set_index('dbgap_rnaseq_sample')
num = lambda s: pd.to_numeric(s, errors='coerce')
C = pd.DataFrame(index=cols)
C['age'] = num(cl.reindex(cols).ageAtDiagnosis)
C['male'] = (cl.reindex(cols).consensus_sex == 'Male').astype(float)
C['blast'] = num(cl.reindex(cols)['%.Blasts.in.BM'])
C['wbc'] = np.log1p(num(cl.reindex(cols).wbcCount))
C['flt3'] = (cl.reindex(cols)['FLT3-ITD'].astype(str).str.lower() == 'positive').astype(float)
C['npm1'] = (cl.reindex(cols)['NPM1'].astype(str).str.lower() == 'positive').astype(float)
eln = cl.reindex(cols).ELN2017
ok = (yall >= 0) & C.notna().all(1).to_numpy() & eln.isin(ELN3).to_numpy()
yo = yall[ok]; elno = eln[ok.tolist()]

resp = spec.induction_response
out = dict(
    labelled=int(resp.notna().sum()),
    clinical_complete_cases=int(ok.sum()),
    clinical_cr=int((yo == 0).sum()),
    clinical_refractory=int((yo == 1).sum()),
    clinical_eln_favourable=int((elno == 'Favorable').sum()),
    clinical_eln_intermediate=int((elno == 'Intermediate').sum()),
    clinical_eln_adverse=int((elno == 'Adverse').sum()),
    wave_train=int(((spec.cohort == 'Waves1+2') & resp.notna()).sum()),
    wave_train_cr=int(((spec.cohort == 'Waves1+2') & (resp == 'CR')).sum()),
    wave_train_refractory=int(((spec.cohort == 'Waves1+2') & (resp == 'Refractory')).sum()),
    wave_test=int(((spec.cohort == 'Waves3+4') & resp.notna()).sum()),
    wave_test_cr=int(((spec.cohort == 'Waves3+4') & (resp == 'CR')).sum()),
    wave_test_refractory=int(((spec.cohort == 'Waves3+4') & (resp == 'Refractory')).sum()),
    wave_both=int(((spec.cohort == 'Both') & resp.notna()).sum()),
)
assert out['wave_train'] + out['wave_test'] + out['wave_both'] == out['labelled']
json.dump(out, open(f'{ROOT}/results/clinical_cohort.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
print('wrote results/clinical_cohort.json')
