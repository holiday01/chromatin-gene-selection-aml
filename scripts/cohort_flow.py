"""The inclusion steps from the Beat AML clinical table to the 346 analysed specimens, and
how the drug-assay criterion changed the composition of the cohort.

The criterion is inherited from the wider project and does not serve the induction-response
endpoint, so the cohort is narrower than the set of diagnostic specimens with expression
data. This script compares the 104 specimens it removed with the 346 retained on everything
the clinical table records that enters the analysis, so the effect on the study population
can be read off rather than asserted. Marginal similarity on one variable says nothing about
the others, so the table reports each variable separately, with its own denominator, and
continuous variables with quartiles rather than a median alone.

The induction regimen field is summarised here as well. The treatment-type field records a
single category for almost every labelled patient, which is not evidence that they received
the same regimen, and the regimen field shows that they did not.
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
inmat['response'] = inmat.responseToInductionTx.map(norm_resp)
inmat['group'] = np.where(inmat.dbgap_rnaseq_sample.isin(set(used.dbgap_rnaseq_sample)),
                          'retained', 'excluded')

steps = [('initial diagnosis records', len(dx)),
         ('naming an RNA-seq sample', len(named)),
         ('present in the expression matrix', len(inmat)),
         ('with an inhibitor over a complete dose series', len(used))]
print('inclusion flow')
for lbl, n in steps: print(f'  {lbl:46s} {n:5d}')
print(f'  the {len(inmat)} records come from {inmat.dbgap_subject_id.nunique()} patients')

lab = inmat[inmat.response.notna()]
G = ['excluded', 'retained']
rows = []

def add(variable, denominator, counts, totals):
    r = dict(variable=variable, denominator=denominator)
    for g in G:
        r[g] = f'{counts[g]}/{totals[g]} ({counts[g] / totals[g]:.1%})' if totals[g] else 'NA'
    rows.append(r)

def add_quart(variable, denominator, frame, col):
    r = dict(variable=variable, denominator=denominator)
    for g in G:
        v = pd.to_numeric(frame.loc[frame.group == g, col], errors='coerce').dropna()
        r[g] = f'{v.median():.1f} ({v.quantile(.25):.1f} to {v.quantile(.75):.1f})'
    rows.append(r)

n_all = {g: int((inmat.group == g).sum()) for g in G}
n_lab = {g: int((lab.group == g).sum()) for g in G}
rows.append(dict(variable='specimens', denominator='all specimens',
                 **{g: str(n_all[g]) for g in G}))
add('with an induction response label', 'all specimens', n_lab, n_all)
add('refractory', 'labelled specimens',
    {g: int(((lab.group == g) & (lab.response == 'Refractory')).sum()) for g in G}, n_lab)
add_quart('age at diagnosis, years, median (IQR)', 'all specimens', inmat, 'ageAtDiagnosis')
add_quart('bone-marrow blasts, per cent, median (IQR)', 'all specimens', inmat,
          '%.Blasts.in.BM')
add('female', 'all specimens',
    {g: int(((inmat.group == g) & (inmat.consensus_sex == 'Female')).sum()) for g in G}, n_all)
add('bone-marrow aspirate', 'all specimens',
    {g: int(((inmat.group == g) & (inmat.specimenType == 'Bone Marrow Aspirate')).sum())
     for g in G}, n_all)
for cat in ['Favorable', 'Intermediate', 'Adverse']:
    add(f'ELN 2017 {cat.lower()}', 'labelled specimens',
        {g: int(((lab.group == g) & (lab.ELN2017 == cat)).sum()) for g in G}, n_lab)
add('ELN 2017 ambiguous or missing karyotype', 'labelled specimens',
    {g: int(((lab.group == g) & ~lab.ELN2017.isin(['Favorable', 'Intermediate', 'Adverse'])).sum())
     for g in G}, n_lab)
for cat, lbl in [('Waves1+2', 'collection waves 1 and 2'),
                 ('Waves3+4', 'collection waves 3 and 4'),
                 ('Both', 'annotated to both wave groups')]:
    add(lbl, 'labelled specimens',
        {g: int(((lab.group == g) & (lab.cohort == cat)).sum()) for g in G}, n_lab)
add('induction recorded as standard chemotherapy', 'labelled specimens',
    {g: int(((lab.group == g) & (lab.typeInductionTx == 'Standard Chemotherapy')).sum())
     for g in G}, n_lab)

T = pd.DataFrame(rows)[['variable', 'denominator'] + G]
print()
print(T.to_string(index=False))

# the recorded regimens of the retained, labelled patients
ret = lab[lab.group == 'retained']
first = ret.cumulativeTreatmentRegimens.str.split('|').str[0]
seven = first.str.contains(r'7\+3|3\+7', na=False)
print(f'\nrecorded regimens of the {len(ret)} retained labelled patients')
print(f'  distinct first entries                 {first.nunique():4d}')
print(f'  first entry is 7+3 or 3+7              {int(seven.sum()):4d} ({seven.mean():.1%})')
print(f'  induction type Standard Chemotherapy   '
      f"{int((ret.typeInductionTx == 'Standard Chemotherapy').sum()):4d}")
reg = pd.DataFrame([dict(item='distinct first regimen entries', value=int(first.nunique())),
                    dict(item='first entry is 7+3 or 3+7', value=int(seven.sum())),
                    dict(item='labelled retained patients', value=len(ret))])

F = pd.DataFrame(steps, columns=['step', 'n'])
F.to_csv(f'{ROOT}/results/cohort_flow.csv', index=False)
T.to_csv(f'{ROOT}/results/cohort_flow_excluded.csv', index=False)
reg.to_csv(f'{ROOT}/results/cohort_regimens.csv', index=False)
print('\nwrote results/cohort_flow.csv, cohort_flow_excluded.csv and cohort_regimens.csv')
