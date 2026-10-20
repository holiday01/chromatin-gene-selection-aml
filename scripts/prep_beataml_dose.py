"""Build CLIFF-ready dose-level viability tensors from Beat AML, restricted to the
Initial Diagnosis specimens that also have bulk RNA-seq.

Outputs to data/interim/:
  specimens_dx.parquet   one row per analysed specimen with covariates
  dose_grid.parquet      per drug: the 7-point concentration grid
  viability.parquet      long table (specimen, drug, dose_rank, concentration, viability01)
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

ROOT = '/home/holiday01/2026_cm'
RAW  = os.path.join(ROOT, 'data/raw/beataml2')
OUT  = os.path.join(ROOT, 'data/interim')

# --- specimens: expression ∩ AUC ∩ Initial Diagnosis ---
hdr = open(os.path.join(RAW, 'beataml_waves1to4_norm_exp_dbgap.txt')).readline().rstrip('\n').split('\t')
expr_samples = set(hdr[4:])
cl = pd.read_excel(os.path.join(RAW, 'beataml_wv1to4_clinical.xlsx'))
keep = ['dbgap_rnaseq_sample','dbgap_subject_id','diseaseStageAtSpecimenCollection','specimenType',
        'cohort','responseToInductionTx','vitalStatus','overallSurvival','ageAtDiagnosis','consensus_sex']
cl = cl[keep].dropna(subset=['dbgap_rnaseq_sample'])
cl = cl[cl.dbgap_rnaseq_sample.isin(expr_samples)]
dx = cl[cl.diseaseStageAtSpecimenCollection == 'Initial Diagnosis'].copy()

# normalise the induction-response field
def norm_resp(x):
    if not isinstance(x, str): return np.nan
    s = x.strip().lower()
    if s.startswith('complete response'): return 'CR'
    if s.startswith('refractory'): return 'Refractory'
    return np.nan
dx['induction_response'] = dx.responseToInductionTx.map(norm_resp)

# --- dose-level viability ---
rv = pd.read_csv(os.path.join(RAW, 'beataml_wv1to4_raw_inhibitor_v4_dbgap.txt'), sep='\t',
                 usecols=['dbgap_rnaseq_sample','inhibitor','well_concentration',
                          'normalized_viability','replicate','passed_qc','inhibitor_panel'])
rv = rv[(rv.passed_qc == True) & rv.dbgap_rnaseq_sample.isin(set(dx.dbgap_rnaseq_sample))]
# average replicates, convert percent -> probability, clip as CLIFF requires
rv = (rv.groupby(['dbgap_rnaseq_sample','inhibitor','well_concentration'], as_index=False)
        .normalized_viability.mean())
rv['viability01'] = np.clip(rv.normalized_viability / 100.0, 0.0, 1.0)
rv['dose_rank'] = rv.groupby(['dbgap_rnaseq_sample','inhibitor']).well_concentration.rank(method='dense').astype(int)

# keep only (drug, specimen) pairs with the full 7-point grid
n_dose = rv.groupby(['inhibitor','dbgap_rnaseq_sample']).dose_rank.max()
full = n_dose[n_dose == 7].reset_index()[['inhibitor','dbgap_rnaseq_sample']]
rv = rv.merge(full, on=['inhibitor','dbgap_rnaseq_sample'])

grid = (rv.groupby(['inhibitor','dose_rank'], as_index=False)
          .well_concentration.median().rename(columns={'well_concentration':'concentration_med'}))

dx_used = dx[dx.dbgap_rnaseq_sample.isin(set(rv.dbgap_rnaseq_sample))].copy()

os.makedirs(OUT, exist_ok=True)
dx_used.to_parquet(os.path.join(OUT, 'specimens_dx.parquet'), index=False)
grid.to_parquet(os.path.join(OUT, 'dose_grid.parquet'), index=False)
rv[['dbgap_rnaseq_sample','inhibitor','dose_rank','well_concentration','viability01']].to_parquet(
    os.path.join(OUT, 'viability.parquet'), index=False)

cov = rv.groupby('inhibitor').dbgap_rnaseq_sample.nunique()
print(f'specimens analysed      : {dx_used.dbgap_rnaseq_sample.nunique()}  (patients {dx_used.dbgap_subject_id.nunique()})')
print(f'drugs with 7-point grid : {rv.inhibitor.nunique()}')
print(f'rows (long viability)   : {len(rv):,}')
print(f'drugs >=100 specimens   : {(cov>=100).sum()}   >=200: {(cov>=200).sum()}   >=300: {(cov>=300).sum()}')
print(f'cohort split            : {dx_used.cohort.value_counts().to_dict()}')
print(f'induction response      : {dx_used.induction_response.value_counts(dropna=False).to_dict()}')
print(f'viability01 quantiles   : {np.round(np.quantile(rv.viability01, [0,.25,.5,.75,1]),3).tolist()}')
print(f'clipped at 1.0          : {(rv.viability01 >= 1.0).mean()*100:.1f}% of measurements')
print('wrote specimens_dx.parquet, dose_grid.parquet, viability.parquet')
