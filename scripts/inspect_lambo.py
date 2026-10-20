"""Inspect the Lambo AML reference before building anything from it.
Questions that decide the build: do samples share a peak set, do the linked scRNA
barcodes resolve to the companion series, and how many malignant HSC-like cells are there.
"""
import sys, gzip, glob, os, re
import pandas as pd, numpy as np
ATAC, RNA = sys.argv[1], sys.argv[2]

am = sorted(glob.glob(f'{ATAC}/*_filtered_metadata.tsv.gz'))
pk = sorted(glob.glob(f'{ATAC}/*_peaks.tsv.gz'))
rm = sorted(glob.glob(f'{RNA}/*_processed_metadata.tsv.gz'))
print(f'ATAC metadata files {len(am)} | peak files {len(pk)} | RNA metadata files {len(rm)}')

# peak-set consistency, by file size then by content hash on a sample
sizes = pd.Series({os.path.basename(f): os.path.getsize(f) for f in pk})
print(f'distinct peak-file sizes: {sizes.nunique()} of {len(sizes)}')
import hashlib
h = {}
for f in pk[:12]:
    with gzip.open(f, 'rt') as fh:
        first = fh.readline()
        n = sum(1 for _ in fh) + 1
    h[os.path.basename(f)] = (n, first.strip()[:60])
for k, v in list(h.items())[:6]:
    print(f'  {k:52s} rows {v[0]:7d}  first: {v[1]}')

M = []
for f in am:
    d = pd.read_csv(f, sep='\t', usecols=lambda c: c in (
        'Cell_Barcode','Patient_Sample','Library_ID','Classified_Celltype','Malignant',
        'Linked_scRNA_barcode','Linked_scRNA_Sample','Linked_scRNA_confidence_score',
        'Patient_ID','Subgroup','Treatment_Outcome','Biopsy_Origin','Age_Months'))
    M.append(d)
M = pd.concat(M, ignore_index=True)
print(f'\nATAC cells total {len(M):,} | libraries {M.Library_ID.nunique()} | '
      f'patients {M.Patient_ID.nunique()}')
print('Patient_Sample:', M.Patient_Sample.value_counts().to_dict())
print('Malignant     :', M.Malignant.value_counts(dropna=False).to_dict())
print(f'linked scRNA barcode present: {M.Linked_scRNA_barcode.notna().sum():,} '
      f'({M.Linked_scRNA_barcode.notna().mean()*100:.1f}%)')
print('link confidence:', M.Linked_scRNA_confidence_score.describe()[['mean','50%','min','max']].round(3).to_dict())
print('\nClassified_Celltype (top 18):')
print(M.Classified_Celltype.value_counts().head(18).to_string())
print('\nmalignant x celltype, HSC-like rows:')
sub = M[M.Classified_Celltype.astype(str).str.contains('HSC|GMP|prog|MPP|LMPP', case=False, na=False)]
print(sub.groupby(['Classified_Celltype','Malignant']).size().unstack(fill_value=0).to_string())
print('\nTreatment_Outcome:', M.drop_duplicates('Patient_ID').Treatment_Outcome.value_counts(dropna=False).to_dict())
print('Subgroup       :', M.drop_duplicates('Patient_ID').Subgroup.value_counts(dropna=False).to_dict())

R = []
for f in rm[:200]:
    d = pd.read_csv(f, sep='\t', nrows=5)
    R.append((os.path.basename(f), d.shape[1], list(d.columns)[:8]))
print(f'\nRNA metadata columns (first file): {R[0][2]}')
rs = set()
for f in rm:
    rs.add(re.sub(r'^GSM\d+_|_processed_metadata\.tsv\.gz$', '', os.path.basename(f)))
asamp = set(M.Linked_scRNA_Sample.dropna().astype(str))
print(f'RNA sample tokens {len(rs)} | linked-sample tokens in ATAC {len(asamp)} | '
      f'intersection {len(rs & asamp)}')
print('  ATAC-linked tokens missing an RNA file:', sorted(asamp - rs)[:8])
