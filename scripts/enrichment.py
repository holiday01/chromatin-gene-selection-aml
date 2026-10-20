"""TF-network and pathway enrichment of the HSC arm A / arm R / arm-A-unique gene sets.

Two parts:
  local  Fisher exact overlap with the published LSC gene sets (Ng et al. 2016), tested
         against the eligible gene pool, not the genome.
  enrichr  TF regulons (TRRUST, ChEA) and pathways (Hallmark, GO BP) via speedrichr, again
         with the ELIGIBLE POOL as background. Using the default whole-genome background
         would inflate significance, because the pool is already restricted to expressed,
         peak-linked genes -- the comparison we care about is arm A against that pool.

Only gene symbols from a public reference are sent to the Enrichr service.
"""
import os, sys, json, time, urllib.request, urllib.parse
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
ROOT = '/home/holiday01/2026_cm'
BASE = 'https://maayanlab.cloud/speedrichr/api'
LIBS = ['TRRUST_Transcription_Factors_2019', 'ChEA_2022',
        'MSigDB_Hallmark_2020', 'GO_Biological_Process_2023']

pr = np.load(f'{ROOT}/data/programs/programs.npz', allow_pickle=True)
pgenes = pr['genes'].astype(str); cts = list(pr['cell_types'].astype(str))
jH = cts.index('HSC'); elig = pr['eligible']
A = set(pgenes[np.flatnonzero(pr['W_A'][:, jH])])
R = set(pgenes[np.flatnonzero(pr['W_R'][:, jH])])
POOL = set(pgenes[elig])
SETS = {'armA_HSC': sorted(A), 'armR_HSC': sorted(R),
        'armA_only': sorted(A - R), 'armR_only': sorted(R - A)}
print({k: len(v) for k, v in SETS.items()}, f'| pool {len(POOL)}')

# ---------- local: overlap with the published LSC sets --------------------------------
res_local = []
for name, f in (('Ng2016_LSC104', 'ng2016_lsc104_resolved.txt'), ('Ng2016_LSC17', 'ng2016_lsc17.txt')):
    ref = {l.strip() for l in open(f'{ROOT}/data/ref/published_signatures/{f}') if l.strip()}
    refp = ref & POOL
    for sname, s in SETS.items():
        s = set(s); ov = len(s & refp)
        tab = [[ov, len(s) - ov], [len(refp) - ov, len(POOL) - len(s) - len(refp) + ov]]
        odds, p = fisher_exact(tab, alternative='greater')
        exp = len(s) * len(refp) / len(POOL)
        res_local.append(dict(reference=name, n_ref_in_pool=len(refp), gene_set=sname,
                              n=len(s), overlap=ov, expected=round(exp, 2),
                              odds=round(odds, 2) if np.isfinite(odds) else np.inf, p=p,
                              genes=','.join(sorted(s & refp))))
L = pd.DataFrame(res_local)
print('\n=== overlap with the published LSC gene sets (background = eligible pool) ===')
print(L[['reference','n_ref_in_pool','gene_set','n','overlap','expected','odds','p']]
      .to_string(index=False, float_format=lambda v: f'{v:.3g}'))
print('\nshared genes:')
for _, r in L[L.overlap > 0].iterrows():
    print(f'  {r.reference:14s} {r.gene_set:10s} {r.genes}')
L.to_csv(f'{ROOT}/results/enrich_lsc_overlap.csv', index=False)

# ---------- enrichr with a custom background ------------------------------------------
def post(path, fields, files=None):
    import uuid
    b = uuid.uuid4().hex
    body = b''
    for k, v in fields.items():
        body += (f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n').encode()
    body += f'--{b}--\r\n'.encode()
    req = urllib.request.Request(f'{BASE}/{path}', data=body,
                                 headers={'Content-Type': f'multipart/form-data; boundary={b}'})
    return json.load(urllib.request.urlopen(req, timeout=180))

print('\nuploading background ...', flush=True)
bg = post('addbackground', {'background': '\n'.join(sorted(POOL))})
bgid = bg['backgroundid']
rows = []
for sname, genes in SETS.items():
    up = post('addList', {'list': '\n'.join(genes), 'description': sname})
    time.sleep(1)
    for lib in LIBS:
        try:
            req = urllib.request.Request(f'{BASE}/backgroundenrich',
                data=urllib.parse.urlencode({'userListId': up['userListId'],
                    'backgroundid': bgid, 'backgroundType': lib}).encode())
            out = json.load(urllib.request.urlopen(req, timeout=240))
        except Exception as e:
            print(f'  {sname}/{lib}: FAILED {type(e).__name__}: {str(e)[:70]}'); continue
        for t in out.get(lib, [])[:400]:
            # rank, term, pval, zscore, combined, overlapping genes, qval(BH), ...
            rows.append(dict(gene_set=sname, library=lib, term=t[1], p=t[2],
                             odds=t[3], combined=t[4], q=t[6], n_overlap=len(t[5]),
                             genes=','.join(t[5])))
        time.sleep(1)
E = pd.DataFrame(rows)
if len(E):
    E.to_csv(f'{ROOT}/results/enrich_enrichr.csv', index=False)
    print(f'\n=== Enrichr, FDR < 0.05, background = {len(POOL)} eligible genes ===')
    sig = E[E.q < 0.05].sort_values(['gene_set','library','q'])
    if len(sig) == 0:
        print('  nothing passes FDR 0.05 in any set/library')
    for (s, lib), g in sig.groupby(['gene_set','library']):
        print(f'\n-- {s} / {lib}  ({len(g)} terms at FDR<0.05)')
        print(g.head(8)[['term','n_overlap','odds','p','q']]
              .to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    print('\n=== top 5 by p per set/library regardless of FDR ===')
    for (s, lib), g in E.sort_values('p').groupby(['gene_set','library']):
        top = g.head(5)
        print(f'-- {s} / {lib}')
        print(top[['term','n_overlap','p','q']].to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    print('\nwrote results/enrich_enrichr.csv')
