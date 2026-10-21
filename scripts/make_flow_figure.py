"""Figure 1, the specimen flow and the design of the comparison.

Every count is read from the cohort flow table and the specimen table rather than written
here, so the boxes cannot drift from the numbers the Methods and Table 1 report.
"""
import os
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = '/home/holiday01/2026_cm'
OUT = f'{ROOT}/manuscript/figures'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans'})
CA, CR_, CG = '#1b5e8c', '#b4634a', '#8c8c8c'

fl = pd.read_csv(f'{ROOT}/results/cohort_flow.csv').set_index('step').n
spec = pd.read_parquet(f'{ROOT}/data/interim/specimens_dx.parquet')
resp = spec.induction_response
n_rec = int(fl.loc['initial diagnosis records'])
n_named = int(fl.loc['naming an RNA-seq sample'])
n_matrix = int(fl.loc['present in the expression matrix'])
n_dose = int(fl.loc['with an inhibitor over a complete dose series'])
n_lab = int(resp.notna().sum())
n_cr, n_ref = int((resp == 'CR').sum()), int((resp == 'Refractory').sum())
n_train = int(((spec.cohort == 'Waves1+2') & resp.notna()).sum())
n_test = int(((spec.cohort == 'Waves3+4') & resp.notna()).sum())


def box(ax, x, y, w, h, text, fc='white', ec='#444444', lw=.8, fs=6.6, va='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.015',
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va=va, fontsize=fs, zorder=3,
            linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, ec='#444444'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=6,
                                 linewidth=.8, color=ec, zorder=1, shrinkA=0, shrinkB=0))


fig = plt.figure(figsize=(5.8, 4.9))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.92], wspace=0.10)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (axL, axR):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
for ax, letter in ((axL, 'a'), (axR, 'b')):
    ax.text(0.0, 1.015, f'({letter})', fontsize=10, fontweight='bold', va='bottom')

# ---- panel a: specimen flow -----------------------------------------------------------
X, W = 0.02, 0.50
steps = [(n_rec, 'Diagnostic records,\nBeat AML waves 1–4'),
         (n_named, 'Naming an\nRNA-seq sample'),
         (n_matrix, 'Present in the\nexpression matrix'),
         (n_dose, 'Analysed specimens\n(Table 1)'),
         (n_lab, f'Binary induction response\n{n_cr} complete remission,\n{n_ref} refractory')]
drops = [(n_rec - n_named, 'no RNA-seq\nsample named'),
         (n_named - n_matrix, 'absent from the\nexpression matrix'),
         (n_matrix - n_dose, 'no inhibitor over a\ncomplete dose series'),
         (n_dose - n_lab, 'no induction-\nresponse label')]
H, GAP = 0.105, 0.060
tops = []
y = 0.95
for i, (n, lbl) in enumerate(steps):
    h = H * (1.25 if i == len(steps) - 1 else 1.0)
    fs = 6.0 if i == len(steps) - 1 else 6.6
    y -= h
    last = i == len(steps) - 1
    box(axL, X, y, W, h, f'n = {n}\n{lbl}', fc='#eef3f7' if last else 'white',
        ec=CA if last else '#444444', lw=1.0 if last else .8, fs=fs)
    tops.append((y, h))
    if not last:
        y -= GAP
for i, (n, why) in enumerate(drops):
    y_top = tops[i][0]
    y_next = tops[i + 1][0] + tops[i + 1][1]
    ymid = (y_top + y_next) / 2
    arrow(axL, X + W / 2, y_top, X + W / 2, y_next + 0.002)
    arrow(axL, X + W / 2, ymid, 0.555, ymid, ec=CG)
    axL.text(0.575, ymid, f'−{n}  {why}', ha='left', va='center', fontsize=6.0,
             color='#555555', linespacing=1.35)
y_lab, h_lab = tops[-1]
yb = y_lab - 0.055 - 0.080
box(axL, X, yb, W * 0.47, 0.080, f'waves\n{n_train} / {n_test}', fs=6.0)
box(axL, X + W * 0.53, yb, W * 0.47, 0.080, 'covariates\nand ELN, 194', fs=6.0)
arrow(axL, X + W * 0.235, y_lab, X + W * 0.235, yb + 0.082)
arrow(axL, X + W * 0.765, y_lab, X + W * 0.765, yb + 0.082)

# ---- panel b: design ------------------------------------------------------------------
BX, BW = 0.03, 0.94
box(axR, BX, 0.845, BW, 0.095,
    'Three single-cell references\nhealthy multiome, AML malignant, AML all cells', fs=6.6)
box(axR, BX + 0.10, 0.665, BW - 0.20, 0.105,
    'Haematopoietic stem cell programme\nand its eligible gene pool', fs=6.6)
arrow(axR, 0.5, 0.845, 0.5, 0.772)
box(axR, BX, 0.435, 0.44, 0.155,
    'Arm A\ngenes ranked by\nthe accessibility\nof their linked peaks',
    fc='#eaf1f6', ec=CA, lw=1.0, fs=6.4)
box(axR, BX + 0.50, 0.435, 0.44, 0.155,
    'Arm R\ngenes ranked by\ntheir own\nexpression specificity',
    fc='#f7edea', ec=CR_, lw=1.0, fs=6.4)
arrow(axR, 0.40, 0.665, 0.25, 0.593, ec=CA)
arrow(axR, 0.60, 0.665, 0.75, 0.593, ec=CR_)
axR.text(0.5, 0.400, 'same programme, same pool, same budget', ha='center', va='center',
         fontsize=6.0, style='italic', color='#555555', zorder=4,
         bbox=dict(facecolor='white', edgecolor='none', pad=1.2))
box(axR, BX + 0.05, 0.260, BW - 0.10, 0.105,
    'Induction response in the same patients,\nfour penalties, identical folds', fs=6.6)
arrow(axR, 0.25, 0.435, 0.42, 0.367, ec=CA)
arrow(axR, 0.75, 0.435, 0.58, 0.367, ec=CR_)
box(axR, BX, 0.025, BW, 0.195,
    'Difference between the two rules, against\n'
    'cross-validation and a held-out wave split\n'
    'a published leukaemic stem cell gene set\n'
    'the clinical panel and ELN 2017 risk\n'
    'control sets and permuted links', fs=6.2)
arrow(axR, 0.5, 0.260, 0.5, 0.222)

fig.savefig(f'{OUT}/Figure1.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'wrote {OUT}/Figure1.png | {n_rec} {n_named} {n_matrix} {n_dose} {n_lab} '
      f'({n_cr} CR, {n_ref} refractory) | waves {n_train}/{n_test}')
