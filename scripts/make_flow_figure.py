"""Figure 1, the specimen flow and the design of the comparison.

Every count is read from the cohort flow table, the specimen table and the saved subset
sizes rather than written here, so the boxes cannot drift from the numbers the Methods
and Table 1 report. The wave counts are checked against the saved sizes as well, so the
two producers cannot disagree.

Each box is drawn around its text after the text has been measured, so no label can
outgrow the box it sits in at the size the figure is finally printed.
"""
import json
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
FS, FS_SMALL, FS_B = 8.2, 7.5, 7.6

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
n_both = int(((spec.cohort == 'Both') & resp.notna()).sum())
sub = json.load(open(f'{ROOT}/results/clinical_cohort.json'))
n_clin = int(sub['clinical_complete_cases'])
assert (sub['wave_train'], sub['wave_test'], sub['wave_both']) == (n_train, n_test, n_both)
assert sub['labelled'] == n_lab

# the saved image is trimmed to its content, so the canvas is drawn larger to land at
# the 5.8 in the other figures use and the text prints at the size set here
fig = plt.figure(figsize=(7.1, 6.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], wspace=0.04)
axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
for ax in (axL, axR):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
REND = fig.canvas.get_renderer()


PLACED, LABELS = [], []


class Node:
    """A text label and the box that is fitted around it once it has been measured."""

    def __init__(self, ax, text, fs=FS, fc='white', ec='#444444', lw=.8,
                 padx=0.030, pady=0.022):
        self.ax, self.fc, self.ec, self.lw = ax, fc, ec, lw
        self.padx, self.pady = padx, pady
        self.t = ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=fs,
                         linespacing=1.4, zorder=4)
        bb = self.t.get_window_extent(REND).transformed(ax.transAxes.inverted())
        self.w, self.h = bb.width + 2 * padx, bb.height + 2 * pady

    def place(self, cx, top):
        """Put the box with its top edge at `top` and its centre on `cx`."""
        cy = top - self.h / 2
        self.t.set_position((cx, cy))
        self.ax.add_patch(FancyBboxPatch((cx - self.w / 2, cy - self.h / 2), self.w, self.h,
                                         boxstyle='round,pad=0,rounding_size=0.018',
                                         linewidth=self.lw, edgecolor=self.ec,
                                         facecolor=self.fc, zorder=2, clip_on=False))
        self.cx, self.top, self.bottom = cx, top, top - self.h
        PLACED.append(self)
        return self


def arrow(ax, x0, y0, x1, y1, ec='#444444'):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=6,
                                 linewidth=.8, color=ec, zorder=1, shrinkA=0, shrinkB=0))


def stack(ax, nodes, cx, top, gap):
    """Place nodes top to bottom, joined by arrows."""
    y = top
    for i, n in enumerate(nodes):
        n.place(cx, y)
        if i:
            arrow(ax, cx, nodes[i - 1].bottom, n.top - 0.002)if False else None
        y = n.bottom - gap
    for a, b in zip(nodes, nodes[1:]):
        arrow(ax, a.cx, a.bottom, b.cx, b.top - 0.002)
    return nodes


for ax, letter in ((axL, 'a'), (axR, 'b')):
    ax.text(0.0, 1.0, f'({letter})', fontsize=10, fontweight='bold', va='top', ha='left')

# ---- panel a: specimen flow -----------------------------------------------------------
CXL = 0.33
flow = [Node(axL, f'n = {n_rec}\nDiagnostic records,\nBeat AML waves 1–4'),
        Node(axL, f'n = {n_named}\nNaming an\nRNA-seq sample'),
        Node(axL, f'n = {n_matrix}\nPresent in the\nexpression matrix'),
        Node(axL, f'n = {n_dose}\nAnalysed specimens\n(Table 1)'),
        Node(axL, f'n = {n_lab}\nBinary induction response\n{n_cr} complete remission,'
                  f'\n{n_ref} refractory', fc='#eef3f7', ec=CA, lw=1.0)]
drops = [(n_rec - n_named, 'no RNA-seq\nsample named'),
         (n_named - n_matrix, 'absent from the\nexpression matrix'),
         (n_matrix - n_dose, 'no inhibitor\nover a complete\ndose series'),
         (n_dose - n_lab, 'no induction-\nresponse label')]
sub_w = [Node(axL, f'Held-out wave split\nwaves 1–2 train, {n_train}\n'
                   f'waves 3–4 test, {n_test}\n−{n_both} in both wave groups', fs=FS_SMALL),
         Node(axL, f'Clinical models\ncovariates and\nELN 2017 risk, {n_clin}', fs=FS_SMALL)]
SUB_GAP = 0.055
reserve = max(n.h for n in sub_w) + SUB_GAP
GAP = (0.93 - sum(n.h for n in flow) - reserve) / 4
stack(axL, flow, CXL, 0.93, GAP)
xe = max(n.w for n in flow) / 2 + CXL + 0.015
for i, (n, why) in enumerate(drops):
    ymid = (flow[i].bottom + flow[i + 1].top) / 2
    arrow(axL, CXL, ymid, xe - 0.004, ymid, ec=CG)
    LABELS.append(axL.text(xe, ymid, f'−{n}  {why}', ha='left', va='center',
                           fontsize=FS_SMALL, color='#555555', linespacing=1.35))
SGX = 0.024
pair = sub_w[0].w + SGX + sub_w[1].w
x0 = max(CXL, pair / 2 + 0.012) - pair / 2        # keep the pair of subsets inside the panel
for n, cx in zip(sub_w, (x0 + sub_w[0].w / 2, x0 + sub_w[0].w + SGX + sub_w[1].w / 2)):
    n.place(cx, flow[-1].bottom - SUB_GAP)
    arrow(axL, CXL, flow[-1].bottom, n.cx, n.top - 0.002)

# ---- panel b: design ------------------------------------------------------------------
CXR = 0.5
refs = Node(axR, 'Three single-cell references\n'
                 'healthy multiome\nAML malignant cells, AML all cells', fs=FS_B)
prog = Node(axR, 'Haematopoietic stem cell programme\nand its eligible gene pool', fs=FS_B)
armA = Node(axR, 'Arm A\ngenes ranked by\nthe accessibility\nof their linked\npeaks',
            fc='#eaf1f6', ec=CA, lw=1.0, fs=FS_B)
armR = Node(axR, 'Arm R\ngenes ranked by\ntheir own\nexpression\nspecificity',
            fc='#f7edea', ec=CR_, lw=1.0, fs=FS_B)
model = Node(axR, 'Each set scored in the bulk RNA of\n'
                  'the same patients, identical folds,\n'
                  'outcome induction response', fs=FS_B)
evals = Node(axR, 'Evaluation: repeated cross-validation\n'
                  'and a held-out wave split\n'
                  'Comparisons: the two rules at four\n'
                  'penalties, a published stem cell gene\n'
                  'set, the clinical panel and ELN 2017\n'
                  'risk, control sets and permuted links', fs=FS_B)
rows = [refs.h, prog.h, max(armA.h, armR.h), model.h, evals.h]
GAPR = (0.93 - sum(rows) - 0.035) / 4                 # 0.035 for the italic note
refs.place(CXR, 0.93)
prog.place(CXR, refs.bottom - GAPR)
arrow(axR, CXR, refs.bottom, CXR, prog.top - 0.002)
GX = 0.02
left = CXR - (armA.w + GX + armR.w) / 2
armA.place(left + armA.w / 2, prog.bottom - GAPR)
armR.place(left + armA.w + GX + armR.w / 2, prog.bottom - GAPR)
arrow(axR, CXR - 0.06, prog.bottom, armA.cx, armA.top - 0.002, ec=CA)
arrow(axR, CXR + 0.06, prog.bottom, armR.cx, armR.top - 0.002, ec=CR_)
arms_bottom = min(armA.bottom, armR.bottom)
model.place(CXR, arms_bottom - 0.035 - GAPR)
LABELS.append(axR.text(CXR, (arms_bottom + model.top) / 2,
                       'same programme, same pool, same budget', ha='center', va='center',
                       fontsize=FS_SMALL, style='italic', color='#555555', zorder=5,
                       bbox=dict(facecolor='white', edgecolor='none', pad=1.0)))
arrow(axR, armA.cx, armA.bottom, CXR - 0.06, model.top - 0.002, ec=CA)
arrow(axR, armR.cx, armR.bottom, CXR + 0.06, model.top - 0.002, ec=CR_)
evals.place(CXR, model.bottom - GAPR)
arrow(axR, CXR, model.bottom, CXR, evals.top - 0.002)

fig.canvas.draw()
out = []
for nd in PLACED:
    x0, x1 = nd.cx - nd.w / 2, nd.cx + nd.w / 2
    if x0 < -0.005 or x1 > 1.005 or nd.bottom < -0.005 or nd.top > 1.005:
        out.append(f'box [{x0:.3f},{x1:.3f}]x[{nd.bottom:.3f},{nd.top:.3f}] '
                   f'{nd.t.get_text().splitlines()[0]!r}')
for lb in LABELS:
    bb = lb.get_window_extent(REND).transformed(lb.axes.transAxes.inverted())
    if bb.x0 < -0.005 or bb.x1 > 1.005:
        out.append(f'label [{bb.x0:.3f},{bb.x1:.3f}] {lb.get_text().splitlines()[0]!r}')
if out:
    print('CONTENT OUTSIDE ITS PANEL:')
    for o in out:
        print('  ', o)
    raise SystemExit(1)

fig.savefig(f'{OUT}/Figure1.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'wrote {OUT}/Figure1.png | {n_rec} {n_named} {n_matrix} {n_dose} {n_lab} '
      f'({n_cr} CR, {n_ref} refractory) | waves {n_train}/{n_test} (−{n_both}) | '
      f'clinical {n_clin} | '
      f'widest box {max(n.w for n in flow):.2f} of panel a, {refs.w:.2f} of panel b')
