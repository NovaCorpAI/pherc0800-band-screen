"""sense_m7.py M7_ZARR LEVEL UMBILICUS.json Z[,Z...] [--label NAME] [--rmin 300] [--rmax 0] [--steps 360]

Spiral sense from the m7 surface prediction: on a z slice, sample rays from the scroll axis at --steps angles, find the
predicted-sheet crossings (run centres of m7 >= 128) along each ray, follow every sheet one full revolution in the
direction of increasing theta (theta = atan2(y - cy, x - cx), y = row, x = column) by nearest-crossing matching, and see on
which crossing it lands back at the start angle: +1 = one sheet further out, -1 = one sheet further in, 0 = same (lost /
closed ring), else = jumped. Ellipticity cancels over the full turn, so the vote only needs the sheets to be followable.
M7_ZARR is a local zarr directory (levels 0..5 as groups) or s3://bucket/path (anonymous). Coordinates in level-0 voxels.
"""
import argparse, json, sys
import numpy as np, zarr
from scipy import ndimage

ap = argparse.ArgumentParser()
ap.add_argument('zarr'); ap.add_argument('level', type=int); ap.add_argument('umb'); ap.add_argument('zs')
ap.add_argument('--label', default=''); ap.add_argument('--rmin', type=float, default=300); ap.add_argument('--rmax', type=float, default=0)
ap.add_argument('--steps', type=int, default=3600); ap.add_argument('--rstep', type=float, default=0.5)
ap.add_argument('--coast', type=int, default=20, help='max consecutive angles without a crossing within the jump tolerance')
a = ap.parse_args()
ds = 2 ** a.level
if a.zarr.startswith('s3://'):
    from zarr.storage import FsspecStore
    store = FsspecStore.from_url(a.zarr, storage_options={'anon': True}); g = zarr.open_group(store, mode='r')
else:
    g = zarr.open_group(a.zarr, mode='r')
arr = g[str(a.level)]
cp = sorted(json.load(open(a.umb))['control_points'], key=lambda q: q['z'])
uz = np.array([q['z'] for q in cp], float); ux = np.array([q['x'] for q in cp], float); uy = np.array([q['y'] for q in cp], float)
Z, Y, X = arr.shape
votes = {}
for z0 in [float(v) for v in a.zs.split(',')]:
    cx = float(np.interp(z0, uz, ux)) / ds; cy = float(np.interp(z0, uz, uy)) / ds
    sl = np.asarray(arr[int(round(z0 / ds))]).astype(np.float32) / 255.0
    rmax = (a.rmax / ds) if a.rmax else min(cx, cy, X - cx, Y - cy) - 2
    rmin = a.rmin / ds
    rs = np.arange(rmin, rmax, a.rstep); th = np.linspace(0, 2 * np.pi, a.steps, endpoint=False)
    yy = cy + np.outer(np.sin(th), rs); xx = cx + np.outer(np.cos(th), rs)
    prof = ndimage.map_coordinates(sl, [yy.ravel(), xx.ravel()], order=1, mode='constant', cval=0).reshape(len(th), len(rs))
    on = prof >= 0.5
    cross = []   # per angle: sorted crossing radii (run centres)
    for j in range(len(th)):
        d = np.diff(np.concatenate([[0], on[j].astype(np.int8), [0]]))
        s = np.nonzero(d == 1)[0]; e = np.nonzero(d == -1)[0]
        cross.append((rs[s] + rs[e - 1]) / 2.0)
    c0 = cross[0]
    if len(c0) < 3:
        votes[z0] = 'no sheets'; continue
    gap = float(np.median(np.diff(c0))); jump = 0.35 * gap
    tally = {'+1': 0, '-1': 0, '0': 0, 'other': 0, 'lost': 0}
    for k, r0 in enumerate(c0):
        r = r0; ok = True; miss = 0
        for j in list(range(1, len(th))) + [0]:
            c = cross[j]
            i = int(np.argmin(np.abs(c - r))) if len(c) else -1
            if i < 0 or abs(c[i] - r) > jump:
                miss += 1
                if miss > a.coast: ok = False; break
                continue
            miss = 0; r = c[i]
        if not ok:
            tally['lost'] += 1; continue
        kk = int(np.argmin(np.abs(c0 - r))); d = kk - k
        tally['+1' if d == 1 else '-1' if d == -1 else '0' if d == 0 else 'other'] += 1
    tally['gap_vox'] = round(gap * ds, 1); tally['sheets'] = len(c0)
    votes[z0] = tally
plus = sum(v['+1'] for v in votes.values() if isinstance(v, dict)); minus = sum(v['-1'] for v in votes.values() if isinstance(v, dict))
print(json.dumps({'label': a.label, 'level': a.level, 'votes': votes, 'plus': plus, 'minus': minus,
                  'outward_with_increasing_theta': '+' if plus > minus else '-' if minus > plus else '?',
                  'margin': round(abs(plus - minus) / max(1, plus + minus), 3)}))
