"""seam_test.py M7_ZARR FITTED_DIR WINDINGS.txt [--bins 36] [--tol 2] [--label NAME]

For every listed winding (fitted, unsnapped tifxyz under FITTED_DIR/<w>/{x,y,z}.tif), sample the m7 level-0 prediction
at each mesh node (max over a (2 tol + 1)^3 neighbourhood) and report the fraction of nodes that sit on a predicted sheet,
per column bin (the column axis of the grid runs around the winding). A fit with the wrong spiral sense must cross from
one sheet to the next somewhere in each turn: that shows as a low-valued bin (seam) recurring at similar columns across
windings. Prints per-winding mean / min-bin / argmin and the aggregate over windings.
"""
import argparse, json, os, sys
import numpy as np, tifffile, zarr

ap = argparse.ArgumentParser()
ap.add_argument('zarr'); ap.add_argument('fitted'); ap.add_argument('windings')
ap.add_argument('--bins', type=int, default=36); ap.add_argument('--tol', type=int, default=2); ap.add_argument('--label', default='')
ap.add_argument('--max-nodes', type=int, default=40000)
a = ap.parse_args()
arr = zarr.open_group(a.zarr, mode='r')['0']; Z, Y, X = arr.shape
ws = [l.strip() for l in open(a.windings) if l.strip()]
rows = []; prof_sum = np.zeros(a.bins); prof_n = np.zeros(a.bins)
for w in ws:
    d = os.path.join(a.fitted, w)
    if not os.path.exists(os.path.join(d, 'x.tif')):
        continue
    xs = tifffile.imread(os.path.join(d, 'x.tif')).astype(np.float64); ys = tifffile.imread(os.path.join(d, 'y.tif')).astype(np.float64)
    zs = tifffile.imread(os.path.join(d, 'z.tif')).astype(np.float64)
    H, W = xs.shape
    valid = (xs > 0) & (ys > 0) & (zs > 0) & np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
    rr, cc = np.nonzero(valid)
    if len(rr) == 0:
        continue
    if len(rr) > a.max_nodes:
        sel = np.random.default_rng(0).choice(len(rr), a.max_nodes, replace=False); rr, cc = rr[sel], cc[sel]
    x = np.rint(xs[rr, cc]).astype(int); y = np.rint(ys[rr, cc]).astype(int); z = np.rint(zs[rr, cc]).astype(int)
    ok = (x >= a.tol) & (x < X - a.tol) & (y >= a.tol) & (y < Y - a.tol) & (z >= a.tol) & (z < Z - a.tol)
    x, y, z, cc = x[ok], y[ok], z[ok], cc[ok]
    # read the band of z the mesh spans once (level 0 chunks are local for the band), then index
    z0, z1 = z.min() - a.tol, z.max() + a.tol + 1
    y0, y1 = y.min() - a.tol, y.max() + a.tol + 1; x0, x1 = x.min() - a.tol, x.max() + a.tol + 1
    vol = np.asarray(arr[z0:z1, y0:y1, x0:x1]) > 0
    on = np.zeros(len(x), bool)
    for dz in range(-a.tol, a.tol + 1):
        for dy in range(-a.tol, a.tol + 1):
            for dx in range(-a.tol, a.tol + 1):
                on |= vol[z - z0 + dz, y - y0 + dy, x - x0 + dx]
    b = np.minimum((cc * a.bins) // W, a.bins - 1)
    cnt = np.bincount(b, minlength=a.bins); hit = np.bincount(b, weights=on, minlength=a.bins)
    prof = np.where(cnt > 0, hit / np.maximum(cnt, 1), np.nan)
    prof_sum += np.nan_to_num(prof); prof_n += (cnt > 0)
    rows.append((w, float(on.mean()), float(np.nanmin(prof)), int(np.nanargmin(prof)), int(len(x))))
    print(f'{w} on_sheet={on.mean():.3f} min_bin={np.nanmin(prof):.3f} at bin {int(np.nanargmin(prof))} nodes={len(x)}', file=sys.stderr)
prof = prof_sum / np.maximum(prof_n, 1)
mins = np.array([r[2] for r in rows]); means = np.array([r[1] for r in rows]); arg = np.array([r[3] for r in rows])
hist = np.bincount(arg, minlength=a.bins)
print(json.dumps({'label': a.label, 'windings': len(rows), 'mean_on_sheet': round(float(means.mean()), 4),
                  'mean_of_min_bin': round(float(mins.mean()), 4), 'seam_depth': round(float((means - mins).mean()), 4),
                  'argmin_histogram_top': [(int(i), int(hist[i])) for i in np.argsort(-hist)[:4]],
                  'profile': [round(float(v), 3) for v in prof]}))
