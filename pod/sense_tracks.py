"""sense_tracks.py TRACKS UMBILICUS.json [--z Z0 Z1] [--blocks 24] [--per 40000] [--label NAME]

Spiral sense from the sheet tracks: pooled least-squares slope of radius against polar angle (both relative to the scroll
axis at each point's z), over hundreds of thousands of tracks. For an Archimedean spiral dr/dtheta = gap/(2 pi) is the
same everywhere, so its sign is the rotation direction in which the sheet moves outward. Per block of the file the slope
is reported too: the decision is "decisive" when nearly every block agrees in sign.
The sign convention (which of +/- is villa's "CW") is calibrated by running the same code on scrolls of known sense
(PHerc0343 = CW per Nieuwlaar's recipe, PHerc0306B = ACW per Warner/Bodill's catalog), so the x/y column order of the
track store and the image-coordinate handedness cancel out.

TRACKS is a local <name>.dbm.vctracks directory or the https URL of one (HTTP range requests: metadata.json, z_bounds.i32,
offsets.i64 of each block of tracks, that block of coordinates.i32). Coordinates are int32 triples (z, c1, c2); c1, c2 are
taken as (y, x) consistently for all scrolls.
"""
import argparse, json, os, sys, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('tracks'); ap.add_argument('umb')
ap.add_argument('--z', type=float, nargs=2, default=None); ap.add_argument('--label', default='')
ap.add_argument('--blocks', type=int, default=24); ap.add_argument('--per', type=int, default=40000)
ap.add_argument('--rmin', type=float, default=150.0, help='ignore points closer than this to the axis')
a = ap.parse_args()
remote = a.tracks.startswith('http')


def read(name, start=None, length=None):
    if remote:
        req = urllib.request.Request(a.tracks.rstrip('/') + '/' + name)
        if start is not None:
            req.add_header('Range', f'bytes={start}-{start + length - 1}')
        return urllib.request.urlopen(req, timeout=300).read()
    with open(os.path.join(a.tracks, name), 'rb') as f:
        if start is not None:
            f.seek(start); return f.read(length)
        return f.read()


m = json.loads(read('metadata.json')); n_tr, n_pt = m['track_count'], m['point_count']
cp = sorted(json.load(open(a.umb))['control_points'], key=lambda q: q['z'])
uz = np.array([q['z'] for q in cp], float); ux = np.array([q['x'] for q in cp], float); uy = np.array([q['y'] for q in cp], float)
if a.z is None:
    a.z = (uz.min(), uz.max())
zb = np.frombuffer(read('z_bounds.i32'), dtype='<i4').reshape(-1, 2)
inside = np.nonzero((zb[:, 0] >= a.z[0]) & (zb[:, 1] <= a.z[1]))[0]
print(f'{a.label or a.tracks}: {n_tr:,} tracks; z {a.z[0]:.0f}..{a.z[1]:.0f}: {len(inside):,} tracks inside', file=sys.stderr)
if len(inside) == 0:
    sys.exit('no tracks in window')
starts = np.linspace(0, max(0, len(inside) - a.per), a.blocks).astype(int)
Sxx_t = Sxy_t = 0.0; block_slopes = []; used = 0; pts_used = 0
for s in starts:
    idx = inside[s:s + a.per]
    i0, i1 = int(idx[0]), int(idx[-1]) + 1
    off = np.frombuffer(read('offsets.i64', i0 * 8, (i1 - i0 + 1) * 8), dtype='<i8')
    p0, p1 = int(off[0]), int(off[-1])
    c = np.frombuffer(read('coordinates.i32', p0 * 12, (p1 - p0) * 12), dtype='<i4').reshape(-1, 3).astype(np.float64)
    z, y, x = c[:, 0], c[:, 1], c[:, 2]
    dx = x - np.interp(z, uz, ux); dy = y - np.interp(z, uz, uy)
    r = np.hypot(dx, dy); th = np.arctan2(dy, dx)
    # per-track means via reduceat over the contiguous point ranges of the tracks in this block (all tracks of [i0, i1))
    b = (off[:-1] - p0).astype(np.int64); e = (off[1:] - p0).astype(np.int64); n = e - b
    keep = n >= 4
    bk = b[keep]; nk = n[keep].astype(float)
    # angle relative to each track's first point (no unwrap needed for |dtheta| < pi)
    # build a per-point "track id" to subtract per-track means
    tid = np.repeat(np.arange(len(b)), n)
    first = th[b][tid]
    rel = np.arctan2(np.sin(th - first), np.cos(th - first))
    ok = (r > a.rmin)
    w = ok.astype(float)
    cnt = np.bincount(tid, weights=w, minlength=len(b))
    mth = np.bincount(tid, weights=w * rel, minlength=len(b)) / np.maximum(cnt, 1)
    mr = np.bincount(tid, weights=w * r, minlength=len(b)) / np.maximum(cnt, 1)
    dth = rel - mth[tid]; dr = r - mr[tid]
    good = (cnt >= 4)[tid] & ok
    Sxx = float(np.sum(dth[good] ** 2)); Sxy = float(np.sum(dth[good] * dr[good]))
    if Sxx > 0:
        block_slopes.append(Sxy / Sxx); Sxx_t += Sxx; Sxy_t += Sxy
    used += int((cnt >= 4).sum()); pts_used += int(good.sum())
bs = np.array(block_slopes); slope = Sxy_t / Sxx_t
agree = float((np.sign(bs) == np.sign(slope)).mean())
print(json.dumps({'label': a.label, 'tracks_used': used, 'points_used': pts_used, 'blocks': len(bs),
                  'pooled_slope_vox_per_rad': round(slope, 2), 'block_slopes_median': round(float(np.median(bs)), 2),
                  'blocks_agreeing_with_sign': round(agree, 3), 'sign': '+' if slope > 0 else '-',
                  'decisive': bool(agree >= 0.85)}))
