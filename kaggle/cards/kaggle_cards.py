"""Kaggle CPU kernel: review cards for the top tri_rank windows and line_find chains of the band0800-ink run.
Inputs: band0800-ink output (maps.tar), band0800-render-0..3 (RENDERS/<w>.zarr for the CT middle layer).
Each card = CT middle layer | V3 z map (clipped -2..4) | 306B overlay, stacked vertically, for one window (2600 px wide,
full height) or one chain (1800 x 1000 px around its centre, polygon drawn). Output: /kaggle/working/cards/*.jpg + index.tsv.
"""
import glob, json, os, pathlib, subprocess, sys, time
import numpy as np

TARGETS = json.loads(r'''[{"kind": "row", "name": "b0800_w082", "side": "rev", "x0": 2600, "score": 7.26, "glyph": 10.953}, {"kind": "row", "name": "b0800_w085", "side": "fwd", "x0": 7800, "score": 4.21, "glyph": 10.836}, {"kind": "row", "name": "b0800_w063", "side": "rev", "x0": 5200, "score": 4.16, "glyph": 9.953}, {"kind": "row", "name": "b0800_w085", "side": "rev", "x0": 7800, "score": 3.4, "glyph": 10.312}, {"kind": "row", "name": "b0800_w089", "side": "fwd", "x0": 2600, "score": 3.3, "glyph": 7.355}, {"kind": "row", "name": "b0800_w080", "side": "fwd", "x0": 3900, "score": 3.29, "glyph": 9.703}, {"kind": "row", "name": "b0800_w064", "side": "rev", "x0": 1300, "score": 3.13, "glyph": 11.047}, {"kind": "row", "name": "b0800_w068", "side": "fwd", "x0": 2600, "score": 2.54, "glyph": 10.68}, {"kind": "row", "name": "b0800_w076", "side": "fwd", "x0": 2600, "score": 1.93, "glyph": 10.086}, {"kind": "row", "name": "b0800_w074", "side": "rev", "x0": 5200, "score": 1.51, "glyph": 10.625}, {"kind": "row", "name": "b0800_w083", "side": "rev", "x0": 2600, "score": 1.5, "glyph": 9.039}, {"kind": "row", "name": "b0800_w078", "side": "rev", "x0": 7800, "score": 1.37, "glyph": 11.539}, {"kind": "glyph", "name": "b0800_w066", "side": "fwd", "x0": 2600, "score": -0.77, "glyph": 13.047}, {"kind": "glyph", "name": "b0800_w087", "side": "rev", "x0": 5200, "score": 3.54, "glyph": 12.734}, {"kind": "glyph", "name": "b0800_w074", "side": "fwd", "x0": 2600, "score": -0.37, "glyph": 12.672}, {"kind": "glyph", "name": "b0800_w068", "side": "fwd", "x0": 5200, "score": 0.97, "glyph": 12.484}, {"kind": "glyph", "name": "b0800_w072", "side": "rev", "x0": 2600, "score": 2.07, "glyph": 11.75}, {"kind": "glyph", "name": "b0800_w061", "side": "fwd", "x0": 2600, "score": -0.03, "glyph": 11.695}, {"kind": "glyph", "name": "b0800_w085", "side": "fwd", "x0": 6500, "score": -0.13, "glyph": 11.555}, {"kind": "glyph", "name": "b0800_w075", "side": "rev", "x0": 7800, "score": -0.15, "glyph": 11.453}, {"kind": "glyph", "name": "b0800_w083", "side": "fwd", "x0": 3900, "score": -0.18, "glyph": 11.359}, {"kind": "line", "name": "b0800_w068", "side": "fwd", "cx": 6193, "cy": 818, "score": 8.942, "poly": [[5199.1, 422.6], [7260.3, 753.9], [7186.6, 1212.5], [5125.4, 881.2]]}, {"kind": "line", "name": "b0800_w075", "side": "rev", "cx": 7830, "cy": 1632, "score": 7.89, "poly": [[7081.5, 1211.9], [8688.5, 1642.5], [8578.6, 2052.7], [6971.6, 1622.1]]}, {"kind": "line", "name": "b0800_w089", "side": "rev", "cx": 3040, "cy": 1533, "score": 7.851, "poly": [[2466.2, 1228.4], [3672.5, 1387.2], [3613.1, 1837.8], [2406.9, 1679.0]]}, {"kind": "line", "name": "b0800_w087", "side": "rev", "cx": 6305, "cy": 787, "score": 7.63, "poly": [[3941.7, 654.6], [8634.5, 368.7], [8668.0, 918.6], [3975.2, 1204.5]]}, {"kind": "line", "name": "b0800_w074", "side": "rev", "cx": 6976, "cy": 962, "score": 7.559, "poly": [[5991.2, 492.7], [8067.2, 947.2], [7961.4, 1430.4], [5885.4, 975.9]]}, {"kind": "line", "name": "b0800_w065", "side": "rev", "cx": 7636, "cy": 2104, "score": 7.53, "poly": [[6997.5, 1605.8], [8444.1, 2058.2], [8273.8, 2602.5], [6827.2, 2150.1]]}, {"kind": "line", "name": "b0800_w086", "side": "fwd", "cx": 7833, "cy": 1808, "score": 7.421, "poly": [[6253.2, 1509.5], [9437.4, 1697.2], [9413.3, 2105.8], [6229.1, 1918.1]]}, {"kind": "line", "name": "b0800_w078", "side": "rev", "cx": 5734, "cy": 430, "score": 7.247, "poly": [[4985.3, 181.7], [6483.7, 181.7], [6483.7, 679.2], [4985.3, 679.2]]}, {"kind": "line", "name": "b0800_w088", "side": "fwd", "cx": 8467, "cy": 842, "score": 7.227, "poly": [[7287.8, 499.6], [9687.4, 706.5], [9646.2, 1184.2], [7246.6, 977.3]]}, {"kind": "line", "name": "b0800_w063", "side": "rev", "cx": 6756, "cy": 1933, "score": 7.03, "poly": [[5588.5, 1388.0], [8044.3, 1932.4], [7923.3, 2478.4], [5467.4, 1933.9]]}, {"kind": "line", "name": "b0800_w077", "side": "fwd", "cx": 3969, "cy": 1613, "score": 6.984, "poly": [[2833.2, 1690.2], [4990.9, 1112.1], [5104.7, 1536.4], [2946.9, 2114.6]]}, {"kind": "line", "name": "b0800_w052", "side": "rev", "cx": 6667, "cy": 2171, "score": 6.957, "poly": [[5735.6, 2135.3], [7496.1, 1745.5], [7598.0, 2206.0], [5837.6, 2595.8]]}]''')
T = pathlib.Path("/kaggle/temp"); W = pathlib.Path("/kaggle/working"); OUT = W / "cards"; OUT.mkdir(parents=True, exist_ok=True)
T.mkdir(parents=True, exist_ok=True)


def sh(cmd):
    print("+", cmd, flush=True); subprocess.run(cmd, shell=True, check=True)


sh(f"{sys.executable} -m pip install -q zarr==3.4.1 opencv-python-headless numcodecs")
import cv2, zarr   # noqa: E402

tar = glob.glob("/kaggle/input/**/maps.tar", recursive=True)[0]
names = sorted({t["name"] for t in TARGETS})
sh(f"cd {T} && tar --wildcards -xf {tar} " + " ".join(f"'maps/{n}/{n}_d0_*_z.npy' 'maps/{n}/{n}_d0_*_overlay.png'" for n in names))
rens = {pathlib.Path(z).name[:-5]: z for z in glob.glob("/kaggle/input/**/RENDERS/*.zarr", recursive=True)}
print("renders", len(rens), "maps", len(list((T / "maps").iterdir())), flush=True)


def stretch(a, lo=1, hi=99.5):
    v = a[a > 0]
    if v.size == 0: return np.zeros(a.shape, np.uint8)
    l, h = np.percentile(v, [lo, hi]); return np.clip((a - l) / (h - l + 1e-6) * 255, 0, 255).astype(np.uint8)


def zimg(z, lo=-2.0, hi=4.0):
    return np.clip((z - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


idx = open(OUT / "index.tsv", "w"); idx.write("card\tkind\tname\tside\tx0_or_cx\tcy\tscore\tglyph\n")
for i, t in enumerate(TARGETS):
    n, side = t["name"], t["side"]; w = n.rsplit("_", 1)[-1]; tag = "reverse" if side == "rev" else "fwd"
    z = np.load(T / "maps" / n / f"{n}_d0_{tag}_z.npy").astype(np.float32); H, Wd = z.shape
    ov = cv2.imread(str(T / "maps" / n / f"{n}_d0_{tag}_overlay.png"))
    ct = None
    if w in rens:
        try:
            arr = zarr.open_array(rens[w], mode="r"); ct = np.asarray(arr[arr.shape[0] // 2])
        except Exception as e:
            print("ct fail", w, repr(e)[:120], flush=True)
    if t["kind"] in ("row", "glyph"):
        x0 = int(t["x0"]); x1 = min(Wd, x0 + 2600)
        valid_rows = np.nonzero((z[:, x0:x1] != 0).any(1))[0]
        y0, y1 = (int(valid_rows.min()), int(valid_rows.max()) + 1) if valid_rows.size else (0, H)
        y0 = max(0, y0 - 40); y1 = min(H, y1 + 40)
        label = f"{t['kind']} {n} {side} x0={x0} row={t['score']} glyph={t['glyph']}"; stem = f"{i:02d}_{t['kind']}_{n}_{side}_x{x0}"; cx = x0
    else:
        cx, cy = int(t["cx"]), int(t["cy"]); x0, x1 = max(0, cx - 900), min(Wd, cx + 900); y0, y1 = max(0, cy - 500), min(H, cy + 500)
        label = f"line {n} {side} centre=({cx},{cy}) score={t['score']}"; stem = f"{i:02d}_line_{n}_{side}_x{cx}_y{cy}"
    panels = []
    if ct is not None and ct.shape == z.shape:
        panels.append(cv2.cvtColor(stretch(ct[y0:y1, x0:x1].astype(np.float32)), cv2.COLOR_GRAY2BGR))
    zc = cv2.cvtColor(zimg(z[y0:y1, x0:x1]), cv2.COLOR_GRAY2BGR)
    if t["kind"] == "line":
        pts = np.array([[int(p[0]) - x0, int(p[1]) - y0] for p in t["poly"]], np.int32)
        for pnl in panels + [zc]: cv2.polylines(pnl, [pts], True, (0, 0, 255), 2)
    panels.append(zc)
    o = cv2.resize(ov[y0 // 3:(y1 + 2) // 3, x0 // 3:(x1 + 2) // 3], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR); panels.append(o)
    scale = min(1.0, 1800.0 / (x1 - x0))
    panels = [cv2.resize(p, (int(p.shape[1] * scale), int(p.shape[0] * scale)), interpolation=cv2.INTER_AREA) for p in panels]
    card = np.concatenate(panels, axis=0)
    cv2.putText(card, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(OUT / f"{stem}.jpg"), card, [cv2.IMWRITE_JPEG_QUALITY, 82])
    idx.write(f"{stem}.jpg\t{t['kind']}\t{n}\t{side}\t{cx}\t{t.get('cy', '')}\t{t['score']}\t{t.get('glyph', '')}\n"); idx.flush()
    print(i, stem, card.shape, flush=True)
idx.close(); sh(f"du -sh {OUT}")
