"""Kaggle GPU kernel, follow-up of band0800-calib: line_find on every text surface and on the 40 null windings under the
same conditions, plus review cards for the two null windows the recalibrated rule keeps that had no card (w067 rev x3900/x5200).

band0800-calib ran line_find --lane on its lanes, but lane_maps() only lists maps named *_wNNN, so the PHerc. 0841 surfaces
(w00, ag144, ag174) and the PHerc. 0306B claim surface were skipped. Here every surface gets a wNNN alias. line_find runs
without CT for text and nulls alike (the CT only feeds its crack >= 2 veto), so the two sides are scored the same way.
Inputs: kernel outputs band0800-calib (maps_calib.tar), band0800-ink (maps.tar), band0800-render-0..3 (CT for the cards);
datasets band0800-code, band0800-calib-labels. Output: line_find_text.json, line_find_null.json, linefind_calib.json,
linefind_report.md, cards/*.jpg.
"""
import glob, json, os, pathlib, subprocess, sys, time

VILLA_COMMIT = "f7756c758"
T = pathlib.Path("/kaggle/temp"); W = pathlib.Path("/kaggle/working"); T.mkdir(parents=True, exist_ok=True)
VENV = T / "venv"; PY = str(VENV / "bin" / "python"); VILLA = T / "villa"; VSRC = VILLA / "vesuvius" / "src"
FLW = T / "flw"; LOG = W / "run_log.jsonl"
HF1447 = "https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/resolve/main"
# text surfaces: (calib lane tag, calib name suffix, alias lane, alias winding, um)
TEXT = [("c1447", "w058", "p1447", "w058", 8.64), ("c1447", "w060", "p1447", "w060", 8.64),
        ("c0841", "w00", "p0841", "w900", 9.366), ("c0841", "ag144", "p0841", "w901", 9.366), ("c0841", "ag174", "p0841", "w902", 9.366),
        ("c306", "claim", "p306", "w950", 8.64)]
CARDS = [("b0800_w067", "rev", 3900), ("b0800_w067", "rev", 5200)]


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def note(**rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S"); print("NOTE", json.dumps(rec), flush=True)
    with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")


CODE = pathlib.Path(glob.glob("/kaggle/input/**/fl/scripts/line_find.py", recursive=True)[0]).parents[2]


def stage_env():
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh(f"uv venv -q -p 3.14 {VENV}")
    sh(f"uv pip install -q -p {PY} -r {CODE / 'requirements.txt'}")
    sh(f"uv pip install -q -p {PY} --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126")
    for link, target in ((FLW / "x_glyph", CODE / "x_glyph"), (FLW / "fl", CODE / "fl")):
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists(): os.symlink(target, link)


def find_tar(name):
    r = [p for p in glob.glob(f"/kaggle/input/**/{name}", recursive=True)]
    if not r: raise SystemExit(f"{name} not found in inputs")
    return r[0]


def lane_dir(lane, um):
    R = FLW / "x_vast_plan" / "results" / lane; (R / "maps").mkdir(parents=True, exist_ok=True); (R / "results").mkdir(parents=True, exist_ok=True)
    (R / "results" / "lane.env").write_text(f"SCROLL={lane}\nUM={um}\nTB={lane}\n")
    return R


def stage_maps():
    t = time.time()
    # text maps from band0800-calib
    tc = find_tar("maps_calib.tar"); X = T / "xcalib"; X.mkdir(exist_ok=True)
    sh(f"tar --wildcards -xf {tc} -C {X} 'maps/*_d0_*_z.npy'")
    for tag, nm, lane, w, um in TEXT:
        R = lane_dir(lane, um); src = X / "maps" / f"{tag}_{nm}"; dst = R / "maps" / f"{lane}_{w}"; dst.mkdir(exist_ok=True)
        for side in ("fwd", "reverse"):
            f = src / f"{tag}_{nm}_d0_{side}_z.npy"
            if f.exists() and not (dst / f"{lane}_{w}_d0_{side}_z.npy").exists(): os.symlink(f, dst / f"{lane}_{w}_d0_{side}_z.npy")
    # null maps from band0800-ink
    tn = find_tar("maps.tar"); R = lane_dir("b0800", 8.64)
    sh(f"tar --wildcards -xf {tn} -C {R} 'maps/*_d0_*_z.npy'")
    note(stage="maps", secs=round(time.time() - t), text=len(list((FLW / "x_vast_plan" / "results").glob("p*/maps/*"))),
         null=len(list((R / "maps").glob("b0800_w*"))))


def stage_linefind():
    FS = CODE / "fl" / "scripts"; e = dict(os.environ, PYTHONPATH=str(VSRC), PYTHONUNBUFFERED="1", FL_W=str(FLW), OMP_NUM_THREADS="4")
    res = {}
    for lane in ("p1447", "p0841", "p306", "b0800"):
        um = "9.366" if lane == "p0841" else "8.64"; out = T / f"lf_{lane}.json"; t = time.time()
        r = subprocess.run([PY, str(FS / "line_find.py"), "--lane", lane, "--faces", "both", "--um", um, "--device", "cuda", "--out", str(out)],
                           env=dict(e, FL_UM=um), stdout=open(W / f"line_find_{lane}.log", "w"), stderr=subprocess.STDOUT)
        note(stage="line_find", lane=lane, rc=r.returncode, secs=round(time.time() - t))
        if out.exists(): res[lane] = json.load(open(out))
    text = dict(maps=[m for k in ("p1447", "p0841", "p306") for m in res.get(k, {}).get("maps", [])])
    json.dump(text, open(W / "line_find_text.json", "w")); json.dump(res.get("b0800", {}), open(W / "line_find_null.json", "w"))


ANALYSIS = r'''
import json, sys, pathlib, urllib.request, glob, shutil
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
W, T, HF = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
TEXT = json.loads(sys.argv[4]); alias = {f"{lane}_{w}": (tag, nm) for tag, nm, lane, w, um in TEXT}
CLAIM_ROWS, CLAIM_COLS = (2030, 2830), (1389, 6018)

def get(url, dest):
    if dest.exists(): return
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "band0800-calib2"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f: shutil.copyfileobj(r, f)

def labels(tag, nm, shape):
    H, Wd = shape
    if tag == "c1447":
        d = T / "lab" / nm
        for f in ("inklabels.png", "mask.png"): get(f"{HF}/{nm}/labels/{f}", d / f)
        ink = np.asarray(Image.open(d / "inklabels.png")); ink = ink[..., 0] if ink.ndim == 3 else ink
        L = np.zeros(shape, bool); h, w_ = min(H, ink.shape[0]), min(Wd, ink.shape[1]); L[:h, :w_] = ink[:h, :w_] > 0; return L
    if tag == "c0841":
        p = [x for x in glob.glob("/kaggle/input/**/label.npy", recursive=True) if pathlib.Path(x).parent.name == nm][0]
        d = pathlib.Path(p).parent; lab = np.load(d / "label.npy"); i0, i1, j0, j1 = json.load(open(d / "meta.json"))["crop"]
        L = np.zeros(shape, bool); h, w_ = min(i1, H) - i0, min(j1, Wd) - j0; L[i0:i0 + h, j0:j0 + w_] = lab[:h, :w_] > 0; return L
    L = np.zeros(shape, bool); L[CLAIM_ROWS[0]:CLAIM_ROWS[1], CLAIM_COLS[0]:CLAIM_COLS[1]] = True; return L

def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0: return None
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float) + 1
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True); r = (np.bincount(inv, r) / cnt)[inv]
    return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)

text = json.load(open(W / "line_find_text.json")); null = json.load(open(W / "line_find_null.json"))
null_scores = [c["score"] for m in null.get("maps", []) for c in m.get("candidates", [])]
null_cm2 = sum(m.get("onsheet_cm2", 0) for m in null.get("maps", []))
on, off, per = [], [], {}
for m in text["maps"]:
    tag, nm = alias[m["name"]]; z = np.load(m["zmap"], mmap_mode="r"); L = labels(tag, nm, z.shape)
    best_on = best_off = None
    for c in m.get("candidates", []):
        pts = np.array(c["poly"]); cy, cx = int(pts[:, 1].mean()), int(pts[:, 0].mean())
        hit = bool(L[max(0, cy - 60):cy + 60, max(0, cx - 400):cx + 400].any())
        (on if hit else off).append(c["score"])
        if hit: best_on = c["score"] if best_on is None else max(best_on, c["score"])
        else: best_off = c["score"] if best_off is None else max(best_off, c["score"])
    per[f"{tag}_{nm}:{m['side']}"] = dict(chains=len(m.get("candidates", [])), best_on_text=best_on, best_off_text=best_off, cm2=m.get("onsheet_cm2"))
ns = np.array(null_scores)
res = dict(null_chains=len(ns), null_cm2=round(null_cm2, 1), null_max=float(ns.max()) if len(ns) else None,
           null_p99=round(float(np.percentile(ns, 99)), 3) if len(ns) else None,
           text_chains_on_text=len(on), text_chains_off_text=len(off), auc_on_text_vs_null=auc(on, ns), auc_on_vs_off_text=auc(on, off), per_map=per)
for thr in (6.0, 6.8, 7.5, 8.0, 9.0):
    res[f"at_{thr}"] = dict(null_lines_per_100cm2=round(float((ns >= thr).sum()) / max(null_cm2, 1e-6) * 100, 2),
                            text_maps_with_line_on_text=sum(1 for v in per.values() if v["best_on_text"] is not None and v["best_on_text"] >= thr),
                            text_maps=len(per))
json.dump(res, open(W / "linefind_calib.json", "w"), indent=1)
Lr = ["# line_find on Reader V3 maps: text vs PHerc. 0800 nulls (no CT on either side)", "",
      f"null chains {res['null_chains']} on {res['null_cm2']} cm2, max {res['null_max']}, p99 {res['null_p99']}",
      f"text chains on labelled text {len(on)}, off text {len(off)}; AUC on-text vs null {res['auc_on_text_vs_null']}, on vs off text {res['auc_on_vs_off_text']}", "",
      "| threshold | null false lines per 100 cm2 | text faces with a line on text |", "|---|---|---|"]
for thr in (6.0, 6.8, 7.5, 8.0, 9.0):
    a = res[f"at_{thr}"]; Lr.append(f"| {thr} | {a['null_lines_per_100cm2']} | {a['text_maps_with_line_on_text']} of {a['text_maps']} |")
Lr += [""] + [f"- {k}: best on text {v['best_on_text']}, best off text {v['best_off_text']}, chains {v['chains']}" for k, v in per.items()]
(W / "linefind_report.md").write_text("\n".join(Lr) + "\n"); print("\n".join(Lr), flush=True)
'''


CARDS_PY = r'''
import glob, json, sys, pathlib, subprocess
import numpy as np, cv2, zarr
W, T = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); targets = json.loads(sys.argv[3]); OUT = W / "cards"; OUT.mkdir(exist_ok=True)
tar = glob.glob("/kaggle/input/**/maps.tar", recursive=True)[0]
names = sorted({t[0] for t in targets})
subprocess.run(f"cd {T} && tar --wildcards -xf {tar} " + " ".join(f"'maps/{n}/{n}_d0_*_overlay.png'" for n in names), shell=True, check=True)
rens = {pathlib.Path(z).name[:-5]: z for z in glob.glob("/kaggle/input/**/RENDERS/*.zarr", recursive=True)}
def stretch(a):
    v = a[a > 0]
    if v.size == 0: return np.zeros(a.shape, np.uint8)
    l, h = np.percentile(v, [1, 99.5]); return np.clip((a - l) / (h - l + 1e-6) * 255, 0, 255).astype(np.uint8)
zimg = lambda z: np.clip((z + 2.0) / 6.0 * 255, 0, 255).astype(np.uint8)
for n, side, x0 in targets:
    w = n.rsplit("_", 1)[-1]; tag = "reverse" if side == "rev" else "fwd"
    z = np.load(T / "lanes_null" / "maps" / n / f"{n}_d0_{tag}_z.npy").astype(np.float32); H, Wd = z.shape
    ov = cv2.imread(str(T / "maps" / n / f"{n}_d0_{tag}_overlay.png"))
    arr = zarr.open_array(rens[w], mode="r"); ct = np.asarray(arr[arr.shape[0] // 2])
    x1 = min(Wd, x0 + 2600); rows = np.nonzero((z[:, x0:x1] != 0).any(1))[0]
    y0, y1 = max(0, int(rows.min()) - 40), min(H, int(rows.max()) + 41)
    panels = [cv2.cvtColor(stretch(ct[y0:y1, x0:x1].astype(np.float32)), cv2.COLOR_GRAY2BGR), cv2.cvtColor(zimg(z[y0:y1, x0:x1]), cv2.COLOR_GRAY2BGR),
              cv2.resize(ov[y0 // 3:(y1 + 2) // 3, x0 // 3:(x1 + 2) // 3], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)]
    s = min(1.0, 1800.0 / (x1 - x0)); panels = [cv2.resize(p, (int(p.shape[1] * s), int(p.shape[0] * s)), interpolation=cv2.INTER_AREA) for p in panels]
    card = np.concatenate(panels, 0); cv2.putText(card, f"null {n} {side} x0={x0}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(str(OUT / f"{n}_{side}_x{x0}.jpg"), card, [cv2.IMWRITE_JPEG_QUALITY, 82]); print("card", n, side, x0, card.shape, flush=True)
'''


def stage_analysis():
    f = T / "analysis.py"; f.write_text(ANALYSIS)
    r = subprocess.run([PY, str(f), str(W), str(T), HF1447, json.dumps(TEXT)], stdout=open(W / "analysis.log", "w"), stderr=subprocess.STDOUT)
    note(stage="analysis", rc=r.returncode); print(open(W / "analysis.log").read()[-5000:], flush=True)


def stage_cards():
    lanes_null = T / "lanes_null"; lanes_null.mkdir(exist_ok=True)
    if not (lanes_null / "maps").exists(): os.symlink(FLW / "x_vast_plan" / "results" / "b0800" / "maps", lanes_null / "maps")
    f = T / "cards.py"; f.write_text(CARDS_PY)
    r = subprocess.run([PY, str(f), str(W), str(T), json.dumps(CARDS)], stdout=open(W / "cards.log", "w"), stderr=subprocess.STDOUT)
    note(stage="cards", rc=r.returncode); print(open(W / "cards.log").read()[-2000:], flush=True)


if __name__ == "__main__":
    t = time.time(); stage_env(); stage_maps(); stage_linefind(); stage_analysis(); stage_cards()
    note(stage="done", secs=round(time.time() - t))
