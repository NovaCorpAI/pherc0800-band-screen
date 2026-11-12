"""Kaggle GPU kernel: calibrate PHerc. 306B's First Letters triage for Reader V3 maps.

Positives (known text, Reader V3 at depth 0, both faces, the same worker and triage as band0800-ink):
  * PHerc. 1447 w058, w060 (8.64 um): tifxyz + inklabels/mask from HF YoussefMoNader/ink-8um-pherc1447-surfaces (CC BY-NC 4.0).
  * PHerc. 0841 w00, ag144, ag174 (9.366 um): the organisers' meshes from the open-data bucket; labels (20260918 human labels,
    resampled onto the mesh canvas) from the dataset band0800-calib-labels.
  * PHerc. 0306B claim surface (8.64 um): the 41-layer render published by Warner and Bodill (HF Prodigle/pherc0306b-first-letters,
    CC BY-NC 4.0), middle 31 layers; the claimed line sits near v = 0 mm of the claim frame (render row 2430).
Nulls: the 40 windings of PHerc. 0800 z 5700-7200 screened by band0800-ink (tri_rank.json, line_find.json), all visually negative.
Output (/kaggle/working): tri_rank_<tag>.json/.txt, line_find_<tag>.json, windows.csv (every positive and null window with its
features and label), calib.json + report.md (per-feature AUC, recalibrated thresholds, false alarms on the nulls), maps_calib.tar.
"""
import glob, json, os, pathlib, shutil, subprocess, sys, tarfile, time

UM_1447, UM_0841 = 8.64, 9.366
VILLA_COMMIT = "f7756c758"; V3_URL = "https://huggingface.co/domenicor046/reader-v3/resolve/main/reader-v3.pth"
S3 = "vesuvius-challenge-open-data"
V1447 = f"s3://{S3}/PHerc1447/volumes/20250521151220-8.640um-1.2m-116keV-masked.zarr"
V0841 = f"s3://{S3}/PHerc0841/volumes/20250821151531-9.366um-1.2m-113keV-masked.zarr"
SEG0841 = {"w00": "20260220213127-w00", "ag144": "20260220214732-auto_grown_20260220144552896",
           "ag174": "20260221022814-auto_grown_20260220174252405"}
HF1447 = "https://huggingface.co/datasets/YoussefMoNader/ink-8um-pherc1447-surfaces/resolve/main"
HF306 = "Prodigle/pherc0306b-first-letters"
GROUPS = [dict(tag="c1447", um=UM_1447, items=["w058", "w060"]),
          dict(tag="c0841", um=UM_0841, items=["w00", "ag144", "ag174"]),
          dict(tag="c306", um=UM_1447, items=["claim"])]
ONLY = [g for g in os.environ.get("GROUPS", "").split(",") if g]
if ONLY: GROUPS = [g for g in GROUPS if g["tag"] in ONLY]
T = pathlib.Path("/kaggle/temp"); W = pathlib.Path("/kaggle/working"); T.mkdir(parents=True, exist_ok=True)
VENV = T / "venv"; PY = str(VENV / "bin" / "python"); VILLA = T / "villa"; VSRC = VILLA / "vesuvius" / "src"
CK = T / "reader-v3.pth"; LOG = W / "run_log.jsonl"; FLW = T / "flw"; MESH = T / "mesh"; LAB = T / "labels"; ROUT = T / "rout"
for d in (MESH, LAB, ROUT, FLW): d.mkdir(parents=True, exist_ok=True)


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def note(**rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S"); print("NOTE", json.dumps(rec), flush=True)
    with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")


def find_code():
    r = glob.glob("/kaggle/input/**/fl/scripts/tri_rank.py", recursive=True)
    if not r: raise SystemExit("band0800-code dataset not attached")
    return pathlib.Path(r[0]).parents[2]


CODE = find_code()


def lane(tag):
    L = T / "lanes" / tag
    return L, L / "maps", L / "renders", L / "ct"


def stage_env():
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv || true")
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh(f"uv venv -q -p 3.14 {VENV}")
    sh(f"uv pip install -q -p {PY} -r {CODE / 'requirements.txt'}")
    sh(f"uv pip install -q -p {PY} --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126")
    if not VSRC.exists():
        sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git {VILLA}")
        sh(f"git -C {VILLA} sparse-checkout set vesuvius/src")
        sh(f"git -C {VILLA} checkout -q {VILLA_COMMIT}")
    if not CK.exists():
        sh(f"curl -sfL -o {CK} {V3_URL}")
    out = subprocess.run([PY, "-c", "import vesuvius.ink_detection.inference.infer as m, cv2, zarr, torch, s3fs, huggingface_hub; print(torch.__version__, torch.cuda.device_count())"],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(VSRC)))
    note(stage="env", out=out.stdout.strip(), err=out.stderr[-1200:])
    if out.returncode: raise SystemExit("env check failed")


FETCH = r'''
import json, sys, time, urllib.request, pathlib, shutil
MESH, LAB, ROUT, HF1447, HF306, S3 = [sys.argv[i] for i in range(1, 7)]
MESH, LAB, ROUT = pathlib.Path(MESH), pathlib.Path(LAB), pathlib.Path(ROUT)
SEG0841 = json.loads(sys.argv[7]); groups = sys.argv[8].split(",")

def get(url, dest):
    if dest.exists() and dest.stat().st_size > 0: return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "band0800-calib"})
            with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f: shutil.copyfileobj(r, f)
            return
        except Exception as e:
            print("retry", url, e, flush=True); time.sleep(2 ** k)
    raise IOError(url)

if "c1447" in groups:
    for w in ("w058", "w060"):
        for f in ("x.tif", "y.tif", "z.tif", "meta.json"): get(f"{HF1447}/{w}/tifxyz/{f}", MESH / "c1447" / w / f)
        for f in ("inklabels.png", "mask.png"): get(f"{HF1447}/{w}/labels/{f}", LAB / f"c1447_{w}" / f)
if "c0841" in groups:
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    for name, seg in SEG0841.items():
        d = MESH / "c0841" / name
        if (d / "meta.json").exists(): continue
        base = f"{S3}/PHerc0841/segments/{seg}/mesh/"
        mesh = [p for p in fs.ls(base) if p.endswith("9.366um.tifxyz")][0]
        fs.get(mesh, str(d), recursive=True)
        if not (d / "meta.json").exists():   # s3fs may nest the dir
            sub = [p for p in d.rglob("meta.json")]
            for f in sub[0].parent.iterdir(): shutil.move(str(f), d / f.name)
if "c306" in groups:
    from huggingface_hub import snapshot_download
    import zarr, numpy as np
    out = ROUT / "c306_claim.zarr"
    if not (out / "zarr.json").exists():
        p = snapshot_download(HF306, repo_type="dataset", allow_patterns=["results/claim_surface_render_41layers.zarr/.z*",
                              "results/claim_surface_render_41layers.zarr/0/**"], local_dir=str(ROUT / "hf306"), max_workers=16)
        src = zarr.open_array(str(pathlib.Path(p) / "results/claim_surface_render_41layers.zarr/0"), mode="r", zarr_format=2)
        D = src.shape[0]; c = D // 2; v = np.asarray(src[c - 15:c + 16])
        dst = zarr.open_array(str(out), mode="w", shape=v.shape, chunks=(v.shape[0], 128, 128), dtype="uint8", fill_value=0)
        dst[:] = v; np.save(ROUT / "c306_claim.valid.npy", v[15] > 0)
        shutil.rmtree(ROUT / "hf306", ignore_errors=True)
        print(json.dumps(dict(claim_shape=list(v.shape), src_layers=D, valid_px=int((v[15] > 0).sum()))), flush=True)
print("fetch done", flush=True)
'''


def stage_fetch():
    t = time.time(); f = T / "fetch.py"; f.write_text(FETCH)
    sh([PY, str(f), str(MESH), str(LAB), str(ROUT), HF1447, HF306, S3, json.dumps(SEG0841), ",".join(g["tag"] for g in GROUPS)],
       env=dict(os.environ, PYTHONPATH=str(VSRC), PYTHONUNBUFFERED="1"))
    labs = [p for p in glob.glob("/kaggle/input/**/label.npy", recursive=True)
            if pathlib.Path(p).parent.name in SEG0841 and (pathlib.Path(p).parent / "support.npy").exists()]
    for p in labs:
        d = pathlib.Path(p).parent; dst = LAB / f"c0841_{d.name}"
        if not dst.exists(): shutil.copytree(d, dst)
    note(stage="fetch", secs=round(time.time() - t), labels=sorted(x.name for x in LAB.iterdir()),
         meshes=sorted(str(p.relative_to(MESH)) for p in MESH.glob("*/*")))


def stage_render():
    render_py = CODE / "render.py"; jobs = []
    for g in GROUPS:
        for w in g["items"]:
            if g["tag"] == "c306": continue
            out = ROUT / f"{g['tag']}_{w}.zarr"
            if (ROUT / f"{g['tag']}_{w}.valid.npy").exists(): continue
            vol = V1447 if g["tag"] == "c1447" else V0841
            jobs.append((f"{g['tag']}_{w}", [PY, str(render_py), str(MESH / g["tag"] / w), vol, str(out), "--layers", "31", "--cache-gb", "3"]))
    running = {}; t0 = time.time()
    while jobs or running:
        while jobs and len(running) < 3:
            n, cmd = jobs.pop(0)
            running[n] = (subprocess.Popen(cmd, env=dict(os.environ, PYTHONPATH=str(VSRC), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1"),
                                           stdout=open(W / f"render_{n}.log", "w"), stderr=subprocess.STDOUT), time.time())
        for n, (p, ts) in list(running.items()):
            if p.poll() is not None:
                ok = (ROUT / f"{n}.valid.npy").exists()
                note(stage="render", item=n, rc=p.returncode, ok=ok, secs=round(time.time() - ts))
                if not ok: print(open(W / f"render_{n}.log").read()[-2000:], flush=True)
                del running[n]
        time.sleep(10)
    note(stage="render_all", secs=round(time.time() - t0))


WORKER = r'''
import json, os, sys, time, pathlib
import numpy as np, zarr, cv2, tifffile, torch
from dataclasses import replace
import vesuvius.ink_detection.inference.infer as INF
TAG, UM, OFFS = sys.argv[1], float(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")]
REN, MAPS, CK = pathlib.Path(sys.argv[4]), pathlib.Path(sys.argv[5]), sys.argv[6]
ws = sys.argv[7].split(","); V3 = 21; B = 48
QTHR = float(os.environ.get("FL_QUALITY_THR", "0.10"))

def quality(v, tile=128):   # fastink.quality_mask: per-tile depth-profile contrast relative to the median mid-layer value
    D = v.shape[0]; mid = v[D // 2]; valid = mid > 0
    med = float(np.median(mid[valid])) if valid.any() else 1.0
    H, Wd = valid.shape; ny, nx = H // tile, Wd // tile; sc = np.full((ny, nx), np.nan, np.float32)
    for i in range(ny):
        for j in range(nx):
            s = (slice(i * tile, (i + 1) * tile), slice(j * tile, (j + 1) * tile)); vv = valid[s]
            if vv.mean() < 0.8: continue
            p = v[:, s[0], s[1]][:, vv].astype(np.float32).mean(1); sc[i, j] = (p.max() - p.min()) / max(med, 1.0)
    good = np.nan_to_num(sc, nan=-1) >= QTHR; m = np.zeros((H, Wd), bool); m[:ny * tile, :nx * tile] = np.kron(good, np.ones((tile, tile), bool))
    return m, sc

def fuse(a, ct, valid, stat):   # fastpipe.fuse with one model
    a = a.astype(np.float32); v = a[stat]; z = (a - v.mean()) / (v.std() + 1e-6)
    z = (z - z[stat].mean()) / (z[stat].std() + 1e-6); z[~valid] = 0
    z[:B] = 0; z[-B:] = 0; z[:, :B] = 0; z[:, -B:] = 0
    return z

def overlay(ct, z, down=3):   # fastpipe.overlay
    c = cv2.resize(ct, (ct.shape[1] // down, ct.shape[0] // down), interpolation=cv2.INTER_AREA)
    zz = cv2.resize(z, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_AREA); vv = c > 0
    lo, hi = np.percentile(c[vv], [1, 99.5]) if vv.any() else (0, 255)
    g = np.clip((c - lo) / (hi - lo + 1e-6), 0, 1) * 0.45; a = np.clip((zz - 0.9) / 1.8, 0, 1)
    return (np.stack([g * (1 - a) + a * .35, g * (1 - a) + a * .85, g * (1 - a) + a * .95], -1) * 255).astype(np.uint8)

def to_u8(p):
    p = p.astype(np.float32)
    if p.max() > 1.0: p = p / (65535.0 if p.max() > 255 else 255.0)
    return np.rint(np.clip(p, 0, 1) * 255).astype(np.uint8)

iargs = INF.normalize_inference_paths(INF.parse_args(["unused.zarr", CK, "unused.tif", "--no-compile", "--num-workers", "0",
                                                      "--loader-threads", "8", "--batch-size", "16", "--device", "cuda"]))
torch.set_float32_matmul_precision("high")
cfg = INF.configure_model(iargs)
model, device = INF.prepare_model_for_inference(cfg.model, gpu_ids=iargs.gpu_ids, device_type=iargs.device, compile_model=False, compile_mode=iargs.compile_mode)
cfg = replace(cfg, model=model)
for w in ws:
    name = f"{TAG}_{w}"; out = MAPS / name; out.mkdir(parents=True, exist_ok=True)
    if (out / "done_d0").exists(): continue
    t0 = time.time(); z = str(REN / f"{name}.zarr" / "0")
    arr = zarr.open_array(z, mode="r"); v = np.asarray(arr[:]); D, H, Wd = v.shape; c = D // 2
    q, sc = quality(v); np.save(out / "quality_tiles.npy", sc)
    cv2.imwrite(str(MAPS.parent / "ct" / f"{w}.jpg"), v[c][::3, ::3], [cv2.IMWRITE_JPEG_QUALITY, 85])   # line_find's CT at 1/3
    timing = {}
    for d in OFFS:
        ct = v[c + d].astype(np.float32); valid = ct > 0
        stat = (q & valid) if (q & valid).sum() > 0.2 * valid.sum() else valid
        for dirn in ("forward", "reverse"):
            t = time.time(); s = c + d - V3 // 2; iargs.layer_start, iargs.layer_end = s, s + V3
            tif = out / f"tmp_{dirn}_{d}.tif"
            INF.infer_single_zarr(args=iargs, input_zarr=z, configured_model=cfg, device=device, output_tiff=tif, layer_direction=dirn)
            m = to_u8(tifffile.imread(tif)); tif.unlink(missing_ok=True)
            if m.shape != ct.shape: m = cv2.resize(m, (Wd, H), interpolation=cv2.INTER_LINEAR)
            suf = "_reverse" if dirn == "reverse" else ""; tag = "reverse" if dirn == "reverse" else "fwd"
            cv2.imwrite(str(out / f"{name}__v3_d{d}{suf}.png"), m)
            zf = fuse(m, ct, valid, stat)
            np.save(out / f"{name}_d{d}_{tag}_z.npy", zf.astype(np.float16))
            cv2.imwrite(str(out / f"{name}_d{d}_{tag}_overlay.png"), overlay(ct, zf))
            timing[f"v3_d{d}_{dirn}"] = round(time.time() - t, 1)
    json.dump(dict(recipe="v3", weights={"v3": 1.0}, dirs=["forward", "reverse"], offsets=OFFS, mask=False, seconds=timing,
                   grid_cm2=round(H * Wd * (UM * 1e-4) ** 2, 3), valid_cm2=round(float((v[c] > 0).sum()) * (UM * 1e-4) ** 2, 3),
                   quality_tiles_good=int((np.nan_to_num(sc, nan=-1) >= QTHR).sum()), tiles=int(np.isfinite(sc).sum())),
              open(out / f"{name}_timing.json", "w"), indent=1)
    (out / "done_d0").write_text("v3 kaggle\n")
    print(json.dumps(dict(winding=w, secs=round(time.time() - t0), timing=timing)), flush=True)
'''


def stage_ink():
    worker = T / "ink_worker.py"; worker.write_text(WORKER)
    for g in GROUPS:
        L, MAPS, REN, CT = lane(g["tag"])
        for d in (MAPS, REN, CT): d.mkdir(parents=True, exist_ok=True)
        ws = []
        for w in g["items"]:
            src = ROUT / f"{g['tag']}_{w}.zarr"
            if not src.exists(): note(stage="ink_skip", item=f"{g['tag']}_{w}"); continue
            gr = REN / f"{g['tag']}_{w}.zarr"; gr.mkdir(exist_ok=True)
            (gr / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group"}))
            if not (gr / "0").exists(): os.symlink(src, gr / "0")
            ws.append(w)
        R = FLW / "x_vast_plan" / "results" / g["tag"]; (R / "results").mkdir(parents=True, exist_ok=True)
        for link, target in ((R / "maps", MAPS), (R / "results" / "ct", CT)):
            if not link.exists(): os.symlink(target, link)
        (R / "results" / "lane.env").write_text(f"SCROLL={g['tag']}\nUM={g['um']}\nTB={g['tag']}\n")
        if not ws: continue
        t = time.time()
        r = subprocess.run([PY, str(worker), g["tag"], str(g["um"]), "0", str(REN), str(MAPS), str(CK), ",".join(ws)],
                           env=dict(os.environ, PYTHONPATH=str(VSRC), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2"),
                           stdout=open(W / f"ink_{g['tag']}.log", "w"), stderr=subprocess.STDOUT)
        note(stage="ink", tag=g["tag"], rc=r.returncode, items=ws, secs=round(time.time() - t))
        if r.returncode: print(open(W / f"ink_{g['tag']}.log").read()[-3000:], flush=True)
    for link, target in ((FLW / "x_glyph", CODE / "x_glyph"), (FLW / "fl", CODE / "fl")):
        if not link.exists(): os.symlink(target, link)


def stage_triage():
    FS = CODE / "fl" / "scripts"
    for g in GROUPS:
        tag, um = g["tag"], g["um"]; L, MAPS, REN, CT = lane(tag)
        if not MAPS.exists() or not any(MAPS.iterdir()): continue
        e = dict(os.environ, PYTHONPATH=str(VSRC), PYTHONUNBUFFERED="1", FL_W=str(FLW), FL_UM=str(um), OMP_NUM_THREADS="4")
        t = time.time()
        r = subprocess.run([PY, str(CODE / "fl" / "remote" / "spiral" / "window_score.py"), str(MAPS), "--um", str(um)],
                           env=e, stdout=open(W / f"window_score_{tag}.log", "w"), stderr=subprocess.STDOUT)
        note(stage="window_score", tag=tag, rc=r.returncode, secs=round(time.time() - t))
        r = subprocess.run([PY, str(FS / "glyph_win.py"), str(MAPS), "--tag", tag, "--um", str(um), "--model", str(CODE / "x_glyph" / "models" / "gA.pt")],
                           env=e, stdout=open(W / f"glyph_win_{tag}.log", "w"), stderr=subprocess.STDOUT)
        note(stage="glyph_win", tag=tag, rc=r.returncode)
        with open(W / f"tri_rank_{tag}.txt", "w") as f, open(W / f"tri_rank_{tag}.err", "w") as er:
            r = subprocess.run([PY, str(FS / "tri_rank.py"), str(MAPS), "--tag", tag, "--min", "-99", "--um", str(um), "--glyph",
                                "--json", str(W / f"tri_rank_{tag}.json")], env=e, stdout=f, stderr=er)
        note(stage="tri_rank", tag=tag, rc=r.returncode)
        r = subprocess.run([PY, str(FS / "line_find.py"), "--lane", tag, "--faces", "both", "--um", str(um), "--device", "cuda",
                            "--out", str(W / f"line_find_{tag}.json")], env=e, stdout=open(W / f"line_find_{tag}.log", "w"), stderr=subprocess.STDOUT)
        note(stage="line_find", tag=tag, rc=r.returncode)
        # glyph scores of every window (tri_rank only lists windows with a row score)
        allg = {}
        for gf in MAPS.glob("*/glyph_win.json"): allg[gf.parent.name] = json.load(open(gf))
        json.dump(allg, open(W / f"glyph_win_{tag}.json", "w"))


ANALYSIS = r'''
import glob, json, sys, pathlib, csv
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
W, T, WIN = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), 2600
groups = json.loads(sys.argv[3])
# claim frame of the 306B line: v = -21 + y / 115.741 mm, line near v = 0 -> rows 2430 +- 400; u -24..16 mm -> cols 1389..6018
CLAIM_ROWS, CLAIM_COLS = (2030, 2830), (1389, 6018)

def labels(tag, w, shape):
    H, Wd = shape; d = T / "labels" / f"{tag}_{w}"
    if tag == "c1447":
        ink = np.asarray(Image.open(d / "inklabels.png")); m = np.asarray(Image.open(d / "mask.png"))
        ink = ink[..., 0] if ink.ndim == 3 else ink; m = m[..., 0] if m.ndim == 3 else m
        L = np.zeros(shape, bool); S = np.zeros(shape, bool); h, w_ = min(H, ink.shape[0]), min(Wd, ink.shape[1])
        L[:h, :w_] = ink[:h, :w_] > 0; S[:h, :w_] = m[:h, :w_] > 0; return L, S
    if tag == "c0841":
        lab = np.load(d / "label.npy"); sup = np.load(d / "support.npy"); i0, i1, j0, j1 = json.load(open(d / "meta.json"))["crop"]
        L = np.zeros(shape, bool); S = np.zeros(shape, bool); h, w_ = min(i1, H) - i0, min(j1, Wd) - j0
        L[i0:i0 + h, j0:j0 + w_] = lab[:h, :w_] > 0; S[i0:i0 + h, j0:j0 + w_] = sup[:h, :w_] > 0; return L, S
    L = np.zeros(shape, bool); S = np.ones(shape, bool)
    L[CLAIM_ROWS[0]:CLAIM_ROWS[1], CLAIM_COLS[0]:CLAIM_COLS[1]] = True; return L, S

def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float); pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0: return None
    allv = np.concatenate([pos, neg]); r = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True); sums = np.bincount(inv, r); r = (sums / cnt)[inv]
    return round(float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)

def pixel_auc(z, L, S):
    v = S & (z != 0)
    if L[v].sum() == 0 or (~L[v]).sum() == 0: return None
    return auc(z[v & L][::7], z[v & ~L][::7])

rows = []
# nulls: band0800-ink tri_rank (every window with a row score, gates run), plus its glyph scores
neg_tr = [p for p in glob.glob("/kaggle/input/**/tri_rank.json", recursive=True) if "calib" not in p]
for r in json.load(open(neg_tr[0])):
    rows.append(dict(set="null_0800", surface=r["name"], side=r["side"], x0=r["x0"], label=0, ink_frac=0.0, **{k: r.get(k) for k in
                ("score", "row_off", "fib_off", "rel_tex", "crack", "shape", "glyph", "queue")}))
face = {}
for g in groups:
    tag = g["tag"]; trp = W / f"tri_rank_{tag}.json"
    if not trp.exists(): continue
    maps = T / "lanes" / tag / "maps"
    for r in json.load(open(trp)):
        w = r["name"].split("_", 1)[1]; zf = maps / r["name"] / f"{r['name']}_d0_{'reverse' if r['side'] == 'rev' else 'fwd'}_z.npy"
        z = np.load(zf).astype(np.float32); L, S = labels(tag, w, z.shape)
        key = (r["name"], r["side"])
        if key not in face: face[key] = pixel_auc(z, L, S)
        sl = (slice(None), slice(r["x0"], r["x0"] + WIN)); sup = (S[sl] & (z[sl] != 0)).sum()
        frac = float(L[sl][S[sl] & (z[sl] != 0)].mean()) if sup else 0.0
        rows.append(dict(set=tag, surface=r["name"], side=r["side"], x0=r["x0"], label=None, ink_frac=round(frac, 4), sup_px=int(sup),
                         **{k: r.get(k) for k in ("score", "row_off", "fib_off", "rel_tex", "crack", "shape", "glyph", "queue")}))
# text face per surface = the face whose V3 map has the higher pixel AUC against the labels
best_face = {}
for (n, s), a in face.items():
    if a is not None and (n not in best_face or a > best_face[n][1]): best_face[n] = (s, a)
for r in rows:
    if r["set"] == "null_0800": continue
    bf = best_face.get(r["surface"]); thr = 0.02 if r["set"] != "c306" else 0.05
    if bf is None or r["side"] != bf[0]: r["label"] = None          # other face: unknown, excluded
    elif r["ink_frac"] >= thr: r["label"] = 1
    elif r["ink_frac"] == 0 and r["set"] != "c306" and r.get("sup_px", 0) > 2e6: r["label"] = 0   # labelled blank papyrus
    else: r["label"] = None
keys = ["set", "surface", "side", "x0", "label", "ink_frac", "sup_px", "score", "row_off", "fib_off", "rel_tex", "crack", "shape", "glyph", "queue"]
with open(W / "windows.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)

num = lambda v: float(v) if isinstance(v, (int, float)) and v is not None else np.nan
pos = [r for r in rows if r["label"] == 1]; nul = [r for r in rows if r["set"] == "null_0800"]
res = dict(n_pos=len(pos), n_null=len(nul), pos_by_set={s: sum(1 for r in pos if r["set"] == s) for s in {r["set"] for r in pos}},
           text_face={n: dict(side=s, pixel_auc=a) for n, (s, a) in best_face.items()}, features={})
# direction: higher = more text-like for score, shape, glyph; lower for row_off, fib_off, crack
sign = dict(score=1, glyph=1, shape=1, rel_tex=1, row_off=-1, fib_off=-1, crack=-1)
for k, sg in sign.items():
    P = np.array([num(r[k]) for r in pos]) * sg; N = np.array([num(r[k]) for r in nul]) * sg
    a = auc(P, N); Pf, Nf = P[np.isfinite(P)], N[np.isfinite(N)]
    ent = dict(auc_vs_0800=a, n_pos=int(len(Pf)), n_null=int(len(Nf)))
    if len(Pf) and len(Nf):
        p99 = float(np.percentile(Nf, 99)); ent["null_p99"] = round(p99 * sg, 3); ent["pos_recall_at_null_p99"] = round(float((Pf > p99).mean()), 3)
        for q in (1.0, 0.9, 0.8):
            thr = float(np.quantile(Pf, 1 - q)) if q < 1 else float(Pf.min())
            ent[f"null_fa_at_recall_{int(q*100)}"] = round(float((Nf >= thr).mean()), 3); ent[f"thr_at_recall_{int(q*100)}"] = round(thr * sg, 3)
        ent["pos_median"] = round(float(np.median(Pf)) * sg, 3); ent["null_median"] = round(float(np.median(Nf)) * sg, 3)
    res["features"][k] = ent
# the shipped rule (306B defaults): admit if score >= 4 or glyph >= 9.06; then hard and soft gates
def shipped(r):
    s, gl = num(r["score"]), num(r["glyph"]); return (s >= 4) or (gl >= 9.06)
res["shipped_admit"] = dict(pos=round(float(np.mean([shipped(r) for r in pos])), 3) if pos else None,
                            null=round(float(np.mean([shipped(r) for r in nul])), 3))
res["shipped_review"] = dict(pos=round(float(np.mean([str(r["queue"]).startswith("REVIEW") and not str(r["queue"]).startswith("REVIEW-2") for r in pos])), 3) if pos else None,
                             null=round(float(np.mean([str(r["queue"]) == "REVIEW" for r in nul])), 3))
# line_find: best chain per surface/face, and whether it lies on labelled ink
lf = {}
neg_lf = [p for p in glob.glob("/kaggle/input/**/line_find.json", recursive=True) if "calib" not in p]
def chains(path):
    d = json.load(open(path)); out = []
    for m in d.get("maps", []):
        for c in m.get("candidates", []): out.append((m["name"], m["side"], c["score"], c.get("poly"), c["feats"].get("glyph_p")))
    return out
null_ch = chains(neg_lf[0]) if neg_lf else []
lf["null_max"] = max((c[2] for c in null_ch), default=None); lf["null_p99"] = round(float(np.percentile([c[2] for c in null_ch], 99)), 3) if null_ch else None
for g in groups:
    p = W / f"line_find_{g['tag']}.json"
    if not p.exists(): continue
    for n, side, sc, poly, gp in chains(p):
        w = n.split("_", 1)[1]; on = None
        if poly:
            pts = np.array(poly); cy, cx = int(pts[:, 1].mean()), int(pts[:, 0].mean())
            maps = T / "lanes" / g["tag"] / "maps"; zf = maps / n / f"{n}_d0_{'reverse' if side == 'rev' else 'fwd'}_z.npy"
            shp = np.load(zf, mmap_mode="r").shape; L, S = labels(g["tag"], w, shp)
            y0, y1, x0, x1 = max(0, cy - 60), cy + 60, max(0, cx - 400), cx + 400
            on = bool(L[y0:y1, x0:x1].any())
        e = lf.setdefault(f"{n}:{side}", dict(best=None, best_on_text=None, n=0, text_face=best_face.get(n, (None,))[0] == side))
        e["n"] += 1
        if e["best"] is None or sc > e["best"]: e["best"] = round(sc, 3)
        if on and (e["best_on_text"] is None or sc > e["best_on_text"]): e["best_on_text"] = round(sc, 3)
res["line_find"] = lf
json.dump(res, open(W / "calib.json", "w"), indent=1)
L = ["# Reader V3 calibration of the 306B triage", "", f"positives: {res['n_pos']} text windows {res['pos_by_set']}; nulls: {res['n_null']} PHerc. 0800 windows", "",
     "| feature | AUC vs 0800 nulls | null p99 | text recall above null p99 | null false alarms at 100 % text recall | at 90 % |", "|---|---|---|---|---|---|"]
for k, e in res["features"].items():
    L.append(f"| {k} | {e.get('auc_vs_0800')} | {e.get('null_p99')} | {e.get('pos_recall_at_null_p99')} | {e.get('null_fa_at_recall_100')} | {e.get('null_fa_at_recall_90')} |")
L += ["", f"shipped admission (score >= 4 or glyph >= 9.06): text {res['shipped_admit']['pos']}, nulls {res['shipped_admit']['null']}",
      f"shipped REVIEW queue: text {res['shipped_review']['pos']}, nulls {res['shipped_review']['null']}", "",
      f"line_find nulls: max {lf['null_max']}, p99 {lf['null_p99']}", ""]
for k, e in lf.items():
    if isinstance(e, dict): L.append(f"- {k}: best {e['best']}, best on text {e['best_on_text']}, chains {e['n']}, text face {e['text_face']}")
L += ["", "text face per surface (pixel AUC of V3 d0 against the labels):"] + [f"- {n}: {v['side']} {v['pixel_auc']}" for n, v in res["text_face"].items()]
(W / "report.md").write_text("\n".join(L) + "\n"); print("\n".join(L), flush=True)
'''


def stage_analysis():
    f = T / "analysis.py"; f.write_text(ANALYSIS)
    r = subprocess.run([PY, str(f), str(W), str(T), json.dumps(GROUPS)], env=dict(os.environ, PYTHONUNBUFFERED="1"),
                       stdout=open(W / "analysis.log", "w"), stderr=subprocess.STDOUT)
    note(stage="analysis", rc=r.returncode)
    print(open(W / "analysis.log").read()[-6000:], flush=True)


def stage_pack():
    with tarfile.open(W / "maps_calib.tar", "w") as tar:
        for g in GROUPS:
            L, MAPS, REN, CT = lane(g["tag"])
            if not MAPS.exists(): continue
            for f in sorted(MAPS.rglob("*")):
                if f.is_file() and (f.suffix in (".png", ".json") or f.name.endswith("_z.npy")):
                    tar.add(f, arcname=f"maps/{f.relative_to(MAPS)}")
    note(stage="pack", tar_mb=round((W / "maps_calib.tar").stat().st_size / 1e6, 1))


if __name__ == "__main__":
    t = time.time(); stage_env(); stage_fetch(); stage_render(); stage_ink(); stage_triage(); stage_analysis(); stage_pack()
    note(stage="done", secs=round(time.time() - t))
