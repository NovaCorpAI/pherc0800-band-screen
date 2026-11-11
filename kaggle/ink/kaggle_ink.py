"""Kaggle GPU kernel: Reader V3 ink maps of the rendered windings of one band, written in 306B's lane format, then 306B's
triage (window_score, glyph_win, tri_rank with --min -99 --flat, line_find).

Inputs: kernel outputs band0800-render-0..3 (RENDERS/<w>.zarr, 31 layers, our render.py), dataset band0800-flat
(flat/<w>/tifxyz/flatten.tifxyz, for the recur gate), dataset band0800-code (render.py, fl/, x_glyph/, requirements.txt).
Per winding (name = TAG_<w>): MAPS/<name>/<name>_d0_{fwd,reverse}_z.npy (fused z, float16, 48-px border blanked),
*_overlay.png, <name>__v3_d0*.png, quality_tiles.npy, <name>_timing.json, done_d0; plus the same at d=+3 (_d3_).
renders/<name>.zarr/0 -> symlink to the input render (zarr v3 group layout the triage scripts expect).
Output (/kaggle/working): maps.tar (MAPS without the d3 z arrays), tri_rank.json/.txt, line_find.json, run_log.jsonl.
"""
import glob, json, os, pathlib, shutil, subprocess, sys, tarfile, time

TAG = os.environ.get("TAG", "b0800"); UM = 8.64; OFFS = [0, 3]; MAXW = int(os.environ.get("MAXW", "0"))
VILLA_COMMIT = "f7756c758"; V3_URL = "https://huggingface.co/domenicor046/reader-v3/resolve/main/reader-v3.pth"
T = pathlib.Path("/kaggle/temp"); W = pathlib.Path("/kaggle/working"); T.mkdir(parents=True, exist_ok=True)
VENV = T / "venv"; PY = str(VENV / "bin" / "python"); VILLA = T / "villa"; VSRC = VILLA / "vesuvius" / "src"
LANE = T / "lane"; MAPS = LANE / "maps"; REN = LANE / "renders"; CT = LANE / "ct"; CK = T / "reader-v3.pth"; LOG = W / "run_log.jsonl"
FLW = T / "flw"   # a vesuvius-work look-alike root for line_find.py --lane (x_vast_plan/results/<lane>/{maps,results/{lane.env,ct}}) and x_glyph
for d in (MAPS, REN, CT): d.mkdir(parents=True, exist_ok=True)


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
    out = subprocess.run([PY, "-c", "import vesuvius.ink_detection.inference.infer as m, cv2, zarr, torch; print(torch.__version__, torch.cuda.device_count())"],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(VSRC)))
    note(stage="env", out=out.stdout.strip(), err=out.stderr[-1200:], ckpt_mb=round(CK.stat().st_size / 1e6))
    if out.returncode: raise SystemExit("env check failed")


def inputs():
    rens = {}
    for z in glob.glob("/kaggle/input/**/RENDERS/*.zarr", recursive=True):
        w = pathlib.Path(z).name[:-5]
        if pathlib.Path(z).with_suffix(".valid.npy").exists() or (pathlib.Path(z).parent / f"{w}.valid.npy").exists():
            rens[w] = z
    FLAT = None
    tars = glob.glob("/kaggle/input/**/flat.tar", recursive=True)
    if tars:   # the flatten kernel's output (kernel_sources)
        if not (T / "flat" / "windings.txt").exists(): sh(f"tar -xf {tars[0]} -C {T}")
        FLAT = T / "flat"
    # else the flatten dataset: the dir holding windings.txt next to <w>/tifxyz/flatten.tifxyz (Kaggle may strip the flat/ level)
    for wt in glob.glob("/kaggle/input/**/windings.txt", recursive=True):
        p = pathlib.Path(wt).parent
        if glob.glob(str(p / "w*" / "tifxyz" / "flatten.tifxyz" / "x.tif")):
            FLAT = p; break
    ws = sorted(rens)
    if MAXW: ws = ws[:MAXW]
    for w in ws:   # renders/<TAG>_<w>.zarr/0 -> input array (zarr v3 group wrapper)
        g = REN / f"{TAG}_{w}.zarr"; g.mkdir(exist_ok=True)
        (g / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group"}))
        if not (g / "0").exists(): os.symlink(rens[w], g / "0")
    R = FLW / "x_vast_plan" / "results" / TAG; (R / "results").mkdir(parents=True, exist_ok=True)
    for link, target in ((R / "maps", MAPS), (R / "results" / "ct", CT), (FLW / "x_glyph", CODE / "x_glyph"), (FLW / "fl", CODE / "fl")):
        if not link.exists(): os.symlink(target, link)
    (R / "results" / "lane.env").write_text(f"SCROLL=PHerc0800\nUM={UM}\nTB={TAG}\n")
    note(stage="inputs", renders=len(rens), planned=ws, flat=str(FLAT))
    return ws, FLAT


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


def stage_ink(ws):
    worker = T / "ink_worker.py"; worker.write_text(WORKER)
    n = max(1, subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU "))
    procs = []
    for g in range(n):
        part = ws[g::n]
        if not part: continue
        log = open(W / f"ink_gpu{g}.log", "w")
        procs.append((g, subprocess.Popen([PY, str(worker), TAG, str(UM), ",".join(map(str, OFFS)), str(REN), str(MAPS), str(CK), ",".join(part)],
                                          env=dict(os.environ, PYTHONPATH=str(VSRC), CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2"),
                                          stdout=log, stderr=subprocess.STDOUT), time.time()))
    for g, p, t0 in procs:
        rc = p.wait(); done = sum(1 for w in ws[g::n] if (MAPS / f"{TAG}_{w}" / "done_d0").exists())
        note(stage="ink", gpu=g, rc=rc, done=done, planned=len(ws[g::n]), secs=round(time.time() - t0))
        if rc: print(open(W / f"ink_gpu{g}.log").read()[-3000:], flush=True)


def run_tool(args, log, env_extra=None, check=False):
    e = dict(os.environ, PYTHONPATH=str(VSRC), PYTHONUNBUFFERED="1", FL_W=str(FLW), FL_UM=str(UM), OMP_NUM_THREADS="4")
    if env_extra: e.update(env_extra)
    t = time.time()
    with open(W / log, "w") as f:
        r = subprocess.run([PY, *map(str, args)], env=e, stdout=f, stderr=subprocess.STDOUT)
    note(stage=pathlib.Path(args[0]).stem, rc=r.returncode, secs=round(time.time() - t))
    if r.returncode and check: raise SystemExit(f"{args[0]} failed")
    if r.returncode: print(open(W / log).read()[-2000:], flush=True)
    return r.returncode


def stage_triage(FLAT):
    FS = CODE / "fl" / "scripts"
    # window_score is CPU-bound (~5 min per winding): 4 roots of symlinked winding dirs, scored in parallel, results land in MAPS
    dirs = sorted(d for d in MAPS.iterdir() if d.is_dir()); n = 4; procs = []; t0 = time.time()
    for k in range(n):
        root = T / f"ws{k}"; (root / "maps").mkdir(parents=True, exist_ok=True)
        if not (root / "renders").exists(): os.symlink(REN, root / "renders")
        for d in dirs[k::n]:
            if not (root / "maps" / d.name).exists(): os.symlink(d, root / "maps" / d.name)
        e = dict(os.environ, PYTHONPATH=str(VSRC), PYTHONUNBUFFERED="1", FL_W=str(FLW), FL_UM=str(UM), OMP_NUM_THREADS="1")
        procs.append(subprocess.Popen([PY, str(CODE / "fl" / "remote" / "spiral" / "window_score.py"), str(root / "maps"), "--um", str(UM)],
                                      env=e, stdout=open(W / f"window_score_{k}.log", "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    note(stage="window_score", rcs=rcs, secs=round(time.time() - t0), scored=len(glob.glob(str(MAPS / "*" / "spectral_win.json"))))
    run_tool([FS / "glyph_win.py", MAPS, "--tag", TAG, "--um", UM, "--model", CODE / "x_glyph" / "models" / "gA.pt"], "glyph_win.log")
    tri = [FS / "tri_rank.py", MAPS, "--tag", TAG, "--min", "-99", "--um", UM, "--glyph", "--json", W / "tri_rank.json"]
    if FLAT: tri += ["--flat", FLAT]
    e = dict(os.environ, PYTHONPATH=str(VSRC), FL_W=str(FLW), FL_UM=str(UM), OMP_NUM_THREADS="4")
    with open(W / "tri_rank.txt", "w") as f, open(W / "tri_rank.err", "w") as g:
        r = subprocess.run([PY, *map(str, tri)], env=e, stdout=f, stderr=g)
    note(stage="tri_rank", rc=r.returncode)
    if r.returncode: print(open(W / "tri_rank.err").read()[-2000:], flush=True)
    run_tool([FS / "line_find.py", "--lane", TAG, "--faces", "both", "--um", UM, "--device", "cuda", "--out", W / "line_find.json"], "line_find.log")


def stage_pack(ws):
    with tarfile.open(W / "maps.tar", "w") as tar:
        for w in ws:
            d = MAPS / f"{TAG}_{w}"
            if not d.exists(): continue
            for f in sorted(d.iterdir()):
                if "_d3_" in f.name and f.suffix == ".npy": continue
                tar.add(f, arcname=f"maps/{d.name}/{f.name}")
    note(stage="pack", tar_mb=round((W / "maps.tar").stat().st_size / 1e6, 1))


if __name__ == "__main__":
    t = time.time(); stage_env(); ws, FLAT = inputs(); stage_ink(ws); stage_triage(FLAT); stage_pack(ws)
    note(stage="done", secs=round(time.time() - t))
