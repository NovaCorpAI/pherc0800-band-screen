"""Kaggle CPU kernel: 31-layer surface volumes of the flattened windings of one band (our hunt/render.py, villa geometry,
CT level 0 streamed from the open-data bucket by chunks). PART selects windings [PART*PER, (PART+1)*PER) of flat/windings.txt.

Inputs (datasets): band0800-flat (flat/<w>/tifxyz/flatten.tifxyz + flat/windings.txt, as the flatten kernel packs them,
extracted by Kaggle) and band0800-code (render.py). Output: /kaggle/working/RENDERS/<w>.zarr (31, H, W) uint8, zstd,
<w>.valid.npy, render_log.jsonl. Three windings render at a time (4 cores, 4 GB chunk cache each).
"""
import glob, json, os, pathlib, subprocess, sys, time

PART = int(os.environ.get("PART", "__PART__")); PER = int(os.environ.get("PER", "10")); MAXW = int(os.environ.get("MAXW", "0"))
PAR = int(os.environ.get("PAR", "3")); LAYERS = 31
VILLA_COMMIT = "f7756c758"
VOL = "s3://vesuvius-challenge-open-data/PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr"
T = pathlib.Path("/kaggle/temp"); W = pathlib.Path("/kaggle/working"); T.mkdir(parents=True, exist_ok=True)
VENV = T / "venv"; PY = str(VENV / "bin" / "python"); VILLA = T / "villa"; VSRC = VILLA / "vesuvius" / "src"
OUT = W / "RENDERS"; OUT.mkdir(exist_ok=True); LOG = W / "render_log.jsonl"


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def note(**rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S"); print("NOTE", json.dumps(rec), flush=True)
    with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")


def stage_env():
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh(f"uv venv -q -p 3.14 {VENV}")
    sh(f"uv pip install -q -p {PY} numpy scipy zarr==3.4.1 numcodecs tifffile fsspec s3fs aiohttp requests imagecodecs pillow scikit-image pynrrd tqdm pyyaml einops huggingface_hub")
    sh(f"uv pip install -q -p {PY} torch --index-url https://download.pytorch.org/whl/cpu")
    if not VSRC.exists():
        sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git {VILLA}")
        sh(f"git -C {VILLA} sparse-checkout set vesuvius/src")
        sh(f"git -C {VILLA} checkout -q {VILLA_COMMIT}")
    out = subprocess.run([PY, "-c", "from vesuvius.tifxyz_label_transfer.io import load_surface; import s3fs, zarr; print('ok', zarr.__version__)"],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(VSRC)))
    note(stage="env", out=out.stdout.strip(), err=out.stderr[-1000:], cores=os.cpu_count())
    if out.returncode: raise SystemExit("env check failed")


def inputs():
    tars = glob.glob("/kaggle/input/**/flat.tar", recursive=True)
    if tars:   # the flatten kernel's output (kernel_sources): extract once to /kaggle/temp/flat
        if not (T / "flat" / "windings.txt").exists(): sh(f"tar -xf {tars[0]} -C {T}")
        flat = T / "flat"
    else:
        wt = glob.glob("/kaggle/input/**/flat/windings.txt", recursive=True) or glob.glob("/kaggle/input/**/windings.txt", recursive=True)
        if not wt:
            sh("find /kaggle/input -maxdepth 4 | head -40"); raise SystemExit("windings.txt not found")
        flat = pathlib.Path(wt[0]).parent
    render_py = glob.glob("/kaggle/input/**/render.py", recursive=True)
    if not render_py: raise SystemExit("render.py not found (band0800-code dataset)")
    ws = [l.strip() for l in open(flat / "windings.txt") if l.strip()]
    ws = ws[PART * PER:(PART + 1) * PER]
    if MAXW: ws = ws[:MAXW]
    have = [w for w in ws if (flat / w / "tifxyz" / "flatten.tifxyz" / "x.tif").exists()]
    note(stage="inputs", part=PART, planned=ws, with_flat=len(have))
    return flat, pathlib.Path(render_py[0]), have


def one(flat, render_py, w):
    o = OUT / f"{w}.zarr"
    if (OUT / f"{w}.valid.npy").exists(): return None
    log = open(W / f"render_{w}.log", "w")
    return subprocess.Popen([PY, str(render_py), str(flat / w / "tifxyz" / "flatten.tifxyz"), VOL, str(o),
                             "--layers", str(LAYERS), "--cache-gb", "4"],
                            env=dict(os.environ, PYTHONPATH=str(VSRC), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1"),
                            stdout=log, stderr=subprocess.STDOUT)


def main():
    t = time.time(); stage_env(); flat, render_py, ws = inputs()
    queue = list(ws); running = {}; done = []
    while queue or running:
        while len(running) < PAR and queue:
            w = queue.pop(0); p = one(flat, render_py, w)
            if p is None: done.append(w); continue
            running[w] = (p, time.time())
        for w, (p, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                ok = (OUT / f"{w}.valid.npy").exists()
                info = {}
                try:
                    last = [l for l in open(W / f"render_{w}.log") if l.startswith("{")]
                    info = json.loads(last[-1]) if last else {}
                except Exception: pass
                note(stage="render", winding=w, rc=rc, ok=ok, secs=round(time.time() - t0), valid_px=info.get("valid_px"), gb_read=info.get("gb_read"))
                if not ok: print(open(W / f"render_{w}.log").read()[-1500:], flush=True)
                done.append(w); del running[w]
        time.sleep(10)
    sh(f"du -sh {OUT}")
    note(stage="done", part=PART, windings=len(done), secs=round(time.time() - t))


if __name__ == "__main__":
    main()
