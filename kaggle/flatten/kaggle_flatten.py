"""Kaggle GPU kernel: lasagna flatten (villa a329895, configs/flatten_fast_nofilter.json) of the snapped windings of one band.

Input: a Kaggle dataset holding results.tar from the pod lane (results/snap/<w>/{x,y,z}.tif + meta.json, results/windings.txt).
Output: /kaggle/working/flat.tar with flat/<w>/tifxyz/flatten.tifxyz (+ per-winding logs) and run_log.jsonl.
Two windings run at a time on a T4 x2 (one per GPU). Each winding has a 3600 s timeout (306B's lane_box value).
"""
import glob, json, os, pathlib, shutil, subprocess, sys, tarfile, time

VILLA_COMMIT = "a329895ea5e9bde78752e60dbc0a4878dd2ab909"
T = pathlib.Path("/kaggle/temp"); W = pathlib.Path("/kaggle/working"); T.mkdir(parents=True, exist_ok=True)
VENV = T / "venv"; PY = str(VENV / "bin" / "python")
VILLA = T / "villa"; VSRC = VILLA / "vesuvius" / "src"; LAS = VILLA / "lasagna"
LOG = W / "run_log.jsonl"
RES = T / "results"; FLAT = T / "flat"
MAXW = int(os.environ.get("MAXW", "0"))          # 0 = all windings in windings.txt
TIMEOUT = int(os.environ.get("FLAT_TIMEOUT", "3600"))
DEPS = ("numpy scipy tifffile zarr==3.4.1 pillow scikit-image tensorstore imagecodecs requests pynrrd numcodecs "
        "opencv-python-headless einops fsspec aiohttp safetensors timm")


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    t = time.time(); subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw); return time.time() - t


def note(**rec):
    rec["t"] = time.strftime("%Y-%m-%dT%H:%M:%S"); print("NOTE", json.dumps(rec), flush=True)
    with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")


def gpu_count():
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    return max(1, out.count("GPU "))


def stage_env():
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv || true")
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    sh(f"uv venv -q -p 3.14 {VENV}")
    sh(f"uv pip install -q -p {PY} {DEPS}")
    sh(f"uv pip install -q -p {PY} torch --index-url https://download.pytorch.org/whl/cu126")
    if not LAS.exists():
        sh(f"git clone -q --filter=blob:none --no-checkout https://github.com/ScrollPrize/villa.git {VILLA}")
        sh(f"git -C {VILLA} sparse-checkout set vesuvius/src lasagna")
        sh(f"git -C {VILLA} checkout -q {VILLA_COMMIT}")
    out = subprocess.run([PY, "fit.py", "--help"], cwd=LAS, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=f"{VSRC}:{LAS}"))
    note(stage="env", rc=out.returncode, err=out.stderr[-1500:], gpus=gpu_count())
    if out.returncode:
        raise SystemExit("lasagna fit.py --help failed")


def stage_inputs():
    global RES
    found = glob.glob("/kaggle/input/**/results.tar", recursive=True)
    if found and not RES.exists():
        sh(f"tar -xf {found[0]} -C {T}")
    elif not found:   # Kaggle extracts archives on upload: the dataset then holds results/snap/<w>/... directly
        wt = glob.glob("/kaggle/input/**/windings.txt", recursive=True)
        if not wt:
            sh("find /kaggle/input -maxdepth 4 | head -50")
            raise SystemExit("neither results.tar nor windings.txt found under /kaggle/input (dataset not attached?)")
        RES = pathlib.Path(wt[0]).parent
    ws = [l.strip() for l in open(RES / "windings.txt") if l.strip()]
    if MAXW: ws = ws[:MAXW]
    have = [w for w in ws if (RES / "snap" / w / "x.tif").exists()]
    note(stage="inputs", planned=len(ws), with_snap=len(have), missing=[w for w in ws if w not in have])
    return have


def one(w, gpu):
    o = FLAT / w; o.mkdir(parents=True, exist_ok=True)
    f = o / "tifxyz" / "flatten.tifxyz" / "x.tif"
    if f.exists(): return None
    cfg = o / "input.json"; cfg.write_text(json.dumps({"external_surfaces": [{"path": str(RES / "snap" / w)}]}))
    log = open(o / "flat.log", "w")
    e = dict(os.environ, PYTHONPATH=f"{VSRC}:{LAS}", CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    return subprocess.Popen(["timeout", str(TIMEOUT), PY, "fit.py", "configs/flatten_fast_nofilter.json", str(cfg),
                             "--out-dir", str(o), "--device", "cuda"], cwd=LAS, env=e, stdout=log, stderr=subprocess.STDOUT)


def stage_flatten(ws):
    n = gpu_count(); running = {}; queue = list(ws); done = []
    while queue or running:
        for gpu in range(n):
            if gpu not in running and queue:
                w = queue.pop(0); p = one(w, gpu)
                if p is None: done.append((w, 0, 0)); continue
                running[gpu] = (w, p, time.time())
        for gpu, (w, p, t0) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                ok = (FLAT / w / "tifxyz" / "flatten.tifxyz" / "x.tif").exists()
                note(stage="flatten", winding=w, rc=rc, ok=ok, secs=round(time.time() - t0))
                if not ok: print(open(FLAT / w / "flat.log").read()[-2000:], flush=True)
                done.append((w, rc, time.time() - t0)); del running[gpu]
        time.sleep(10)
    return done


def stage_pack(done):
    with tarfile.open(W / "flat.tar", "w") as tar:
        for w, rc, secs in done:
            d = FLAT / w / "tifxyz"
            if d.exists(): tar.add(d, arcname=f"flat/{w}/tifxyz")
            lg = FLAT / w / "flat.log"
            if lg.exists(): tar.add(lg, arcname=f"flat/{w}/flat.log")
        tar.add(RES / "windings.txt", arcname="flat/windings.txt")
    for w in FLAT.iterdir():   # drop model checkpoints etc. from temp, nothing else to keep
        pass
    ok = sum(1 for w, rc, s in done if (FLAT / w / "tifxyz" / "flatten.tifxyz" / "x.tif").exists())
    note(stage="pack", windings=len(done), flattened=ok, tar_mb=round((W / "flat.tar").stat().st_size / 1e6, 1))


if __name__ == "__main__":
    t = time.time(); stage_env()
    ws = stage_inputs()
    done = stage_flatten(ws)
    stage_pack(done)
    note(stage="done", secs=round(time.time() - t))
