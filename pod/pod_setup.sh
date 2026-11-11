#!/bin/bash
# pod_setup.sh : fit environment on the RunPod box (image runpod/pytorch:*-cu1281-*-ubuntu2404, run as root).
# Adapted from 306B's fl/remote/setup.sh + x_recipe_erwin/remote/erwin_setup.sh (MIT): only what the fit + snap need.
#   $W/envs/spiral   python 3.14, torch cu128 (cu126 fallback), villa 0fb38c45 scripts/spiral (editable) + the two patches,
#                    awscli, nanobind; also used for snap_meshes.py and the umbilicus tools
#   $PKG/native_pkg  vc.track_crossings / track_store / spiral_sampling built from the villa sources (native/build.sh)
#   $W/vc3d          VC3D AppImage (pinned c9f8c57 2026-11-02) extracted: only vc_gen_umbilicus is used here
set -uo pipefail
W=${W:-/workspace/w}; PKG=$W/lane0800; SP=$PKG/spiral_old; SPY=$W/envs/spiral/bin/python
APPIMG=https://github.com/ScrollPrize/villa/releases/download/vc3d-archive-latest-432302465/VC3D-c9f8c57-2026-11-02-linux-x86_64.AppImage
export DEBIAN_FRONTEND=noninteractive PATH=$HOME/.local/bin:$PATH UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600
mkdir -p $W/envs $W/vc3d $W/bin $W/logs
(apt-get update -qq && apt-get install -y -qq --no-install-recommends curl git rsync file procps build-essential cmake ninja-build \
   libgomp1 libgl1 libglib2.0-0 libglx0 libegl1 libopengl0 libxkbcommon0 libfontconfig1 libfreetype6 libxcb-cursor0 libxcb-icccm4 \
   libxcb-keysyms1 libxcb-shape0 libxcb-xkb1 libxkbcommon-x11-0 libdbus-1-3 libx11-6 libxcb1 >/dev/null) || true
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null)
uv python install -q 3.14 || true
[ -x $SPY ] || uv venv -q -p 3.14 $W/envs/spiral || exit 1
if ! $SPY -c "import torch" 2>/dev/null; then
  uv pip install -q -p $SPY 'torch>=2.11,<2.13' --index-url https://download.pytorch.org/whl/cu128 \
    || uv pip install -q -p $SPY 'torch>=2.11,<2.13' --index-url https://download.pytorch.org/whl/cu126 || exit 1
fi
(cd $SP && uv pip install -q -p $SPY -e .) || exit 1
uv pip install -q -p $SPY awscli nanobind || exit 1
# patch 1 (343, Erwin): run without an outer_shell/ folder
$SPY - "$SP/fit_spiral.py" <<'PY' || exit 1
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = "filter_tracks_by_shell = bool(self.tracks_dbm_path) and bool(self.shell_path)\n"
new = "filter_tracks_by_shell = bool(self.tracks_dbm_path) and bool(self.shell_path) and os.path.isdir(self.shell_path)  # flpipe: shell optional\n"
if 'flpipe: shell optional' in s: print('shell-optional already patched')
elif s.count(old) != 1: sys.exit(f'shell-optional anchor found {s.count(old)} times')
else: p.write_text(s.replace(old, new)); print('fit_spiral.py patched: shell optional')
PY
# patch 2 (306B): FIT_SPIRAL_STOP_AT for the short sense-test fits
$SPY $PKG/patch_fitter.py $SP || exit 1
# native modules
ls $PKG/native_pkg/vc/track_store*.so >/dev/null 2>&1 || bash $PKG/native/build.sh || exit 1
# VC3D AppImage: vc_gen_umbilicus for the scroll axis
if [ ! -x $W/vc3d/squashfs-root/AppRun ]; then
  (cd $W/vc3d && curl -fsSL -o VC3D.AppImage $APPIMG && chmod +x VC3D.AppImage && ./VC3D.AppImage --appimage-extract >/dev/null) || exit 1
fi
printf '#!/bin/bash\nexport OPENBLAS_NUM_THREADS=1\nexec %s/vc3d/squashfs-root/AppRun vc_gen_umbilicus "$@"\n' $W > $W/bin/vc_gen_umbilicus
chmod +x $W/bin/vc_gen_umbilicus
$W/bin/vc_gen_umbilicus --help 2>&1 | head -3
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nproc; free -g | head -2; df -h $W | tail -1
(cd $SP && PYTHONPATH=$PKG/native_pkg $SPY -c "import torch, fit_spiral, vc.track_crossings, vc.track_store, vc.spiral_sampling
print('spiral ok torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))") || exit 1
$SPY -c "import zarr, tifffile, scipy, numpy; print('snap deps ok zarr', zarr.__version__)" || exit 1
aws --version || exit 1
echo SETUP_OK
