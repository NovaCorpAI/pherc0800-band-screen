#!/bin/bash
# build.sh : -> $PKG/native_pkg/vc/{__init__.py, track_crossings*.so, track_store*.so, spiral_sampling*.so}
# (306B's x_recipe_erwin/native/build.sh with the paths taken from the environment)
set -euo pipefail
W=${W:-/workspace/w}; PKG=$W/lane0800; X=$PKG/native; PY=${SPIRAL_PY:-$W/envs/spiral/bin/python}
export PATH=$HOME/.local/bin:$PATH
uv pip install -q -p $PY nanobind
mkdir -p $PKG/native_pkg/vc && echo '"""old volume-cartographer native spiral modules (standalone build)"""' > $PKG/native_pkg/vc/__init__.py
cmake -S $X -B $X/build -G Ninja -DPython_EXECUTABLE=$PY -DPKG_OUT=$PKG/native_pkg > $X/cmake.log 2>&1 || { tail -30 $X/cmake.log; exit 1; }
cmake --build $X/build -j $(nproc) > $X/build.log 2>&1 || { tail -30 $X/build.log; exit 1; }
cd $PKG && PYTHONPATH=$PKG/native_pkg $PY -c "import vc.track_crossings, vc.track_store, vc.spiral_sampling; print('native ok')"
