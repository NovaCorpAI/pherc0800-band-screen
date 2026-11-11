"""Render a tifxyz mesh into a flat surface volume straight from a scroll volume on S3.

    python render.py MESH_DIR VOLUME_URL OUT.zarr [--layers 21] [--tile 512] [--flip-normals]

Geometry is villa's own (tifxyz_label_transfer: bilinear upsampling of the stored grid to the
full-resolution canvas, normals from +-1 stored-grid differences, as vc_render_tifxyz does).
Layer i samples the volume trilinearly at xyz + (i - (L-1)/2) * normal, one voxel apart.
Only the 128^3 chunks the surface band touches are fetched (uncompressed zarr v2 chunks read
as raw bytes, many in parallel) and kept in an LRU cache shared by neighbouring tiles.

Output: OUT.zarr, a (L, H, W) uint8 array (chunks L x 128 x 128); pixels off the mesh are 0.
"""
from __future__ import annotations

import argparse, json, math, sys, threading, time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates


def _villa_imports():
    from vesuvius.tifxyz_canvas import full_resolution_extent
    from vesuvius.tifxyz_label_transfer.io import load_surface
    from vesuvius.tifxyz_label_transfer.self_render_tifxyz import surface_tile_geometry
    return full_resolution_extent, load_surface, surface_tile_geometry


class ChunkCache:
    """LRU cache of raw uint8 chunks of one zarr v2 level (no compressor)."""

    def __init__(self, url: str, level: str = "0", max_gb: float = 8.0, workers: int = 48):
        import fsspec
        self.fs, root = fsspec.core.url_to_fs(url, anon=True) if url.startswith("s3://") else fsspec.core.url_to_fs(url)
        self.root = root.rstrip("/") + "/" + level
        meta = json.loads(self.fs.cat(self.root + "/.zarray"))
        if meta.get("compressor") not in (None,) or meta["dtype"] not in ("|u1", "uint8"):
            raise SystemExit(f"unsupported volume level: compressor {meta.get('compressor')} dtype {meta['dtype']}")
        self.shape = tuple(meta["shape"]); self.chunks = tuple(meta["chunks"])
        self.sep = meta.get("dimension_separator", ".")
        self.max_chunks = max(64, int(max_gb * 1e9 / np.prod(self.chunks)))
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(workers)
        self.bytes_read = 0

    def _fetch_batch(self, keys):
        """Fetch many chunks in one async s3fs call; missing chunks are the fill value 0."""
        paths = [self.root + "/" + self.sep.join(map(str, k)) for k in keys]
        out = {}
        todo = list(zip(keys, paths))
        for attempt in range(6):
            got = self.fs.cat([p for _, p in todo], on_error="return", batch_size=256)
            retry = []
            for k, p in todo:
                v = got.get(p)
                if isinstance(v, FileNotFoundError):
                    out[k] = np.zeros(self.chunks, np.uint8)
                elif isinstance(v, BaseException) or v is None:
                    retry.append((k, p))
                else:
                    self.bytes_read += len(v)
                    out[k] = np.frombuffer(v, np.uint8).reshape(self.chunks)
            if not retry:
                return out
            todo = retry
            time.sleep(1 + 2 * attempt)
        raise IOError(f"{len(todo)} chunks failed after retries, e.g. {todo[0][1]}")

    def get_many(self, keys):
        with self.lock:
            missing = [k for k in keys if k not in self.cache]
        if missing:
            got = self._fetch_batch(missing)
            with self.lock:
                self.cache.update(got)
        out = {}
        with self.lock:
            for k in keys:
                self.cache.move_to_end(k)
                out[k] = self.cache[k]
            while len(self.cache) > self.max_chunks:
                self.cache.popitem(last=False)
        return out


def sample_points(cache: ChunkCache, pts_zyx: np.ndarray) -> np.ndarray:
    """Trilinear samples (float32) of the volume at (N, 3) zyx points; outside = 0."""
    out = np.zeros(len(pts_zyx), np.float32)
    shp = np.array(cache.shape); ch = np.array(cache.chunks)
    inside = np.all((pts_zyx >= 0) & (pts_zyx <= shp - 1), axis=1)
    if not inside.any():
        return out
    p = pts_zyx[inside]
    lo = np.floor(p).astype(np.int64); hi = np.minimum(lo + 1, shp - 1)
    ck = np.unique(np.concatenate([lo // ch, hi // ch]), axis=0)
    c0 = ck.min(0); c1 = ck.max(0) + 1
    block = np.zeros(tuple((c1 - c0) * ch), np.uint8)
    for key, arr in cache.get_many([tuple(map(int, k)) for k in ck]).items():
        o = (np.array(key) - c0) * ch
        block[o[0]:o[0] + ch[0], o[1]:o[1] + ch[1], o[2]:o[2] + ch[2]] = arr
    local = (p - c0 * ch).T
    out[inside] = map_coordinates(block, local, order=1, mode="nearest", prefilter=False)
    return out


def render(mesh_dir: Path, volume_url: str, out_zarr: Path, layers: int = 21, tile: int = 512,
           flip_normals: bool = False, cache_gb: float = 8.0, log=print, roi=None) -> dict:
    import zarr
    full_extent, load_surface, tile_geometry = _villa_imports()
    surf = load_surface(mesh_dir)
    sy, sx = surf.scale_yx
    H = full_extent(surf.x.shape[0], sy); W = full_extent(surf.x.shape[1], sx)
    if surf.valid is None:
        surf.valid = np.isfinite(surf.x) & (surf.x != -1) & (surf.z != -1)
    cache = ChunkCache(volume_url, max_gb=cache_gb)
    offsets = (np.arange(layers) - (layers - 1) / 2.0) * (-1.0 if flip_normals else 1.0)
    z = zarr.open_array(str(out_zarr), mode="w", shape=(layers, H, W), chunks=(layers, 128, 128),
                        dtype="uint8", fill_value=0)
    t0 = time.time(); n_valid = 0; done = 0
    R0, R1, C0, C1 = roi if roi else (0, H, 0, W)
    tiles = [(r, c) for r in range(R0, R1, tile) for c in range(C0, C1, tile)]
    for r, c in tiles:
        r1, c1 = min(r + tile, R1), min(c + tile, C1)
        # cheap skip: no valid stored node near this tile
        gy0 = max(0, int(r * surf.x.shape[0] / H) - 1); gy1 = int(math.ceil(r1 * surf.x.shape[0] / H)) + 1
        gx0 = max(0, int(c * surf.x.shape[1] / W) - 1); gx1 = int(math.ceil(c1 * surf.x.shape[1] / W)) + 1
        if not surf.valid[gy0:gy1, gx0:gx1].any():
            continue
        xyz, nrm, valid = tile_geometry(surf, (H, W), (r, r1, c, c1))
        if not valid.any():
            continue
        P = xyz[valid][:, ::-1]; N = nrm[valid][:, ::-1]          # xyz -> zyx
        stack = np.zeros((layers, r1 - r, c1 - c), np.uint8)
        for i, off in enumerate(offsets):
            v = sample_points(cache, P + N * off)
            plane = np.zeros(valid.shape, np.uint8)
            plane[valid] = np.clip(np.rint(v), 0, 255).astype(np.uint8)
            stack[i] = plane
        z[:, r:r1, c:c1] = stack
        n_valid += int(valid.sum()); done += 1
        if done % 20 == 0:
            log(f"  render {mesh_dir.name}: {done} tiles, {n_valid / 1e6:.1f} Mpx, "
                f"{cache.bytes_read / 1e9:.2f} GB read, {time.time() - t0:.0f}s")
    info = dict(mesh=str(mesh_dir), volume=volume_url, shape=[layers, H, W], valid_px=n_valid,
                flip_normals=flip_normals, gb_read=round(cache.bytes_read / 1e9, 3),
                secs=round(time.time() - t0, 1))
    np.save(out_zarr.with_suffix(".valid.npy"), np.asarray(z[layers // 2]) > 0)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", type=Path); ap.add_argument("volume"); ap.add_argument("out", type=Path)
    ap.add_argument("--layers", type=int, default=21); ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--flip-normals", action="store_true"); ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--roi", nargs=4, type=int, help="row0 row1 col0 col1 of the canvas (testing)")
    a = ap.parse_args()
    info = render(a.mesh, a.volume, a.out, a.layers, a.tile, a.flip_normals, a.cache_gb, roi=a.roi)
    print(json.dumps(info))


if __name__ == "__main__":
    sys.exit(main())
