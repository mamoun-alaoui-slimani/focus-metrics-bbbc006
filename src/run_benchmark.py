"""Benchmark focus metrics on BBBC006 (U2OS cells, Hoechst + phalloidin).

Usage: python run_benchmark.py <dir containing BBBC006_v1_images_z_XX folders> <out_dir>

Ground truth: BBBC006 acquires the optimal focal plane at z = 16, with 2 um
between planes. A metric "hits" a field if its maximum over the available
planes is at z = 16.
"""
import sys, os, re, glob, time, zlib
import numpy as np
import pandas as pd
import tifffile
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from focus_metrics import METRICS

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
ZS = sorted(int(d[-2:]) for d in os.listdir(ROOT) if re.match(r"BBBC006_v1_images_z_\d\d$", d))
BEST = 16
DZ_UM = 2.0
NOISE = [0.0, 0.05, 0.15]  # Gaussian read noise, as a fraction of the in-focus image std
RNG_SEED = 0
PAT = re.compile(r"_(\w\d\d)_(s\d)_(w\d)")

def index_folder(z):
    idx = {}
    for f in glob.glob(os.path.join(ROOT, f"BBBC006_v1_images_z_{z:02d}", "*.tif")):
        m = PAT.search(os.path.basename(f))
        if m: idx[m.groups()] = f
    return idx

def process(key, paths):
    well, site, ch = key
    rng = np.random.default_rng(zlib.crc32("_".join(key).encode()) + RNG_SEED)
    stack = np.stack([tifffile.imread(p).astype(np.float64) for p in paths])
    sigma_ref = stack[ZS.index(BEST)].std()
    rows = []
    for nl in NOISE:
        noisy = stack + (rng.normal(0, nl * sigma_ref, stack.shape) if nl > 0 else 0)
        for name, fn in METRICS.items():
            t0 = time.perf_counter()
            vals = [fn(im) for im in noisy]
            dt = (time.perf_counter() - t0) / len(vals)
            rows.append(dict(well=well, site=site, channel=ch, noise=nl, metric=name,
                             z_pred=ZS[int(np.argmax(vals))], ms_per_image=dt * 1e3,
                             mean_intensity=float(stack[ZS.index(BEST)].mean()),
                             **{f"v_z{z:02d}": v for z, v in zip(ZS, vals, strict=True)}))
    return rows

if __name__ == "__main__":
    idx = {z: index_folder(z) for z in ZS}
    keys = sorted(set.intersection(*[set(v) for v in idx.values()]))
    print(f"planes {ZS}, fields x channels: {len(keys)}", flush=True)
    # optional chunking: run_benchmark.py ROOT OUT START END (results merged afterwards)
    start, end = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, len(keys))
    keys = keys[start:end]
    rows = []
    with ProcessPoolExecutor() as ex:
        futs = [ex.submit(process, k, [idx[z][k] for z in ZS]) for k in keys]
        for i, f in enumerate(futs):
            rows += f.result()
            if i % 100 == 0: print(i, flush=True)
    df = pd.DataFrame(rows)
    name = "results.csv" if (start, end) == (0, len(idx[ZS[0]])) or len(sys.argv) <= 4 else f"results_{start:04d}.csv"
    df.to_csv(os.path.join(OUT, name), index=False)
    print("done", len(df), flush=True)
