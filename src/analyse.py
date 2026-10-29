"""Analyse results.csv from run_benchmark.py.

Usage: python analyse.py <results_dir> <bbbc_root>

BBBC006 documents z = 16 as the optimal focal plane. In this data every metric
puts the sharpest plane 2-4 um above it, consistently across fields and both
stains (and visual inspection agrees), so "hit rate at z16" says more about the
label than about the metrics. Metrics are therefore also compared on properties
that need no label:
  * cross-channel precision: nuclei and actin are imaged in the same field, so
    the per-field difference between their estimated focal planes should be
    constant (up to a fixed chromatic offset); its spread (robust SD) measures
    each metric's precision.
  * noise stability: shift of the estimated focal plane when Gaussian noise is
    added (15% of the in-focus image SD).
  * unimodality: share of focus curves with a single peak.
  * speed per 696x520 image.
The focal plane is estimated below the plane spacing by fitting a parabola
through the maximum and its two neighbours (planes are unevenly spaced).
"""
import sys, os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile
from focus_metrics import parabolic_peak

RES, ROOT = sys.argv[1], sys.argv[2]
df = pd.read_csv(os.path.join(RES, "results.csv"))
ZS = sorted(int(c[3:]) for c in df.columns if c.startswith("v_z"))
VC = [f"v_z{z:02d}" for z in ZS]
BEST, DZ = 16, 2.0
X = (np.array(ZS) - BEST) * DZ  # um relative to documented focus
CH = {"w1": "Nuclei (Hoechst)", "w2": "Actin (phalloidin)"}

def subplane(v):
    return parabolic_peak(X, v)

def unimodal(v):
    d = np.sign(np.diff(v)); d = d[d != 0]
    return int((np.diff(d) != 0).sum() <= 1 and (len(d) == 0 or d[0] > 0 or d[-1] < 0))

V = df[VC].to_numpy()
df["est_um"] = [subplane(v) for v in V]
df["unimodal"] = [unimodal(v) for v in V]
df["hit16"] = df.z_pred == BEST

clean, noisy = df[df.noise == 0], df[df.noise == 0.15]
key = ["well", "site", "channel", "metric"]
m = clean.merge(noisy[key + ["est_um"]], on=key, suffixes=("", "_noisy"))
m["noise_shift"] = (m.est_um_noisy - m.est_um).abs()

w = clean.pivot_table(index=["well", "site", "metric"], columns="channel", values="est_um").dropna()
w["d"] = w.w2 - w.w1
def rsd(s): return 1.4826 * np.median(np.abs(s - np.median(s)))
cc = w.groupby("metric").d.agg(chromatic_offset_um="median", cross_channel_rsd_um=rsd)

score = pd.DataFrame({
    "median_focus_offset_um": clean.groupby("metric").est_um.median(),
    "hit_rate_at_documented_z16": clean.groupby("metric").hit16.mean(),
    "unimodal_share": clean.groupby("metric").unimodal.mean(),
    "noise_shift_um_median": m.groupby("metric").noise_shift.median(),
    "noise_shift_um_p90": m.groupby("metric").noise_shift.quantile(0.9),
    "ms_per_image": clean.groupby("metric").ms_per_image.median(),
}).join(cc).sort_values("cross_channel_rsd_um")
score.round(3).to_csv(os.path.join(RES, "scorecard.csv"))
print(score.round(3).to_string())
print("\nfocus offset by channel (median um):")
print(clean.groupby(["metric", "channel"]).est_um.median().unstack().round(2).to_string())

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
order = score.index.tolist()
best = order[0]

# Fig 1: where is focus really? histogram of estimated focal plane (Variance of Laplacian)
ref = "Variance of Laplacian"
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.arange(-9, 9.5, 1)
for ch, col in [("w1", "#3b6fb6"), ("w2", "#d1495b")]:
    s = clean[(clean.metric == ref) & (clean.channel == ch)].est_um
    ax.hist(s, bins=bins, alpha=0.6, color=col, label=f"{CH[ch]}: median {s.median():+.1f} µm")
ax.axvline(0, color="k", ls="--", lw=1)
ax.text(0.25, ax.get_ylim()[1] * 0.9, "labelled\nfocal plane", fontsize=8)
ax.set_xlabel("Sharpest plane relative to the labelled focal plane (µm)"); ax.set_ylabel("Fields")
ax.set_title("BBBC006, 768 fields: the sharpest plane sits ~4 µm above the label", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig1_focus_offset.png"), dpi=200)

# Fig 2: scorecard, three label-free criteria
o = score.sort_values("cross_channel_rsd_um", ascending=False)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
cols = ["#3b6fb6" if n == ref else "#9aa5b1" for n in o.index]
axes[0].barh(o.index, o.cross_channel_rsd_um, color=cols); axes[0].set_xlabel("Precision: nuclei vs actin\nspread (robust SD, µm), lower is better")
axes[1].barh(o.index, o.noise_shift_um_p90, color=cols); axes[1].set_xscale("log")
axes[1].set_xlabel("Noise robustness: focus shift with\n15% added noise (90th pct, µm, log)")
axes[2].barh(o.index, o.unimodal_share * 100, color=cols); axes[2].set_xlim(0, 105)
axes[2].set_xlabel("Single-peaked focus curves (%),\nhigher is better")
for ax, col, fmt in [(axes[0], "cross_channel_rsd_um", "{:.2f}"), (axes[1], "noise_shift_um_p90", "{:.2f}"), (axes[2], "unimodal_share", "{:.0%}")]:
    for y, v in enumerate(o[col]): ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8)
fig.suptitle("9 classic focus metrics on 768 fields x 2 stains (BBBC006), no ground-truth label needed", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig2_scorecard.png"), dpi=200)

# Fig 3: normalised focus curves for three representative metrics
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
for ax, ch in zip(axes, CH, strict=False):
    for mt in ["Variance of Laplacian", "Brenner gradient", "Variance"]:
        v = clean[(clean.channel == ch) & (clean.metric == mt)][VC].to_numpy()
        v = (v - v.min(1, keepdims=True)) / np.ptp(v, 1, keepdims=True).clip(1e-12)
        ax.plot(X, np.median(v, 0), marker="o", label=mt)
        ax.fill_between(X, *np.percentile(v, [25, 75], 0), alpha=0.15)
    ax.axvline(0, color="k", lw=0.8, ls="--"); ax.set_title(CH[ch], fontsize=10); ax.set_xlabel("Position relative to labelled focus (µm)")
axes[0].set_ylabel("Normalised metric (median, IQR)"); axes[1].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig3_curves.png"), dpi=200)

# Fig 4: example field through focus (nuclei), the field with most signal at the label plane
ex = clean[(clean.channel == "w1") & (clean.metric == ref)].sort_values("mean_intensity").iloc[-40]
pat = f"_{ex.well}_{ex.site}_w1"
ims = [tifffile.imread(glob.glob(os.path.join(ROOT, f"BBBC006_v1_images_z_{z:02d}", f"*{pat}*.tif"))[0]).astype(float) for z in ZS]
crop = (slice(110, 410), slice(200, 500))
lo, hi = np.percentile(ims[ZS.index(18)][crop], [1, 99.8])
fig, axes = plt.subplots(1, len(ZS), figsize=(14, 2.6))
for ax, x, im in zip(axes, X, ims, strict=True):
    ax.imshow(im[crop], cmap="gray", vmin=lo, vmax=hi); ax.axis("off")
    ax.set_title(f"{x:+.0f} µm" + (" (label)" if x == 0 else ""), fontsize=9)
fig.suptitle(f"Nuclei, field {ex.well} {ex.site}, through focus", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig4_example.png"), dpi=150)
print("figures saved")
