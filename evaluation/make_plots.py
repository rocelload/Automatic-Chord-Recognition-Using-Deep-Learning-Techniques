# make_plots.py
# Create plots for Maj/Min and Large-vocab evaluations + comparisons.
# Expected files:
#   <mm_dir>/summary.csv, <mm_dir>/per_song_metrics.csv
#   <lg_dir>/summary.csv, <lg_dir>/per_song_metrics.csv
#
# Example runs (Windows CMD):
#   python make_plots.py --mm_dir outputs/eval_mm --lg_dir outputs/eval_large --out_dir outputs/plots_all
#
# Outputs:
#   - mm_bar_means_std.(png|pdf), mm_box_per_song.(png|pdf), mm_trend_complexity.(png|pdf)
#   - lg_bar_means_std.(png|pdf), lg_box_per_song.(png|pdf), lg_trend_complexity.(png|pdf)
#   - compare_bar_shared.(png|pdf), compare_scatter_root.(png|pdf), compare_scatter_mirex.(png|pdf)
#   - goodness_stacked.(png|pdf)   (percentage of songs ≥80 / ≥90)

import argparse, os, csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# --------- IO helpers ---------
def read_summary_csv(path):
    metrics, means, medians, n_songs = [], [], [], None
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            metrics.append(row["metric"])
            means.append(float(row["mean"]) if row["mean"] != "" else np.nan)
            medians.append(float(row["median"]) if row["median"] != "" else np.nan)
            if n_songs is None and "n_songs" in row:
                try:
                    n_songs = int(row["n_songs"])
                except Exception:
                    n_songs = None
    return metrics, np.array(means), np.array(medians), n_songs

def read_per_song_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)
    if not rows:
        return [], {}
    metrics = [k for k in rows[0].keys() if k != "song"]
    data = {m: np.array([float(r[m]) for r in rows if r[m] != ""], dtype=float) for m in metrics}
    songs = [r["song"] for r in rows]
    return songs, data

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_fig(basepath):
    for ext in ("png", "pdf"):
        plt.savefig(f"{basepath}.{ext}", dpi=300, bbox_inches="tight")
    print("saved:", f"{basepath}.png", "and .pdf")

# --------- Plot helpers ---------
def plot_bar_means_std(data_dict, metrics_order, title, out_base):
    means = [data_dict[m].mean() for m in metrics_order if m in data_dict]
    stds  = [data_dict[m].std(ddof=1) for m in metrics_order if m in data_dict]
    labels = [m for m in metrics_order if m in data_dict]
    plt.figure(figsize=(8,4), dpi=150)
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=3)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("WCSR (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.tight_layout()
    save_fig(out_base)
    plt.close()

def plot_box_per_song(data_dict, metrics_order, title, out_base):
    vals = [data_dict[m] for m in metrics_order if m in data_dict]
    labels = [m for m in metrics_order if m in data_dict]
    plt.figure(figsize=(9,4.8), dpi=150)
    plt.boxplot(vals, labels=labels, showfliers=False)
    plt.ylabel("WCSR (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.tight_layout()
    save_fig(out_base)
    plt.close()

def plot_trend_complexity(data_dict, title, out_base):
    seq = [("triads","Triads"), ("sevenths","Sevenths"), ("tetrads","Tetrads")]
    seq = [(m,lbl) for m,lbl in seq if m in data_dict]
    if not seq: 
        return
    means = [data_dict[m].mean() for m,_ in seq]
    labels = [lbl for _,lbl in seq]
    plt.figure(figsize=(7,3.5), dpi=150)
    plt.plot(labels, means, marker="o")
    plt.ylim(0, 100)
    plt.ylabel("Mean WCSR (%)")
    plt.title(title)
    plt.tight_layout()
    save_fig(out_base)
    plt.close()

def align_intersection(mm_songs, lg_songs, mm_data, lg_data, metric):
    mm_index = {s:i for i,s in enumerate(mm_songs)}
    lg_index = {s:i for i,s in enumerate(lg_songs)}
    common = [s for s in mm_songs if s in lg_index]
    x = np.array([mm_data[metric][mm_index[s]] for s in common if metric in mm_data], float)
    y = np.array([lg_data[metric][lg_index[s]] for s in common if metric in lg_data], float)
    # sanity: lengths may mismatch if some songs lack metric
    n = min(len(x), len(y))
    return x[:n], y[:n], n

def plot_scatter_compare(mm_songs, lg_songs, mm_data, lg_data, metric, out_base):
    if metric not in mm_data or metric not in lg_data:
        return
    x, y, n = align_intersection(mm_songs, lg_songs, mm_data, lg_data, metric)
    if n == 0: 
        return
    plt.figure(figsize=(4.8,4.8), dpi=150)
    plt.scatter(x, y, s=14, alpha=0.7)
    lim = [0, 100]
    plt.plot(lim, lim, 'k--', linewidth=1)
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel(f"Maj/Min {metric} (%)")
    plt.ylabel(f"Large {metric} (%)")
    corr = np.corrcoef(x, y)[0,1] if len(x) > 1 else np.nan
    plt.title(f"{metric}: per-song comparison (n={n}, r={corr:.2f})")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    save_fig(out_base)
    plt.close()

def plot_bar_compare_shared(mm_data, lg_data, title, out_base):
    shared = [m for m in ("root","thirds","triads","sevenths","tetrads","mirex") if m in mm_data and m in lg_data]
    if not shared:
        return
    mm_means = [mm_data[m].mean() for m in shared]
    lg_means = [lg_data[m].mean() for m in shared]
    x = np.arange(len(shared))
    w = 0.38
    plt.figure(figsize=(9,4), dpi=150)
    plt.bar(x - w/2, mm_means, width=w, label="Maj/Min")
    plt.bar(x + w/2, lg_means, width=w, label="Large")
    plt.xticks(x, shared)
    plt.ylabel("Mean WCSR (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    save_fig(out_base)
    plt.close()

def plot_goodness_stacked(mm_data, lg_data, metrics, thresholds=(80,90), out_base="goodness"):
    # %songs >= threshold for each metric and setup (mm, lg)
    setups = []
    if mm_data: setups.append(("Maj/Min", mm_data))
    if lg_data: setups.append(("Large",   lg_data))
    if not setups: return
    for T in thresholds:
        cats, vals = [], []
        for name, data in setups:
            for m in metrics:
                if m not in data: 
                    continue
                v = data[m]
                p = 100.0 * ( (v >= T).sum() / len(v) ) if len(v) else 0.0
                cats.append(f"{name}-{m}")
                vals.append(p)
        if not vals: 
            continue
        plt.figure(figsize=(max(7, 0.28*len(cats)), 3.6), dpi=150)
        x = np.arange(len(cats))
        plt.bar(x, vals)
        plt.xticks(x, cats, rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.ylabel(f"% songs ≥ {T}%")
        plt.title(f"Model goodness at threshold {T}% (higher is better)")
        plt.tight_layout()
        save_fig(f"{out_base}_ge{T}")
        plt.close()

# --------- main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm_dir",  required=True, help="Folder with Maj/Min eval (summary.csv, per_song_metrics.csv)")
    ap.add_argument("--lg_dir",  required=True, help="Folder with Large-vocab eval")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # Load Maj/Min
    mm_sum = os.path.join(args.mm_dir, "summary.csv")
    mm_per = os.path.join(args.mm_dir, "per_song_metrics.csv")
    if not (os.path.exists(mm_sum) and os.path.exists(mm_per)):
        raise FileNotFoundError(f"Maj/Min files not found: {mm_sum} or {mm_per}")
    mm_metrics, mm_means, mm_medians, mm_n = read_summary_csv(mm_sum)
    mm_songs, mm_data = read_per_song_csv(mm_per)

    # Load Large
    lg_sum = os.path.join(args.lg_dir, "summary.csv")
    lg_per = os.path.join(args.lg_dir, "per_song_metrics.csv")
    if not (os.path.exists(lg_sum) and os.path.exists(lg_per)):
        raise FileNotFoundError(f"Large files not found: {lg_sum} or {lg_per}")
    lg_metrics, lg_means, lg_medians, lg_n = read_summary_csv(lg_sum)
    lg_songs, lg_data = read_per_song_csv(lg_per)

    # --- per-scenario plots ---
    # Maj/Min metrics to show (Root + MajMin + MIREX if present)
    mm_order = [m for m in ["root","majmin","mirex","thirds","triads","sevenths","tetrads"] if m in mm_data]
    plot_bar_means_std(mm_data, mm_order, "Maj/Min — Mean ± SD", os.path.join(args.out_dir, "mm_bar_means_std"))
    plot_box_per_song(mm_data, mm_order, "Maj/Min — Per-song distribution", os.path.join(args.out_dir, "mm_box_per_song"))
    plot_trend_complexity(mm_data, "Maj/Min — Performance vs. complexity", os.path.join(args.out_dir, "mm_trend_complexity"))

    # Large metrics to show (no MajMin by default)
    lg_order = [m for m in ["root","thirds","triads","sevenths","tetrads","mirex"] if m in lg_data]
    plot_bar_means_std(lg_data, lg_order, "Large — Mean ± SD", os.path.join(args.out_dir, "lg_bar_means_std"))
    plot_box_per_song(lg_data, lg_order, "Large — Per-song distribution", os.path.join(args.out_dir, "lg_box_per_song"))
    plot_trend_complexity(lg_data, "Large — Performance vs. complexity", os.path.join(args.out_dir, "lg_trend_complexity"))

    # --- comparisons ---
    plot_bar_compare_shared(mm_data, lg_data, "Maj/Min vs Large — Mean WCSR (shared metrics)", os.path.join(args.out_dir, "compare_bar_shared"))
    # per-song paired scatter on shared songs (root + mirex if both present)
    if "root" in mm_data and "root" in lg_data:
        plot_scatter_compare(mm_songs, lg_songs, mm_data, lg_data, "root", os.path.join(args.out_dir, "compare_scatter_root"))
    if "mirex" in mm_data and "mirex" in lg_data:
        plot_scatter_compare(mm_songs, lg_songs, mm_data, lg_data, "mirex", os.path.join(args.out_dir, "compare_scatter_mirex"))

    # --- goodness plot: % songs >= 80% and >= 90% for key metrics ---
    key_metrics = [m for m in ["root","triads","sevenths","tetrads","mirex","majmin"] if (m in mm_data) or (m in lg_data)]
    plot_goodness_stacked(mm_data, lg_data, key_metrics, thresholds=(80,90), out_base=os.path.join(args.out_dir, "goodness"))

    print("✅ All plots written to:", args.out_dir)

if __name__ == "__main__":
    main()
