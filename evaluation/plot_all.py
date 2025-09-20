# plot_all.py
# Thesis-ready plots for Maj/Min and Large-vocab evaluations + confusion heatmaps.
#
# Inputs:
#   --mm_dir  <.../eval_mm>      (summary.csv, per_song_metrics.csv)
#   --lg_dir  <.../eval_large>   (summary.csv, per_song_metrics.csv)
#   --ref_dir <.../beatles_labels>   (for confusion heatmaps)
#   --mm_est_dir <.../beatles_est>        estimates for Maj/Min (for heatmap)
#   --lg_est_dir <.../beatles_est_large>  estimates for Large   (for heatmap)
#
# Example:
#   python plot_all.py --mm_dir outputs/eval_mm --lg_dir outputs/eval_large ^
#       --ref_dir data/isophonics/beatles_labels ^
#       --mm_est_dir outputs/beatles_est --lg_est_dir outputs/beatles_est_large ^
#       --out_dir outputs/plots_all
#
# Outputs (PNG + PDF):
#   mm_* , lg_*  (bar/box/trend/hist/cdf)
#   compare_*    (shared metrics, scatter)
#   goodness_*   (%songs ≥ thresholds)
#   heatmap_mm_quality.* , heatmap_large_quality.*   (confusion heatmaps by quality)

import argparse, os, csv, glob, re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------- CSV IO ---------------------------------
def read_summary_csv(path):
    metrics, means, medians, n_songs = [], [], [], None
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            metrics.append(row["metric"])
            means.append(float(row["mean"]) if row["mean"] != "" else np.nan)
            medians.append(float(row["median"]) if row["median"] != "" else np.nan)
            if n_songs is None and "n_songs" in row:
                try: n_songs = int(row["n_songs"])
                except: n_songs = None
    return metrics, np.array(means), np.array(medians), n_songs

def read_per_song_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f); rows = list(rdr)
    if not rows: return [], {}
    metrics = [k for k in rows[0].keys() if k != "song"]
    data = {m: np.array([float(r[m]) for r in rows if r[m] != ""], dtype=float) for m in metrics}
    songs = [r["song"] for r in rows]
    return songs, data

# ---------------------------- Utils ----------------------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_fig(base):
    for ext in ("png","pdf"):
        plt.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
    print("saved:", f"{base}.png and .pdf")

# ---------------------------- Standard plots -------------------------
def bar_means_std(data_dict, metrics_order, title, out_base):
    labels = [m for m in metrics_order if m in data_dict]
    means  = [data_dict[m].mean() for m in labels]
    stds   = [data_dict[m].std(ddof=1) for m in labels]
    plt.figure(figsize=(8.5,4), dpi=150)
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=3)
    plt.xticks(x, labels)
    plt.ylabel("WCSR (%)")
    plt.title(title); plt.ylim(0, 100)
    plt.tight_layout(); save_fig(out_base); plt.close()

def box_per_song(data_dict, metrics_order, title, out_base):
    labels = [m for m in metrics_order if m in data_dict]
    vals   = [data_dict[m] for m in labels]
    plt.figure(figsize=(9,4.8), dpi=150)
    plt.boxplot(vals, labels=labels, showfliers=False)
    plt.ylabel("WCSR (%)")
    plt.title(title); plt.ylim(0, 100)
    plt.tight_layout(); save_fig(out_base); plt.close()

def trend_complexity(data_dict, title, out_base):
    seq = [("triads","Triads"),("sevenths","Sevenths"),("tetrads","Tetrads")]
    seq = [(m,lbl) for m,lbl in seq if m in data_dict]
    if not seq: return
    means  = [data_dict[m].mean() for m,_ in seq]
    labels = [lbl for _,lbl in seq]
    plt.figure(figsize=(7,3.5), dpi=150)
    plt.plot(labels, means, marker="o")
    plt.ylim(0,100); plt.grid(alpha=0.2)
    plt.ylabel("Mean WCSR (%)"); plt.title(title)
    plt.tight_layout(); save_fig(out_base); plt.close()

def hist_per_song(data_dict, metric, title, out_base, bins=20):
    if metric not in data_dict: return
    v = data_dict[metric]
    plt.figure(figsize=(6.8,4), dpi=150)
    plt.hist(v, bins=bins, edgecolor="black", alpha=0.85)
    plt.xlim(0,100); plt.xlabel(f"{metric} WCSR (%)")
    plt.ylabel("#Songs"); plt.title(title); plt.grid(alpha=0.2)
    plt.tight_layout(); save_fig(out_base); plt.close()

def cdf_per_song(data_dict, metric, title, out_base):
    if metric not in data_dict: return
    v = np.sort(data_dict[metric]); y = np.arange(1,len(v)+1)/len(v)*100.0
    plt.figure(figsize=(6.8,4), dpi=150)
    plt.plot(v, y)
    plt.xlim(0,100); plt.ylim(0,100)
    plt.xlabel(f"{metric} WCSR (%)"); plt.ylabel("CDF (% songs ≤ x)")
    plt.title(title); plt.grid(alpha=0.2)
    plt.tight_layout(); save_fig(out_base); plt.close()

def bar_compare_shared(mm_data, lg_data, title, out_base):
    shared = [m for m in ("root","thirds","triads","sevenths","tetrads","mirex") if m in mm_data and m in lg_data]
    if not shared: return
    mm_means = [mm_data[m].mean() for m in shared]
    lg_means = [lg_data[m].mean() for m in shared]
    x = np.arange(len(shared)); w=0.38
    plt.figure(figsize=(9,4), dpi=150)
    plt.bar(x-w/2, mm_means, width=w, label="Maj/Min")
    plt.bar(x+w/2, lg_means, width=w, label="Large")
    plt.xticks(x, shared); plt.ylabel("Mean WCSR (%)")
    plt.title(title); plt.ylim(0,100); plt.legend()
    plt.tight_layout(); save_fig(out_base); plt.close()

def align_intersection(mm_songs, lg_songs, mm_data, lg_data, metric):
    if metric not in mm_data or metric not in lg_data: return None, None, 0
    mm_idx = {s:i for i,s in enumerate(mm_songs)}
    lg_idx = {s:i for i,s in enumerate(lg_songs)}
    common = [s for s in mm_songs if s in lg_idx]
    x = np.array([mm_data[metric][mm_idx[s]] for s in common], float)
    y = np.array([lg_data[metric][lg_idx[s]] for s in common], float)
    n = min(len(x), len(y)); return x[:n], y[:n], n

def scatter_compare(mm_songs, lg_songs, mm_data, lg_data, metric, out_base):
    x,y,n = align_intersection(mm_songs, lg_songs, mm_data, lg_data, metric)
    if n==0: return
    plt.figure(figsize=(4.8,4.8), dpi=150)
    plt.scatter(x,y,s=14,alpha=0.7)
    lim=[0,100]; plt.plot(lim,lim,'k--',lw=1)
    plt.xlim(lim); plt.ylim(lim); plt.grid(alpha=0.2)
    r = np.corrcoef(x,y)[0,1] if n>1 else np.nan
    plt.xlabel(f"Maj/Min {metric} (%)"); plt.ylabel(f"Large {metric} (%)")
    plt.title(f"{metric}: per-song (n={n}, r={r:.2f})")
    plt.tight_layout(); save_fig(out_base); plt.close()

def goodness_stacked(mm_data, lg_data, metrics, thresholds=(80,90), out_prefix="goodness"):
    setups = []
    if mm_data: setups.append(("Maj/Min", mm_data))
    if lg_data: setups.append(("Large", lg_data))
    if not setups: return
    for T in thresholds:
        cats, vals = [], []
        for name, data in setups:
            for m in metrics:
                if m not in data: continue
                v=data[m]; p = 100.0*((v>=T).sum()/len(v)) if len(v) else 0.0
                cats.append(f"{name}-{m}"); vals.append(p)
        if not vals: continue
        plt.figure(figsize=(max(7,0.28*len(cats)),3.6), dpi=150)
        x=np.arange(len(cats))
        plt.bar(x, vals)
        plt.xticks(x, cats, rotation=45, ha="right")
        plt.ylim(0,100); plt.ylabel(f"% songs ≥ {T}%")
        plt.title(f"Model goodness at threshold {T}%")
        plt.tight_layout(); save_fig(f"{out_prefix}_ge{T}"); plt.close()

# ---------------------------- Confusion Heatmap ----------------------
# We’ll compute TIME-WEIGHTED confusion on chord QUALITY.
QUALITY_ORDER = [
    "maj","min","dim","aug","min6","maj6","min7","minmaj7","maj7","7","dim7","hdim7","sus2","sus4","N","X"
]

def norm_basename(path):
    b=os.path.basename(path); b=os.path.splitext(b)[0]
    b=re.sub(r"[\s\-\_]+"," ",b).strip().lower()
    b=re.sub(r"[^\w\s]","",b); return b

def load_lab_strict(path):
    rows=[]
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split()
            if len(parts)<3: continue
            try:
                s=float(parts[0].replace(",",".")); e=float(parts[1].replace(",",".")); 
            except: 
                continue
            if not np.isfinite(s) or not np.isfinite(e) or e<=s: continue
            lab=" ".join(parts[2:]).strip()
            if not lab: continue
            rows.append((s,e,lab))
    if not rows: return None,None
    rows.sort(key=lambda r:(r[0],r[1]))
    ivals=np.array([[r[0],r[1]] for r in rows],float)
    labs =np.array([r[2] for r in rows],object)
    return ivals,labs

def chord_quality_only(label: str):
    if label in ("N","X"): return label
    if ":" in label: return label.split(":",1)[1]
    return label

def clip_overlap(intv,labs,t0,t1):
    keep=[]
    for (s,e),lab in zip(intv,labs):
        if e<=t0 or s>=t1: continue
        ss=max(s,t0); ee=min(e,t1)
        if ee>ss: keep.append((ss,ee,lab))
    if not keep: return None,None
    iv=np.array([[s,e] for s,e,_ in keep],float)
    lb=np.array([lab for _,_,lab in keep],object)
    return iv,lb

def align_span(ref_i,ref_l,est_i,est_l):
    t0=max(ref_i[0,0], est_i[0,0]); t1=min(ref_i[-1,1], est_i[-1,1])
    if t1<=t0: return None
    ri,rl=clip_overlap(ref_i,ref_l,t0,t1); ei,el=clip_overlap(est_i,est_l,t0,t1)
    if ri is None or ei is None: return None
    return ri,rl,ei,el

def time_weighted_confusion(ref_dir, est_dir):
    # pair by normalized basename
    refs={norm_basename(p):p for p in glob.glob(os.path.join(ref_dir,"*.lab"))}
    ests={norm_basename(p):p for p in glob.glob(os.path.join(est_dir,"*.lab"))}
    keys=sorted(set(refs)&set(ests))
    idx={q:i for i,q in enumerate(QUALITY_ORDER)}
    C=np.zeros((len(QUALITY_ORDER),len(QUALITY_ORDER)),float)
    skipped=0
    for k in keys:
        rpath,epath=refs[k],ests[k]
        r=load_lab_strict(rpath); e=load_lab_strict(epath)
        if r[0] is None or e[0] is None: 
            skipped+=1; continue
        aligned=align_span(r[0],r[1],e[0],e[1])
        if aligned is None:
            skipped+=1; continue
        ri,rl,ei,el=aligned
        # sweep by overlapping segments (two-pointer)
        i=j=0
        while i<len(ri) and j<len(ei):
            s=max(ri[i,0], ei[j,0]); t=min(ri[i,1], ei[j,1])
            if t> s:
                rq=chord_quality_only(rl[i]); eq=chord_quality_only(el[j])
                if rq in idx and eq in idx:
                    C[idx[rq], idx[eq]] += (t-s)
            if ri[i,1] <= ei[j,1]: i+=1
            else: j+=1
    # row-normalize to percentages
    row_sums = C.sum(axis=1, keepdims=True)
    P = np.divide(C, row_sums, out=np.zeros_like(C), where=row_sums>0) * 100.0
    return P, skipped

def plot_heatmap(P, title, out_base):
    plt.figure(figsize=(9,7), dpi=150)
    plt.imshow(P, aspect="auto", origin="lower")
    plt.colorbar(label="Time-weighted %")
    plt.xticks(range(len(QUALITY_ORDER)), QUALITY_ORDER, rotation=45, ha="right")
    plt.yticks(range(len(QUALITY_ORDER)), QUALITY_ORDER)
    plt.xlabel("Estimated quality"); plt.ylabel("Reference quality")
    plt.title(title)
    plt.tight_layout(); save_fig(out_base); plt.close()

# ---------------------------- Main -----------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm_dir",  required=True)
    ap.add_argument("--lg_dir",  required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--mm_est_dir", required=True)
    ap.add_argument("--lg_est_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # Load Maj/Min eval CSVs
    mm_sum = os.path.join(args.mm_dir,"summary.csv")
    mm_per = os.path.join(args.mm_dir,"per_song_metrics.csv")
    if not (os.path.exists(mm_sum) and os.path.exists(mm_per)):
        raise FileNotFoundError(f"Maj/Min files not found: {mm_sum} or {mm_per}")
    _,_,_,_ = read_summary_csv(mm_sum)
    mm_songs, mm_data = read_per_song_csv(mm_per)

    # Load Large eval CSVs
    lg_sum = os.path.join(args.lg_dir,"summary.csv")
    lg_per = os.path.join(args.lg_dir,"per_song_metrics.csv")
    if not (os.path.exists(lg_sum) and os.path.exists(lg_per)):
        raise FileNotFoundError(f"Large files not found: {lg_sum} or {lg_per}")
    _,_,_,_ = read_summary_csv(lg_sum)
    lg_songs, lg_data = read_per_song_csv(lg_per)

    # ---- per-scenario plots ----
    mm_order = [m for m in ["root","majmin","mirex","thirds","triads","sevenths","tetrads"] if m in mm_data]
    bar_means_std(mm_data, mm_order, "Maj/Min — Mean ± SD", os.path.join(args.out_dir,"mm_bar_means_std"))
    box_per_song(mm_data, mm_order, "Maj/Min — Per-song distribution", os.path.join(args.out_dir,"mm_box_per_song"))
    trend_complexity(mm_data, "Maj/Min — Performance vs. complexity", os.path.join(args.out_dir,"mm_trend_complexity"))
    for m in [x for x in ("root","majmin","mirex") if x in mm_data]:
        hist_per_song(mm_data, m, f"Maj/Min — {m} histogram", os.path.join(args.out_dir,f"mm_hist_{m}"))
        cdf_per_song(mm_data,  m, f"Maj/Min — {m} CDF",       os.path.join(args.out_dir,f"mm_cdf_{m}"))

    lg_order = [m for m in ["root","thirds","triads","sevenths","tetrads","mirex"] if m in lg_data]
    bar_means_std(lg_data, lg_order, "Large — Mean ± SD", os.path.join(args.out_dir,"lg_bar_means_std"))
    box_per_song(lg_data, lg_order, "Large — Per-song distribution", os.path.join(args.out_dir,"lg_box_per_song"))
    trend_complexity(lg_data, "Large — Performance vs. complexity", os.path.join(args.out_dir,"lg_trend_complexity"))
    for m in [x for x in ("root","mirex") if x in lg_data]:
        hist_per_song(lg_data, m, f"Large — {m} histogram", os.path.join(args.out_dir,f"lg_hist_{m}"))
        cdf_per_song(lg_data,  m, f"Large — {m} CDF",       os.path.join(args.out_dir,f"lg_cdf_{m}"))

    # ---- comparisons ----
    bar_compare_shared(mm_data, lg_data, "Maj/Min vs Large — Mean WCSR (shared metrics)",
                       os.path.join(args.out_dir,"compare_bar_shared"))
    if "root" in mm_data and "root" in lg_data:
        scatter_compare(mm_songs, lg_songs, mm_data, lg_data, "root",
                        os.path.join(args.out_dir,"compare_scatter_root"))
    if "mirex" in mm_data and "mirex" in lg_data:
        scatter_compare(mm_songs, lg_songs, mm_data, lg_data, "mirex",
                        os.path.join(args.out_dir,"compare_scatter_mirex"))

    key_metrics = [m for m in ["root","triads","sevenths","tetrads","mirex","majmin"]
                   if (m in mm_data) or (m in lg_data)]
    goodness_stacked(mm_data, lg_data, key_metrics, thresholds=(80,90),
                     out_prefix=os.path.join(args.out_dir,"goodness"))

    # ---- confusion heatmaps (quality) ----
    P_mm, skip_mm = time_weighted_confusion(args.ref_dir, args.mm_est_dir)
    plot_heatmap(P_mm, f"Confusion (Quality) — Maj/Min  [skipped songs: {skip_mm}]",
                 os.path.join(args.out_dir,"heatmap_mm_quality"))
    P_lg, skip_lg = time_weighted_confusion(args.ref_dir, args.lg_est_dir)
    plot_heatmap(P_lg, f"Confusion (Quality) — Large  [skipped songs: {skip_lg}]",
                 os.path.join(args.out_dir,"heatmap_large_quality"))

    print("✅ All plots are saved to:", args.out_dir)

if __name__ == "__main__":
    main()
