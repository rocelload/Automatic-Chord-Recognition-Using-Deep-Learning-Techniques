# make_heatmaps.py
# Build two confusion heatmaps from .lab files:
#   (A) Maj/Min-only (restrict to ref∈{maj,min}, project both sides to {maj,min})
#   (B) Large-vocab (full quality set), time-weighted
#
# Usage (Windows CMD):
#   python make_heatmaps.py ^
#     --ref_dir .\data\isophonics\beatles_labels ^
#     --mm_est_dir .\outputs\beatles_est ^
#     --lg_est_dir .\outputs\beatles_est_large ^
#     --out_dir .\outputs\plots_all
#
# Outputs:
#   heatmap_mm_only.(png|pdf)
#   heatmap_large_quality.(png|pdf)

import argparse, os, glob, re
import numpy as np
import matplotlib.pyplot as plt

# -------------------- utilities --------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def save_fig(base):
    for ext in ("png","pdf"):
        plt.savefig(f"{base}.{ext}", dpi=300, bbox_inches="tight")
    print("saved:", f"{base}.png and .pdf")

def norm_basename(path):
    b = os.path.basename(path)
    b = os.path.splitext(b)[0]
    b = re.sub(r"[\s\-\_]+", " ", b).strip().lower()
    b = re.sub(r"[^\w\s]", "", b)
    return b

def load_lab_strict(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = ln.split()
            if len(parts) < 3: continue
            try:
                s = float(parts[0].replace(",",".")); e = float(parts[1].replace(",",".")); 
            except:
                continue
            if not np.isfinite(s) or not np.isfinite(e) or e <= s: continue
            lab = " ".join(parts[2:]).strip()
            if not lab: continue
            rows.append((s,e,lab))
    if not rows: return None, None
    rows.sort(key=lambda r:(r[0], r[1]))
    iv = np.array([[r[0], r[1]] for r in rows], float)
    lb = np.array([r[2] for r in rows], object)
    return iv, lb

def clip_overlap(intv, labs, t0, t1):
    keep = []
    for (s,e), lab in zip(intv, labs):
        if e <= t0 or s >= t1: continue
        ss = max(s, t0); ee = min(e, t1)
        if ee > ss: keep.append((ss,ee,lab))
    if not keep: return None, None
    iv = np.array([[s,e] for s,e,_ in keep], float)
    lb = np.array([lab for _,_,lab in keep], object)
    return iv, lb

def align_span(ref_i, ref_l, est_i, est_l):
    t0 = max(ref_i[0,0], est_i[0,0]); t1 = min(ref_i[-1,1], est_i[-1,1])
    if t1 <= t0: return None
    ri, rl = clip_overlap(ref_i, ref_l, t0, t1)
    ei, el = clip_overlap(est_i, est_l, t0, t1)
    if ri is None or ei is None: return None
    return ri, rl, ei, el

def quality_only(label: str):
    if label in ("N","X"): return label
    if ":" in label: return label.split(":",1)[1]
    return label

# -------------------- (A) Maj/Min-only confusion --------------------
# We project qualities to {maj,min} and only include time where REF is maj/min.
# Projection rule (simple, thesis-friendly):
#   - Any quality containing "min" -> "min" (min, min7, min6, minmaj7, ...)
#   - Else -> "maj" (maj, maj7, 7, 6, aug, dim, sus2/4, etc.)
# Rationale: maj/min task is about the third; labels lacking a third are treated as non-min -> maj.
def to_mm(qual: str):
    q = qual.lower()
    if q == "n" or q == "x": return q
    return "min" if "min" in q else "maj"

def mm_only_confusion(ref_dir, est_dir):
    # 2x2 matrix on {maj,min}; we exclude times where REF is not maj/min.
    order = ["maj","min"]
    idx = {q:i for i,q in enumerate(order)}
    C = np.zeros((2,2), float)
    refs = {norm_basename(p): p for p in glob.glob(os.path.join(ref_dir, "*.lab"))}
    ests = {norm_basename(p): p for p in glob.glob(os.path.join(est_dir, "*.lab"))}
    keys = sorted(set(refs) & set(ests))
    skipped = 0
    for k in keys:
        ri, rl = load_lab_strict(refs[k]); ei, el = load_lab_strict(ests[k])
        if ri is None or ei is None: skipped += 1; continue
        aligned = align_span(ri, rl, ei, el)
        if aligned is None: skipped += 1; continue
        ri2, rl2, ei2, el2 = aligned
        i = j = 0
        while i < len(ri2) and j < len(ei2):
            s = max(ri2[i,0], ei2[j,0]); t = min(ri2[i,1], ei2[j,1])
            if t > s:
                rq = to_mm(quality_only(rl2[i]))  # project REF to maj/min/n/x
                # only keep spans where reference is maj/min
                if rq in ("maj","min"):
                    eq = to_mm(quality_only(el2[j]))  # project EST
                    # and only count if estimate is also maj/min (skip N/X)
                    if eq in ("maj","min"):
                        C[idx[rq], idx[eq]] += (t - s)
            if ri2[i,1] <= ei2[j,1]: i += 1
            else: j += 1
    # row-normalize to %
    row = C.sum(axis=1, keepdims=True)
    P = np.divide(C, row, out=np.zeros_like(C), where=row>0) * 100.0
    return P, skipped, order

def plot_mm_heatmap(P, order, title, out_base):
    plt.figure(figsize=(4.8,4.2), dpi=150)
    plt.imshow(P, origin="lower", vmin=0, vmax=100)
    plt.colorbar(label="Time-weighted %")
    plt.xticks(range(len(order)), order)
    plt.yticks(range(len(order)), order)
    plt.xlabel("Estimated"); plt.ylabel("Reference")
    plt.title(title)
    plt.tight_layout(); save_fig(out_base); plt.close()

# -------------------- (B) Large-vocab quality confusion --------------------
QUALITY_ORDER = [
    "maj","min","dim","aug","min6","maj6","min7","minmaj7","maj7","7",
    "dim7","hdim7","sus2","sus4","N","X"
]
def large_quality_confusion(ref_dir, est_dir):
    idx = {q:i for i,q in enumerate(QUALITY_ORDER)}
    C = np.zeros((len(QUALITY_ORDER), len(QUALITY_ORDER)), float)
    refs = {norm_basename(p): p for p in glob.glob(os.path.join(ref_dir, "*.lab"))}
    ests = {norm_basename(p): p for p in glob.glob(os.path.join(est_dir, "*.lab"))}
    keys = sorted(set(refs) & set(ests))
    skipped = 0
    for k in keys:
        ri, rl = load_lab_strict(refs[k]); ei, el = load_lab_strict(ests[k])
        if ri is None or ei is None: skipped += 1; continue
        aligned = align_span(ri, rl, ei, el)
        if aligned is None: skipped += 1; continue
        ri2, rl2, ei2, el2 = aligned
        i = j = 0
        while i < len(ri2) and j < len(ei2):
            s = max(ri2[i,0], ei2[j,0]); t = min(ri2[i,1], ei2[j,1])
            if t > s:
                rq = quality_only(rl2[i]); eq = quality_only(el2[j])
                if rq in idx and eq in idx:
                    C[idx[rq], idx[eq]] += (t - s)
            if ri2[i,1] <= ei2[j,1]: i += 1
            else: j += 1
    row = C.sum(axis=1, keepdims=True)
    P = np.divide(C, row, out=np.zeros_like(C), where=row>0) * 100.0
    return P, skipped

def plot_large_heatmap(P, title, out_base):
    plt.figure(figsize=(9,7), dpi=150)
    plt.imshow(P, origin="lower")
    plt.colorbar(label="Time-weighted %")
    plt.xticks(range(len(QUALITY_ORDER)), QUALITY_ORDER, rotation=45, ha="right")
    plt.yticks(range(len(QUALITY_ORDER)), QUALITY_ORDER)
    plt.xlabel("Estimated quality"); plt.ylabel("Reference quality")
    plt.title(title)
    plt.tight_layout(); save_fig(out_base); plt.close()

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--mm_est_dir", required=True)
    ap.add_argument("--lg_est_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # (A) Maj/Min-only
    Pmm, skip_mm, order = mm_only_confusion(args.ref_dir, args.mm_est_dir)
    plot_mm_heatmap(Pmm, order, f"Maj/Min Confusion (ref-only maj/min)  [skipped: {skip_mm}]",
                    os.path.join(args.out_dir, "heatmap_mm_only"))

    # (B) Large-vocab quality
    Plg, skip_lg = large_quality_confusion(args.ref_dir, args.lg_est_dir)
    plot_large_heatmap(Plg, f"Large-vocab Quality Confusion  [skipped: {skip_lg}]",
                       os.path.join(args.out_dir, "heatmap_large_quality"))

    print("✅ Done. Heatmaps saved in:", args.out_dir)

if __name__ == "__main__":
    main()
