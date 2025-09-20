# plot_quality_confusion.py
# Usage (example):
#   python plot_quality_confusion.py --ref_dir data/isophonics/beatles_labels ^
#       --est_dir outputs/beatles_est --out_png outputs/quality_confusion.png

import argparse, os, re, glob
import numpy as np
import mir_eval
import matplotlib.pyplot as plt

# Chord qualities (mir_eval strings) to include in the confusion matrix
QUALITIES = [
    "maj","min","dim","aug",
    "maj6","min6","maj7","7","min7","minmaj7","hdim7","dim7",
    "sus2","sus4",
    "N","X"   # no-chord & unknown
]

def norm_basename(p):
    b = os.path.basename(p)
    b = os.path.splitext(b)[0]
    b = re.sub(r"[\s\-\_]+", " ", b).strip().lower()
    b = re.sub(r"[^\w\s]", "", b)
    return b

def load_lab_strict(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                s = float(parts[0].replace(",", "."))
                e = float(parts[1].replace(",", "."))
            except Exception:
                continue
            if not np.isfinite(s) or not np.isfinite(e) or e <= s:
                continue
            lab = " ".join(parts[2:]).strip()
            if not lab:
                continue
            rows.append((s, e, lab))
    if not rows:
        raise ValueError(f"No valid rows parsed from {path}")
    rows.sort(key=lambda r: (r[0], r[1]))
    ivals = np.array([[r[0], r[1]] for r in rows], dtype=float)
    labs  = np.array([r[2] for r in rows], dtype=object)
    return ivals, labs

def pair_paths(ref_dir, est_dir):
    refs = {norm_basename(p): p for p in glob.glob(os.path.join(ref_dir, "*.lab"))}
    ests = {norm_basename(p): p for p in glob.glob(os.path.join(est_dir, "*.lab"))}
    keys = sorted(set(refs) & set(ests))
    return [(refs[k], ests[k]) for k in keys]

def parse_root_quality(label):
    # mir_eval uses "C:maj", "A:min7", "N", "X"
    if label in ("N","X"):
        return None, label
    if ":" in label:
        root, qual = label.split(":", 1)
        return root, qual
    return None, label  # fallback

def expand_to_frames(intervals, labels, sr=10.0):
    """Turn variable-length intervals into ~sr fps frame labels."""
    start = float(intervals[0,0]); end = float(intervals[-1,1])
    if end <= start:
        return []
    T = int(np.ceil((end - start) * sr))
    times = start + np.arange(T) / sr
    out = []
    # For each time, find the first interval that contains it
    idx = 0
    for t in times:
        # advance idx while interval end is <= t
        while idx < len(intervals) and intervals[idx][1] <= t:
            idx += 1
        if idx >= len(intervals):
            out.append(labels[-1])
        else:
            if intervals[idx][0] <= t < intervals[idx][1]:
                out.append(labels[idx])
            else:
                # If t is before current interval start (rare due to ceil), backtrack
                j = max(0, idx-1)
                out.append(labels[j])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--est_dir", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--fps", type=float, default=10.8, help="frames per second for frameization")
    args = ap.parse_args()

    # map qualities to indices
    Q2I = {q:i for i,q in enumerate(QUALITIES)}
    M = np.zeros((len(QUALITIES), len(QUALITIES)), dtype=float)
    row_counts = np.zeros(len(QUALITIES), dtype=float)

    pairs = pair_paths(args.ref_dir, args.est_dir)
    used = 0
    for rpath, epath in pairs:
        try:
            ri, rl = load_lab_strict(rpath)
            ei, el = load_lab_strict(epath)
        except Exception:
            continue

        # Clip to common span (simple robust aligner)
        t_min = max(float(ri[0,0]), float(ei[0,0]))
        t_max = min(float(ri[-1,1]), float(ei[-1,1]))
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            continue

        def clip(iv, lb):
            keep = []
            for (s, e), lab in zip(iv, lb):
                if e <= t_min or s >= t_max:
                    continue
                s2 = max(s, t_min); e2 = min(e, t_max)
                if e2 > s2: keep.append((s2, e2, lab))
            if not keep: return None, None
            iv2 = np.array([[s, e] for s,e,_ in keep], float)
            lb2 = np.array([lab for _,_,lab in keep], object)
            return iv2, lb2

        ri2, rl2 = clip(ri, rl)
        ei2, el2 = clip(ei, el)
        if ri2 is None or ei2 is None:
            continue

        r_frames = expand_to_frames(ri2, rl2, sr=args.fps)
        e_frames = expand_to_frames(ei2, el2, sr=args.fps)
        L = min(len(r_frames), len(e_frames))
        if L == 0:
            continue

        for k in range(L):
            r_root, r_q = parse_root_quality(r_frames[k])
            e_root, e_q = parse_root_quality(e_frames[k])
            # Within-root confusion: only count when roots match (or unknown)
            if r_q in Q2I and e_q in Q2I and (r_root == e_root or r_root is None or e_root is None):
                M[Q2I[r_q], Q2I[e_q]] += 1.0
                row_counts[Q2I[r_q]] += 1.0
        used += 1

    # Row-normalize to probabilities
    for i in range(len(QUALITIES)):
        if row_counts[i] > 0:
            M[i, :] /= row_counts[i]

    plt.figure(figsize=(9, 7), dpi=150)
    im = plt.imshow(M, aspect='auto', origin='lower')
    plt.xticks(range(len(QUALITIES)), QUALITIES, rotation=45, ha='right')
    plt.yticks(range(len(QUALITIES)), QUALITIES)
    plt.xlabel("Predicted quality")
    plt.ylabel("Reference quality")
    plt.title("Within-root chord quality confusion")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png)
    print(f"✅ Saved {args.out_png} (from {used} paired songs)")

if __name__ == "__main__":
    main()
