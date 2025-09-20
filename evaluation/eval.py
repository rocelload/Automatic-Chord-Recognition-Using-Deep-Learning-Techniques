# eval.py
# Unified evaluator for Maj/Min (small vocab) and Large-vocab runs.
# - Auto-detects mode from estimated labels, or accept --mode {auto,mm,large}
# - Uses robust .lab parsing + simple overlap alignment (no adjust_intervals)
# - Writes per_song_metrics.csv and summary.csv with the right metric set
#
# Usage examples:
#   python eval.py --ref_dir data/isophonics/beatles_labels --est_dir outputs/beatles_est --out_dir outputs/eval_mm2 --as_percent
#   python eval.py --ref_dir data/isophonics/beatles_labels --est_dir outputs/beatles_est_large --out_dir outputs/eval_large2 --as_percent
#   python eval.py --ref_dir ... --est_dir ... --out_dir ... --mode mm
#   python eval.py --ref_dir ... --est_dir ... --out_dir ... --mode large --include_majmin

import argparse, os, re, glob, csv, math
import numpy as np
import mir_eval

# ---------- utilities ----------

def norm_basename(path: str) -> str:
    b = os.path.basename(path)
    b = os.path.splitext(b)[0]
    b = re.sub(r"[\s\-\_]+", " ", b).strip().lower()
    b = re.sub(r"[^\w\s]", "", b)
    return b

def load_lab_strict(path):
    """Robust .lab reader -> (Nx2 intervals float, N labels object). Skips bad rows."""
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

def clip_to_overlap(intv, labs, t_min, t_max):
    """Clip labeled intervals to [t_min, t_max] and merge identical adjacents."""
    keep = []
    for (s, e), lab in zip(intv, labs):
        if e <= t_min or s >= t_max:
            continue
        s2 = max(s, t_min); e2 = min(e, t_max)
        if e2 > s2:
            keep.append((s2, e2, lab))
    if not keep:
        return None, None
    merged = [keep[0]]
    for s, e, lab in keep[1:]:
        ps, pe, plab = merged[-1]
        if lab == plab and abs(s - pe) < 1e-9:
            merged[-1] = (ps, e, plab)
        else:
            merged.append((s, e, lab))
    ivals = np.array([[s, e] for s, e, _ in merged], float)
    labs  = np.array([lab for _, _, lab in merged], object)
    return ivals, labs

def align_simple(ref_i, ref_l, est_i, est_l):
    """Robust alignment by clipping both to common span (no adjust_intervals)."""
    t_min = max(float(ref_i[0,0]), float(est_i[0,0]))
    t_max = min(float(ref_i[-1,1]), float(est_i[-1,1]))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return None
    ri2, rl2 = clip_to_overlap(ref_i, ref_l, t_min, t_max)
    ei2, el2 = clip_to_overlap(est_i, est_l, t_min, t_max)
    if ri2 is None or ei2 is None:
        return None
    return ri2, rl2, ei2, el2

def pair_files(ref_dir, est_dir):
    refs = {norm_basename(p): p for p in glob.glob(os.path.join(ref_dir, "*.lab"))}
    ests = {norm_basename(p): p for p in glob.glob(os.path.join(est_dir, "*.lab"))}
    keys = sorted(set(refs) & set(ests))
    return [(k, refs[k], ests[k]) for k in keys], sorted(set(ests)-set(refs)), sorted(set(refs)-set(ests))

def chord_quality_only(label: str):
    """Return the quality token after ':' if present (e.g., 'maj', 'min', '7', ...), or 'N'/'X'."""
    if label in ("N","X"):
        return label
    if ":" in label:
        return label.split(":", 1)[1]
    return label

def detect_mode_from_est(est_dir, sample_limit=5):
    """Inspect a few estimate files; if all qualities ∈ {maj,min,N,X} => mm, else large."""
    mm_ok = True
    checked = 0
    for p in glob.glob(os.path.join(est_dir, "*.lab")):
        try:
            _, labs = load_lab_strict(p)
        except Exception:
            continue
        quals = {chord_quality_only(l) for l in labs}
        allowed = {"maj","min","N","X"}
        if not quals.issubset(allowed):
            mm_ok = False
            break
        checked += 1
        if checked >= sample_limit:
            break
    return "mm" if mm_ok else "large"

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--est_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", default="auto", choices=["auto","mm","large"],
                    help="Maj/Min (mm), Large, or auto-detect from estimates.")
    ap.add_argument("--include_majmin", action="store_true",
                    help="If mode=large, also compute/write majmin for comparison.")
    ap.add_argument("--as_percent", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    mode = args.mode
    if mode == "auto":
        mode = detect_mode_from_est(args.est_dir)
    print(f"[info] evaluation mode: {mode}")

    # Metric subsets by mode
    metrics_large = ["root","thirds","triads","sevenths","tetrads","mirex"]
    metrics_mm    = ["root","majmin"]

    pairs, missing_ref, missing_est = pair_files(args.ref_dir, args.est_dir)
    if missing_ref:
        with open(os.path.join(args.out_dir, "missing_ref.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing_ref))
    if missing_est:
        with open(os.path.join(args.out_dir, "missing_est.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing_est))

    rows = []
    bad  = []
    for key, rpath, epath in pairs:
        try:
            ref_i, ref_l = load_lab_strict(rpath)
        except Exception as e:
            bad.append((key, "REF_PARSE", str(e))); continue
        try:
            est_i, est_l = load_lab_strict(epath)
        except Exception as e:
            bad.append((key, "EST_PARSE", str(e))); continue

        aligned = align_simple(ref_i, ref_l, est_i, est_l)
        if aligned is None:
            bad.append((key, "ALIGN", "no overlap after clipping")); continue
        ri2, rl2, ei2, el2 = aligned

        try:
            scores = mir_eval.chord.evaluate(ri2, rl2, ei2, el2)
        except Exception as e:
            bad.append((key, "EVAL", str(e))); continue

        row = {"song": key}
        if mode == "mm":
            for m in metrics_mm:
                v = float(scores.get(m, 0.0))
                if args.as_percent: v *= 100.0
                row[m] = v
        else:
            for m in metrics_large:
                v = float(scores.get(m, 0.0))
                if args.as_percent: v *= 100.0
                row[m] = v
            if args.include_majmin and "majmin" in scores:
                row["majmin"] = float(scores["majmin"])*(100.0 if args.as_percent else 1.0)
        rows.append(row)

    # Decide final header order
    if mode == "mm":
        header = ["song"] + metrics_mm
    else:
        header = ["song"] + metrics_large + (["majmin"] if any(("majmin" in r) for r in rows) else [])

    # Write per-song CSV
    per_csv = os.path.join(args.out_dir, "per_song_metrics.csv")
    with open(per_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows: w.writerow(r)

    # Write summary CSV (mean/median)
    sum_csv = os.path.join(args.out_dir, "summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["metric","mean","median","n_songs"])
        keys = header[1:]
        n = len(rows)
        for m in keys:
            arr = np.array([float(r[m]) for r in rows if m in r], float)
            mean = float(arr.mean()) if arr.size else math.nan
            med  = float(np.median(arr)) if arr.size else math.nan
            wr.writerow([m, mean, med, n])

    if bad:
        log = os.path.join(args.out_dir, "bad_pairs.tsv")
        with open(log, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["song","stage","error"])
            wr.writerows(bad)
        print(f"[warn] skipped {len(bad)} song(s); see {log}")

    print("✅ Wrote:")
    print("  -", per_csv)
    print("  -", sum_csv)

if __name__ == "__main__":
    main()
