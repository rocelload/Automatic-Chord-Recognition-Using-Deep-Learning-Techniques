# eval_chords_wcsr.py — robust mir_eval evaluator without adjust_intervals()
# Usage:
#   python eval_chords_wcsr.py --ref_dir data/isophonics/beatles_labels --est_dir outputs/beatles_est --out_dir outputs/eval_mm --as_percent

import argparse, os, re, glob, csv, math
import numpy as np
import mir_eval

def norm_basename(path: str) -> str:
    b = os.path.basename(path)
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

def trim_to_overlap(intv, labs, t_min, t_max):
    """Clip labeled intervals to [t_min, t_max] and drop anything outside."""
    keep = []
    for (s, e), lab in zip(intv, labs):
        if e <= t_min or s >= t_max:
            continue
        s2 = max(s, t_min)
        e2 = min(e, t_max)
        if e2 > s2:
            keep.append((s2, e2, lab))
    if not keep:
        return None, None
    # merge adjacent identical labels to keep arrays reasonable
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

def safe_align(ref_i, ref_l, est_i, est_l):
    """Align by clipping both to the common time span; no adjust_intervals()."""
    t_min = max(float(ref_i[0,0]), float(est_i[0,0]))
    t_max = min(float(ref_i[-1,1]), float(est_i[-1,1]))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return None
    ref_i2, ref_l2 = trim_to_overlap(ref_i, ref_l, t_min, t_max)
    est_i2, est_l2 = trim_to_overlap(est_i, est_l, t_min, t_max)
    if ref_i2 is None or est_i2 is None:
        return None
    return ref_i2, ref_l2, est_i2, est_l2

def pair_files(ref_dir, est_dir):
    refs = {norm_basename(p): p for p in glob.glob(os.path.join(ref_dir, "*.lab"))}
    ests = {norm_basename(p): p for p in glob.glob(os.path.join(est_dir, "*.lab"))}
    keys = sorted(set(refs) & set(ests))
    return [(k, refs[k], ests[k]) for k in keys], sorted(set(ests) - set(refs)), sorted(set(refs) - set(ests))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--est_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--as_percent", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs, missing_ref, missing_est = pair_files(args.ref_dir, args.est_dir)

    if missing_ref:
        with open(os.path.join(args.out_dir, "missing_ref.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing_ref))
    if missing_est:
        with open(os.path.join(args.out_dir, "missing_est.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing_est))

    metrics = ["root","thirds","triads","sevenths","tetrads","majmin","mirex"]
    bad = []
    rows = []

    for key, rpath, epath in pairs:
        try:
            ref_i, ref_l = load_lab_strict(rpath)
        except Exception as e:
            bad.append((key, "REF_PARSE", str(e))); continue
        try:
            est_i, est_l = load_lab_strict(epath)
        except Exception as e:
            bad.append((key, "EST_PARSE", str(e))); continue

        aligned = safe_align(ref_i, ref_l, est_i, est_l)
        if aligned is None:
            bad.append((key, "ALIGN", "no overlapping time span after clipping")); continue
        ref_i2, ref_l2, est_i2, est_l2 = aligned

        try:
            scores = mir_eval.chord.evaluate(ref_i2, ref_l2, est_i2, est_l2)
        except Exception as e:
            bad.append((key, "EVAL", str(e))); continue

        row = {"song": key}
        for m in metrics:
            v = float(scores.get(m, 0.0))
            if args.as_percent: v *= 100.0
            row[m] = v
        rows.append(row)

    # per-song CSV
    per_csv = os.path.join(args.out_dir, "per_song_metrics.csv")
    with open(per_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["song"]+metrics)
        w.writeheader()
        for r in rows: w.writerow(r)

    # summary CSV
    sum_csv = os.path.join(args.out_dir, "summary.csv")
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["metric","mean","median","n_songs"])
        n = len(rows)
        for m in metrics:
            arr = np.array([r[m] for r in rows], float)
            wr.writerow([m, float(arr.mean()) if n else math.nan, float(np.median(arr)) if n else math.nan, n])

    if bad:
        log = os.path.join(args.out_dir, "bad_pairs.tsv")
        with open(log, "w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["song","stage","error"])
            wr.writerows(bad)
        print(f"⚠ Skipped {len(bad)} song(s). See {log}")
    print(f"✅ Wrote:\n  - {per_csv}\n  - {sum_csv}")

if __name__ == "__main__":
    main()
