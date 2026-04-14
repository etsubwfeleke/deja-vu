"""
Perceptual hash-based evaluation of video re-upload detection.
Uses frame-level imagehash.phash on sampled frames, then compares videos by
the minimum Hamming distance across all frame-pair combinations.
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


TRANSFORM_TYPES = [
    "crop",
    "mirror",
    "color",
    "speed",
    "lowq",
    "text",
    "trim",
    "combined",
    "heavytext",
    "reaction",
    "grayscale",
    "heavycropzoom",
    "vintage",
    "heavycombined",
]


VIDEO_DIRS = {
    "banned": "data/banned",
    "easy": "data/reuploads",
    "hard": "data/reuploads_hard",
    "negatives": "data/negatives",
}


def sample_frames_1fps(video_path):
    """Sample frames at approximately 1 fps using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(fps))) if fps and fps > 0 else 1

    frames = []
    frame_idx = 0
    success = True

    while success:
        success, frame = cap.read()
        if success and frame_idx % frame_interval == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


def hash_frame(frame):
    """Convert a BGR frame to a 64-bit perceptual hash."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return imagehash.phash(img)


def hash_video_frames(video_path):
    """Sample 1 fps and compute image hashes for all sampled frames."""
    frames = sample_frames_1fps(video_path)
    frame_hashes = []
    for idx, frame in enumerate(frames, start=1):
        try:
            frame_hashes.append(hash_frame(frame))
        except Exception as exc:
            print(f"    WARNING: frame hash failed in {Path(video_path).name} at sampled frame {idx}: {exc}")
    return frame_hashes


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sanitize_label(label):
    if label is None:
        return None
    cleaned = re.sub(r"\s+", "_", label.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", cleaned)
    return cleaned or None


def create_next_test_dir(results_dir, label=None):
    base = Path(results_dir)
    ensure_dir(base)

    max_idx = 0
    pattern = re.compile(r"^test_(\d+)(?:_.*)?$")
    for child in base.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    next_idx = max_idx + 1
    folder_name = f"test_{next_idx}"

    cleaned_label = sanitize_label(label)
    if cleaned_label:
        folder_name = f"{folder_name}_{cleaned_label}"

    run_dir = base / folder_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_reupload_info(filename):
    stem = Path(filename).stem
    for transform in TRANSFORM_TYPES:
        suffix = f"_{transform}"
        if stem.endswith(suffix):
            source_stem = stem[: -len(suffix)]
            return source_stem, transform
    return stem, "unknown"


def tier_counts(scores):
    auto_flag = int(np.sum(scores >= 0.90))
    review = int(np.sum((scores >= 0.70) & (scores < 0.90)))
    allow = int(np.sum(scores < 0.70))
    return {"auto_flag": auto_flag, "review": review, "allow": allow}


def precision_recall_f1(y_true, scores, thresholds):
    rows = []
    for th in thresholds:
        y_pred = (scores >= th).astype(np.int32)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append((th, precision, recall, f1, tp, fp, fn))
    return rows


def similarity_from_distance(distance):
    return 1.0 - (distance / 64.0)


def min_frame_pair_distance(frame_hashes_a, frame_hashes_b):
    """Minimum Hamming distance across all frame-pair combinations."""
    if not frame_hashes_a or not frame_hashes_b:
        return 64

    min_dist = 64
    for ha in frame_hashes_a:
        for hb in frame_hashes_b:
            dist = ha - hb
            if dist < min_dist:
                min_dist = dist
                if min_dist == 0:
                    return 0
    return min_dist


def hash_all_videos(video_dirs):
    """Precompute frame hashes for every video in the provided directories."""
    all_hashes = {}
    order = []
    count = 0

    for set_name, video_dir in video_dirs.items():
        if not os.path.isdir(video_dir):
            print(f"Skipping missing directory: {video_dir}")
            continue

        videos = sorted(
            f
            for f in os.listdir(video_dir)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        )

        print(f"\n=== HASHING {set_name.upper()} ({len(videos)} videos) ===")
        for video in videos:
            count += 1
            video_path = os.path.join(video_dir, video)
            print(f"[{count}] Hashing frames: {video}")
            try:
                frame_hashes = hash_video_frames(video_path)
                if not frame_hashes:
                    print(f"  WARNING: no usable frame hashes for {video}")
                    continue
                all_hashes[video] = frame_hashes
                order.append(video)
            except Exception as exc:
                print(f"  WARNING: failed to hash {video}: {exc}")

    return all_hashes, order


def compare_against_banned(candidate_hashes, candidate_names, banned_hashes, banned_names, label):
    """Compute max similarity to the banned set for each candidate video."""
    scores = []
    best_matches = []

    total = len(candidate_names)
    print(f"\n=== COMPARING {label.upper()} ({total} videos) ===")
    for idx, cand_name in enumerate(candidate_names, start=1):
        print(f"[{idx}/{total}] Comparing: {cand_name}")
        cand_frame_hashes = candidate_hashes[cand_name]

        best_sim = 0.0
        best_banned = None
        for ban_name in banned_names:
            ban_frame_hashes = banned_hashes[ban_name]
            min_dist = min_frame_pair_distance(cand_frame_hashes, ban_frame_hashes)
            sim = similarity_from_distance(min_dist)
            if sim > best_sim:
                best_sim = sim
                best_banned = ban_name

        scores.append(best_sim)
        best_matches.append(best_banned)

    return np.array(scores, dtype=np.float32), best_matches


def write_histogram(scores_a, scores_b, title, xlabel, path, label_a, label_b):
    plt.figure(figsize=(8, 5))
    plt.hist(scores_a, bins=30, alpha=0.6, label=label_a)
    plt.hist(scores_b, bins=30, alpha=0.6, label=label_b)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_pr_curve(pr_rows, title, path):
    sorted_rows = sorted(pr_rows, key=lambda x: x[2])
    recalls = [r for _, _, r, _, _, _, _ in sorted_rows]
    precisions = [p for _, p, _, _, _, _, _ in sorted_rows]
    plt.figure(figsize=(6, 6))
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def write_bar_chart(rates, title, path):
    x = np.arange(len(TRANSFORM_TYPES))
    y = [rates[t] for t in TRANSFORM_TYPES]
    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    plt.xticks(x, TRANSFORM_TYPES, rotation=30)
    plt.ylim(0, 1.0)
    plt.ylabel("Detection rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    print("Loading and precomputing frame hashes...\n")

    hashes_by_name, _ = hash_all_videos(VIDEO_DIRS)

    banned_names = sorted([n for n in hashes_by_name if n in os.listdir(VIDEO_DIRS["banned"])])
    easy_names = sorted([n for n in hashes_by_name if n in os.listdir(VIDEO_DIRS["easy"])])
    hard_names = sorted([n for n in hashes_by_name if n in os.listdir(VIDEO_DIRS["hard"])])
    neg_names = sorted([n for n in hashes_by_name if n in os.listdir(VIDEO_DIRS["negatives"])])

    # More reliable grouping by file origin, preserving hashed names only.
    banned_hashes = {name: hashes_by_name[name] for name in hashes_by_name if os.path.exists(os.path.join(VIDEO_DIRS["banned"], name))}
    easy_hashes = {name: hashes_by_name[name] for name in hashes_by_name if os.path.exists(os.path.join(VIDEO_DIRS["easy"], name))}
    hard_hashes = {name: hashes_by_name[name] for name in hashes_by_name if os.path.exists(os.path.join(VIDEO_DIRS["hard"], name))}
    neg_hashes = {name: hashes_by_name[name] for name in hashes_by_name if os.path.exists(os.path.join(VIDEO_DIRS["negatives"], name))}

    banned_names = sorted(banned_hashes.keys())
    easy_names = sorted(easy_hashes.keys())
    hard_names = sorted(hard_hashes.keys())
    neg_names = sorted(neg_hashes.keys())

    print(f"\nLoaded hashes for {len(banned_names)} banned, {len(easy_names)} easy, {len(hard_names)} hard, {len(neg_names)} negatives\n")

    easy_sims, easy_best = compare_against_banned(easy_hashes, easy_names, banned_hashes, banned_names, "easy re-uploads")
    hard_sims, hard_best = compare_against_banned(hard_hashes, hard_names, banned_hashes, banned_names, "hard re-uploads")
    neg_sims, neg_best = compare_against_banned(neg_hashes, neg_names, banned_hashes, banned_names, "negatives")

    easy_mean = float(np.mean(easy_sims)) if len(easy_sims) else 0.0
    hard_mean = float(np.mean(hard_sims)) if len(hard_sims) else 0.0
    neg_mean = float(np.mean(neg_sims)) if len(neg_sims) else 0.0

    easy_tiers = tier_counts(easy_sims)
    hard_tiers = tier_counts(hard_sims)
    neg_tiers = tier_counts(neg_sims)

    thresholds = np.arange(0.50, 0.951, 0.05)
    y_true_easy = np.concatenate([
        np.ones(len(easy_sims), dtype=np.int32),
        np.zeros(len(neg_sims), dtype=np.int32),
    ])
    y_scores_easy = np.concatenate([easy_sims, neg_sims])
    prf_easy = precision_recall_f1(y_true_easy, y_scores_easy, thresholds)

    y_true_hard = np.concatenate([
        np.ones(len(hard_sims), dtype=np.int32),
        np.zeros(len(neg_sims), dtype=np.int32),
    ])
    y_scores_hard = np.concatenate([hard_sims, neg_sims])
    prf_hard = precision_recall_f1(y_true_hard, y_scores_hard, thresholds)

    detection_threshold = 0.70
    easy_transform_totals = defaultdict(int)
    easy_transform_detected = defaultdict(int)
    hard_transform_totals = defaultdict(int)
    hard_transform_detected = defaultdict(int)

    for name, sim in zip(easy_names, easy_sims):
        _, transform = parse_reupload_info(name)
        easy_transform_totals[transform] += 1
        if sim >= detection_threshold:
            easy_transform_detected[transform] += 1

    for name, sim in zip(hard_names, hard_sims):
        _, transform = parse_reupload_info(name)
        hard_transform_totals[transform] += 1
        if sim >= detection_threshold:
            hard_transform_detected[transform] += 1

    easy_transform_rates = {
        t: (easy_transform_detected[t] / easy_transform_totals[t]) if easy_transform_totals[t] > 0 else 0.0
        for t in TRANSFORM_TYPES
    }
    hard_transform_rates = {
        t: (hard_transform_detected[t] / hard_transform_totals[t]) if hard_transform_totals[t] > 0 else 0.0
        for t in TRANSFORM_TYPES
    }

    run_dir = create_next_test_dir("results", "hashing_baseline")

    txt_path = run_dir / "hashing_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Perceptual Hash-Based Evaluation\n")
        f.write("=" * 50 + "\n\n")

        f.write("Counts:\n")
        f.write(f"  Banned videos: {len(banned_names)}\n")
        f.write(f"  Easy re-uploads: {len(easy_names)}\n")
        f.write(f"  Hard re-uploads: {len(hard_names)}\n")
        f.write(f"  Negatives: {len(neg_names)}\n\n")

        f.write("Overall Mean Similarity:\n")
        f.write(f"  Easy re-uploads: {easy_mean:.4f}\n")
        f.write(f"  Hard re-uploads: {hard_mean:.4f}\n")
        f.write(f"  Negatives: {neg_mean:.4f}\n\n")

        f.write("Tier Counts (>=0.90 auto-flag, 0.70-0.89 review, <0.70 allow):\n")
        f.write(f"  Easy re-uploads: auto-flag={easy_tiers['auto_flag']}, review={easy_tiers['review']}, allow={easy_tiers['allow']}\n")
        f.write(f"  Hard re-uploads: auto-flag={hard_tiers['auto_flag']}, review={hard_tiers['review']}, allow={hard_tiers['allow']}\n")
        f.write(f"  Negatives:      auto-flag={neg_tiers['auto_flag']}, review={neg_tiers['review']}, allow={neg_tiers['allow']}\n\n")

        f.write("Precision/Recall/F1 (Easy Re-uploads):\n")
        f.write("  threshold\tprecision\trecall\tf1\ttp\tfp\tfn\n")
        for th, p, r, f1, tp, fp, fn in prf_easy:
            f.write(f"  {th:.2f}\t{p:.4f}\t{r:.4f}\t{f1:.4f}\t{tp}\t{fp}\t{fn}\n")
        f.write("\n")

        f.write("Precision/Recall/F1 (Hard Re-uploads):\n")
        f.write("  threshold\tprecision\trecall\tf1\ttp\tfp\tfn\n")
        for th, p, r, f1, tp, fp, fn in prf_hard:
            f.write(f"  {th:.2f}\t{p:.4f}\t{r:.4f}\t{f1:.4f}\t{tp}\t{fp}\t{fn}\n")
        f.write("\n")

        f.write(f"Detection Rate (Easy) @ {detection_threshold:.2f}:\n")
        for t in TRANSFORM_TYPES:
            f.write(f"  {t}: {easy_transform_detected[t]}/{easy_transform_totals[t]} ({easy_transform_rates[t]:.4f})\n")
        f.write("\n")

        f.write(f"Detection Rate (Hard) @ {detection_threshold:.2f}:\n")
        for t in TRANSFORM_TYPES:
            f.write(f"  {t}: {hard_transform_detected[t]}/{hard_transform_totals[t]} ({hard_transform_rates[t]:.4f})\n")

    hist_easy_path = run_dir / "histogram_easy.png"
    write_histogram(
        easy_sims,
        neg_sims,
        "Similarity Distribution (Perceptual Hash) - Easy",
        "Hamming Similarity (0-1)",
        hist_easy_path,
        "Easy Re-uploads",
        "Negatives",
    )

    hist_hard_path = run_dir / "histogram_hard.png"
    write_histogram(
        hard_sims,
        neg_sims,
        "Similarity Distribution (Perceptual Hash) - Hard",
        "Hamming Similarity (0-1)",
        hist_hard_path,
        "Hard Re-uploads",
        "Negatives",
    )

    pr_easy_path = run_dir / "precision_recall_curve_easy.png"
    write_pr_curve(prf_easy, "Precision-Recall Curve (Perceptual Hash) - Easy", pr_easy_path)

    pr_hard_path = run_dir / "precision_recall_curve_hard.png"
    write_pr_curve(prf_hard, "Precision-Recall Curve (Perceptual Hash) - Hard", pr_hard_path)

    bar_easy_path = run_dir / "detection_rate_easy.png"
    write_bar_chart(
        easy_transform_rates,
        f"Detection Rate by Transformation (Easy) @ {detection_threshold:.2f}",
        bar_easy_path,
    )

    bar_hard_path = run_dir / "detection_rate_hard.png"
    write_bar_chart(
        hard_transform_rates,
        f"Detection Rate by Transformation (Hard) @ {detection_threshold:.2f}",
        bar_hard_path,
    )

    print("\nEvaluation complete.")
    print(f"Results saved to: {run_dir}")
    print("\nMean similarities:")
    print(f"  Easy re-uploads: {easy_mean:.4f}")
    print(f"  Hard re-uploads: {hard_mean:.4f}")
    print(f"  Negatives:       {neg_mean:.4f}")


if __name__ == "__main__":
    main()
