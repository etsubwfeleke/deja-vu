import argparse
import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


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


def load_embeddings(path):
    data = np.load(path, allow_pickle=True).item()
    embeddings = {}
    for name, vec in data.items():
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        embeddings[name] = arr
    return embeddings


def cosine_max_scores(candidates, banned_names, banned_matrix):
    candidate_names = list(candidates.keys())
    if not candidate_names:
        return [], np.array([]), [], np.array([])

    candidate_matrix = np.stack([candidates[n] for n in candidate_names], axis=0)
    sim_matrix = candidate_matrix @ banned_matrix.T
    best_idx = np.argmax(sim_matrix, axis=1)
    best_scores = sim_matrix[np.arange(sim_matrix.shape[0]), best_idx]
    best_match_names = [banned_names[i] for i in best_idx]
    return candidate_names, candidate_matrix, best_match_names, best_scores


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
    return {
        "auto_flag": auto_flag,
        "review": review,
        "allow": allow,
    }


def precision_recall_f1(y_true, scores, thresholds):
    rows = []
    for th in thresholds:
        y_pred = (scores >= th).astype(np.int32)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append((th, precision, recall, f1, tp, fp, fn))
    return rows


def average_precision_binary(sorted_relevance):
    total_relevant = int(np.sum(sorted_relevance))
    if total_relevant == 0:
        return None

    hit_count = 0
    precision_sum = 0.0
    for rank, rel in enumerate(sorted_relevance, start=1):
        if rel:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / total_relevant


def compute_map(banned_names, banned_matrix, reupload_names, reupload_matrix, negative_names, negative_matrix):
    candidate_names = reupload_names + negative_names
    candidate_matrix = np.concatenate([reupload_matrix, negative_matrix], axis=0)

    aps = []
    per_query = []
    for query_idx, query_name in enumerate(banned_names):
        query_stem = Path(query_name).stem
        sims = candidate_matrix @ banned_matrix[query_idx]

        relevance = []
        for cand in candidate_names:
            if cand in reupload_names:
                source_stem, _ = parse_reupload_info(cand)
                relevance.append(1 if source_stem == query_stem else 0)
            else:
                relevance.append(0)
        relevance = np.asarray(relevance, dtype=np.int32)

        order = np.argsort(-sims)
        sorted_relevance = relevance[order]
        ap = average_precision_binary(sorted_relevance)
        if ap is not None:
            aps.append(ap)
            per_query.append((query_name, ap))

    m_ap = float(np.mean(aps)) if aps else 0.0
    return m_ap, per_query


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate NDVR embeddings")
    parser.add_argument("--model", choices=["clip", "resnet"], required=True)
    parser.add_argument("--embeddings_dir", default="embeddings", help="Folder containing embedding .npy files")
    parser.add_argument(
        "--reuploads_override",
        default=None,
        help="Optional custom path to the reuploads embeddings .npy file",
    )
    parser.add_argument("--results_dir", default="results", help="Folder to save evaluation results")
    parser.add_argument("--label", default=None, help="Optional label appended to run folder name")
    args = parser.parse_args()

    banned_path = Path(args.embeddings_dir) / f"banned_{args.model}.npy"
    reuploads_path = (
        Path(args.reuploads_override)
        if args.reuploads_override
        else Path(args.embeddings_dir) / f"reuploads_{args.model}.npy"
    )
    negatives_path = Path(args.embeddings_dir) / f"negatives_{args.model}.npy"

    for path in [banned_path, reuploads_path, negatives_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding file: {path}")

    run_results_dir = create_next_test_dir(args.results_dir, args.label)

    print(f"Loading embeddings for model={args.model}...")
    banned = load_embeddings(banned_path)
    reuploads = load_embeddings(reuploads_path)
    negatives = load_embeddings(negatives_path)

    banned_names = list(banned.keys())
    banned_matrix = np.stack([banned[n] for n in banned_names], axis=0)

    print("Computing max similarity scores for re-uploads and negatives...")
    reup_names, reup_matrix, reup_best_match, reup_scores = cosine_max_scores(
        reuploads, banned_names, banned_matrix
    )
    neg_names, neg_matrix, neg_best_match, neg_scores = cosine_max_scores(
        negatives, banned_names, banned_matrix
    )

    # Overall stats
    reup_mean = float(np.mean(reup_scores)) if len(reup_scores) else 0.0
    neg_mean = float(np.mean(neg_scores)) if len(neg_scores) else 0.0

    # Tier analysis
    reup_tiers = tier_counts(reup_scores)
    neg_tiers = tier_counts(neg_scores)

    # Threshold metrics
    thresholds = np.arange(0.50, 0.951, 0.05)
    y_true = np.concatenate([
        np.ones(len(reup_scores), dtype=np.int32),
        np.zeros(len(neg_scores), dtype=np.int32),
    ])
    y_scores = np.concatenate([reup_scores, neg_scores])
    prf_rows = precision_recall_f1(y_true, y_scores, thresholds)

    # mAP
    print("Computing mAP...")
    m_ap, ap_per_query = compute_map(
        banned_names,
        banned_matrix,
        reup_names,
        reup_matrix,
        neg_names,
        neg_matrix,
    )

    # Transformation breakdown at threshold 0.70
    detection_threshold = 0.70
    transform_totals = {t: 0 for t in TRANSFORM_TYPES}
    transform_detected = {t: 0 for t in TRANSFORM_TYPES}

    for name, score in zip(reup_names, reup_scores):
        _, transform = parse_reupload_info(name)
        if transform in transform_totals:
            transform_totals[transform] += 1
            if score >= detection_threshold:
                transform_detected[transform] += 1

    transform_rates = {}
    for t in TRANSFORM_TYPES:
        total = transform_totals[t]
        rate = (transform_detected[t] / total) if total > 0 else 0.0
        transform_rates[t] = rate

    # Save text summary
    txt_path = run_results_dir / "evaluation_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Banned embeddings: {len(banned_names)}\n")
        f.write(f"Re-upload embeddings: {len(reup_names)}\n")
        f.write(f"Negative embeddings: {len(neg_names)}\n\n")

        f.write("Overall similarity:\n")
        f.write(f"  Mean re-upload max similarity: {reup_mean:.4f}\n")
        f.write(f"  Mean negative max similarity:  {neg_mean:.4f}\n\n")

        f.write("Tier counts (>=0.90 auto-flag, 0.70-0.89 review, <0.70 allow):\n")
        f.write(
            f"  Re-uploads: auto-flag={reup_tiers['auto_flag']}, review={reup_tiers['review']}, allow={reup_tiers['allow']}\n"
        )
        f.write(
            f"  Negatives:  auto-flag={neg_tiers['auto_flag']}, review={neg_tiers['review']}, allow={neg_tiers['allow']}\n\n"
        )

        f.write("Precision/Recall/F1 by threshold:\n")
        f.write("  threshold\tprecision\trecall\tf1\ttp\tfp\tfn\n")
        for th, p, r, f1, tp, fp, fn in prf_rows:
            f.write(f"  {th:.2f}\t{p:.4f}\t{r:.4f}\t{f1:.4f}\t{tp}\t{fp}\t{fn}\n")
        f.write("\n")

        f.write(f"mAP (banned as query over re-uploads+negatives): {m_ap:.4f}\n")
        f.write("AP per query:\n")
        for qname, ap in ap_per_query:
            f.write(f"  {qname}: {ap:.4f}\n")
        f.write("\n")

        f.write(f"Detection rate per transformation @ threshold {detection_threshold:.2f}:\n")
        for t in TRANSFORM_TYPES:
            f.write(
                f"  {t}: {transform_detected[t]}/{transform_totals[t]} ({transform_rates[t]:.4f})\n"
            )

    # Plot (a): histogram
    hist_path = run_results_dir / "similarity_histogram.png"
    plt.figure(figsize=(8, 5))
    plt.hist(reup_scores, bins=30, alpha=0.6, label="Re-uploads")
    plt.hist(neg_scores, bins=30, alpha=0.6, label="Negatives")
    plt.xlabel("Max cosine similarity to banned set")
    plt.ylabel("Count")
    plt.title(f"Similarity Distribution ({args.model})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()

    # Plot (b): precision-recall curve
    pr_curve_path = run_results_dir / "precision_recall_curve.png"
    pr_sorted = sorted(prf_rows, key=lambda x: x[2])
    recalls = [r for _, _, r, _, _, _, _ in pr_sorted]
    precisions = [p for _, p, _, _, _, _, _ in pr_sorted]
    plt.figure(figsize=(6, 6))
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({args.model})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_curve_path, dpi=200)
    plt.close()

    # Plot (c): bar chart by transform at threshold 0.70
    bar_path = run_results_dir / "detection_rate_by_transform.png"
    x = np.arange(len(TRANSFORM_TYPES))
    y = [transform_rates[t] for t in TRANSFORM_TYPES]
    plt.figure(figsize=(10, 5))
    plt.bar(x, y)
    plt.xticks(x, TRANSFORM_TYPES, rotation=30)
    plt.ylim(0, 1.0)
    plt.ylabel("Detection rate")
    plt.title(f"Detection Rate by Transformation @ {detection_threshold:.2f} ({args.model})")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200)
    plt.close()

    print("Evaluation complete.")
    print(f"Mean similarity (re-uploads): {reup_mean:.4f}")
    print(f"Mean similarity (negatives):  {neg_mean:.4f}")
    print(f"mAP: {m_ap:.4f}")
    print(f"Saved summary to: {txt_path}")
    print(f"Saved plot: {hist_path}")
    print(f"Saved plot: {pr_curve_path}")
    print(f"Saved plot: {bar_path}")
    print(f"Created test folder: {run_results_dir}")


if __name__ == "__main__":
    main()