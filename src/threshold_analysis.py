import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


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


def compute_prf_at_thresholds(pos_scores, neg_scores, thresholds):
    y_true = np.concatenate([
        np.ones(len(pos_scores), dtype=np.int32),
        np.zeros(len(neg_scores), dtype=np.int32),
    ])
    y_scores = np.concatenate([pos_scores, neg_scores])

    rows = []
    for th in thresholds:
        y_pred = (y_scores >= th).astype(np.int32)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append((float(th), precision, recall, f1, tp, fp, fn))

    return y_true, y_scores, rows


def best_threshold_row(rows):
    if not rows:
        return None
    max_f1 = max(r[3] for r in rows)
    best_rows = [r for r in rows if abs(r[3] - max_f1) < 1e-12]
    best = max(best_rows, key=lambda r: r[0])
    return best


def load_model_scores(embeddings_dir, model):
    banned_path = Path(embeddings_dir) / f"banned_{model}.npy"
    easy_path = Path(embeddings_dir) / f"reuploads_{model}.npy"
    hard_path = Path(embeddings_dir) / f"reuploads_hard_{model}.npy"
    neg_path = Path(embeddings_dir) / f"negatives_{model}.npy"

    for path in [banned_path, easy_path, hard_path, neg_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding file: {path}")

    banned = load_embeddings(banned_path)
    easy = load_embeddings(easy_path)
    hard = load_embeddings(hard_path)
    negatives = load_embeddings(neg_path)

    banned_names = list(banned.keys())
    banned_matrix = np.stack([banned[name] for name in banned_names], axis=0)

    _, _, _, easy_scores = cosine_max_scores(easy, banned_names, banned_matrix)
    _, _, _, hard_scores = cosine_max_scores(hard, banned_names, banned_matrix)
    _, _, _, neg_scores = cosine_max_scores(negatives, banned_names, banned_matrix)

    return {
        "easy_scores": easy_scores,
        "hard_scores": hard_scores,
        "neg_scores": neg_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Threshold analysis for CLIP and ResNet NDVR models")
    parser.add_argument("--embeddings_dir", default="embeddings", help="Folder containing embedding .npy files")
    parser.add_argument("--results_dir", default="results", help="Folder to save analysis results")
    args = parser.parse_args()

    run_dir = create_next_test_dir(args.results_dir, "threshold_analysis")
    thresholds = np.arange(0.50, 0.951, 0.05)

    print("Loading embeddings and computing max cosine similarities...")
    clip_data = load_model_scores(args.embeddings_dir, "clip")
    resnet_data = load_model_scores(args.embeddings_dir, "resnet")

    # Prepare labels/scores for ROC
    y_true_clip_easy = np.concatenate([
        np.ones(len(clip_data["easy_scores"]), dtype=np.int32),
        np.zeros(len(clip_data["neg_scores"]), dtype=np.int32),
    ])
    y_score_clip_easy = np.concatenate([clip_data["easy_scores"], clip_data["neg_scores"]])

    y_true_clip_hard = np.concatenate([
        np.ones(len(clip_data["hard_scores"]), dtype=np.int32),
        np.zeros(len(clip_data["neg_scores"]), dtype=np.int32),
    ])
    y_score_clip_hard = np.concatenate([clip_data["hard_scores"], clip_data["neg_scores"]])

    y_true_resnet_easy = np.concatenate([
        np.ones(len(resnet_data["easy_scores"]), dtype=np.int32),
        np.zeros(len(resnet_data["neg_scores"]), dtype=np.int32),
    ])
    y_score_resnet_easy = np.concatenate([resnet_data["easy_scores"], resnet_data["neg_scores"]])

    y_true_resnet_hard = np.concatenate([
        np.ones(len(resnet_data["hard_scores"]), dtype=np.int32),
        np.zeros(len(resnet_data["neg_scores"]), dtype=np.int32),
    ])
    y_score_resnet_hard = np.concatenate([resnet_data["hard_scores"], resnet_data["neg_scores"]])

    # Plot 1: ROC (4 curves)
    print("Generating Plot 1: ROC curves...")
    roc_path = run_dir / "plot1_roc_curves.png"
    plt.figure(figsize=(8, 6))

    fpr, tpr, _ = roc_curve(y_true_clip_easy, y_score_clip_easy)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"CLIP Easy (AUC={roc_auc:.4f})")

    fpr, tpr, _ = roc_curve(y_true_clip_hard, y_score_clip_hard)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"CLIP Hard (AUC={roc_auc:.4f})")

    fpr, tpr, _ = roc_curve(y_true_resnet_easy, y_score_resnet_easy)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ResNet Easy (AUC={roc_auc:.4f})")

    fpr, tpr, _ = roc_curve(y_true_resnet_hard, y_score_resnet_hard)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"ResNet Hard (AUC={roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: CLIP vs ResNet (Easy + Hard)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # Plot 2: PR curves (Hard only, CLIP + ResNet)
    print("Generating Plot 2: Precision-Recall curves (hard)...")
    pr_path = run_dir / "plot2_precision_recall_hard.png"
    plt.figure(figsize=(8, 6))

    precision, recall, _ = precision_recall_curve(y_true_clip_hard, y_score_clip_hard)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"CLIP Hard (AUC={pr_auc:.4f})")

    precision, recall, _ = precision_recall_curve(y_true_resnet_hard, y_score_resnet_hard)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f"ResNet Hard (AUC={pr_auc:.4f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Hard Transforms)")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=200)
    plt.close()

    # Threshold metrics for summary + F1 plot
    _, _, clip_easy_rows = compute_prf_at_thresholds(clip_data["easy_scores"], clip_data["neg_scores"], thresholds)
    _, _, clip_hard_rows = compute_prf_at_thresholds(clip_data["hard_scores"], clip_data["neg_scores"], thresholds)
    _, _, resnet_easy_rows = compute_prf_at_thresholds(resnet_data["easy_scores"], resnet_data["neg_scores"], thresholds)
    _, _, resnet_hard_rows = compute_prf_at_thresholds(resnet_data["hard_scores"], resnet_data["neg_scores"], thresholds)

    clip_hard_best = best_threshold_row(clip_hard_rows)
    resnet_hard_best = best_threshold_row(resnet_hard_rows)

    # Plot 3: F1 vs threshold (Hard only)
    print("Generating Plot 3: F1 vs threshold (hard)...")
    f1_path = run_dir / "plot3_f1_vs_threshold_hard.png"
    plt.figure(figsize=(8, 6))

    clip_hard_t = [r[0] for r in clip_hard_rows]
    clip_hard_f1 = [r[3] for r in clip_hard_rows]
    resnet_hard_t = [r[0] for r in resnet_hard_rows]
    resnet_hard_f1 = [r[3] for r in resnet_hard_rows]

    plt.plot(clip_hard_t, clip_hard_f1, marker="o", label="CLIP Hard")
    plt.plot(resnet_hard_t, resnet_hard_f1, marker="o", label="ResNet Hard")

    if clip_hard_best is not None:
        plt.axvline(clip_hard_best[0], linestyle="--", alpha=0.7, color="C0")
        plt.annotate(
            f"CLIP best\n(t={clip_hard_best[0]:.2f}, F1={clip_hard_best[3]:.4f})",
            xy=(clip_hard_best[0], clip_hard_best[3]),
            xytext=(clip_hard_best[0] + 0.005, clip_hard_best[3] - 0.08),
            arrowprops={"arrowstyle": "->", "color": "C0"},
            fontsize=9,
            color="C0",
        )

    if resnet_hard_best is not None:
        plt.axvline(resnet_hard_best[0], linestyle="--", alpha=0.7, color="C1")
        plt.annotate(
            f"ResNet best\n(t={resnet_hard_best[0]:.2f}, F1={resnet_hard_best[3]:.4f})",
            xy=(resnet_hard_best[0], resnet_hard_best[3]),
            xytext=(resnet_hard_best[0] - 0.18, resnet_hard_best[3] - 0.12),
            arrowprops={"arrowstyle": "->", "color": "C1"},
            fontsize=9,
            color="C1",
        )

    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 vs Threshold (Hard Transforms)")
    plt.xticks(thresholds)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f1_path, dpi=200)
    plt.close()

    # Plot 4: Optimal threshold summary text table
    print("Generating Plot 4: Optimal threshold summary text...")
    summary_path = run_dir / "plot4_optimal_threshold_summary.txt"

    rows_for_summary = [
        ("CLIP", "Easy", best_threshold_row(clip_easy_rows)),
        ("CLIP", "Hard", best_threshold_row(clip_hard_rows)),
        ("ResNet", "Easy", best_threshold_row(resnet_easy_rows)),
        ("ResNet", "Hard", best_threshold_row(resnet_hard_rows)),
    ]

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Optimal Threshold Summary (max F1)\n")
        f.write("=" * 72 + "\n")
        f.write("Method\tTransform\tThreshold\tPrecision\tRecall\tF1\tTP\tFP\tFN\n")
        for method, difficulty, row in rows_for_summary:
            if row is None:
                f.write(f"{method}\t{difficulty}\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\tN/A\n")
                continue
            th, p, r, f1, tp, fp, fn = row
            f.write(
                f"{method}\t{difficulty}\t{th:.2f}\t{p:.4f}\t{r:.4f}\t{f1:.4f}\t{tp}\t{fp}\t{fn}\n"
            )

    print("\nThreshold analysis complete.")
    print(f"Results saved to: {run_dir}")
    print(f"- Plot 1 ROC: {roc_path.name}")
    print(f"- Plot 2 PR (hard): {pr_path.name}")
    print(f"- Plot 3 F1 vs threshold (hard): {f1_path.name}")
    print(f"- Plot 4 summary: {summary_path.name}")


if __name__ == "__main__":
    main()
