import argparse
import importlib
import os
import re
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import auc, roc_curve

try:
    import clip
    CLIP_LIB = "openai"
except ImportError:
    try:
        open_clip = importlib.import_module("open_clip")
        CLIP_LIB = "open_clip"
    except ImportError:
        clip = None
        open_clip = None
        CLIP_LIB = None


FRAME_SETTINGS = [1, 2, 3, 5, 10, "all"]
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


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip_model(device):
    if CLIP_LIB == "openai":
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        return model, preprocess
    if CLIP_LIB == "open_clip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = model.to(device)
        model.eval()
        return model, preprocess
    raise ImportError("No CLIP library available. Install clip or open-clip-torch.")


def sample_frames_1fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(fps))) if fps and fps > 0 else 1

    frames = []
    frame_count = 0
    success = True
    while success:
        success, frame = cap.read()
        if success and frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames


def unique_preserve_order(indices):
    seen = set()
    out = []
    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


def sample_frames_uniform(video_path, sample_count):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if total_frames <= 0:
        frames = []
        success = True
        while success:
            success, frame = cap.read()
            if success:
                frames.append(frame)
        cap.release()
        if not frames:
            return []
        if sample_count >= len(frames):
            return frames
        indices = np.linspace(0, len(frames) - 1, num=sample_count, endpoint=True)
        indices = unique_preserve_order(np.round(indices).astype(int))
        return [frames[i] for i in indices]

    if sample_count >= total_frames:
        indices = list(range(total_frames))
    else:
        duration = (total_frames - 1) / fps if fps > 0 else float(total_frames - 1)
        times = np.linspace(0.0, duration, num=sample_count, endpoint=True)
        raw_indices = [min(int(round(t * fps)), total_frames - 1) if fps > 0 else int(round(t)) for t in times]
        indices = unique_preserve_order(raw_indices)
        if len(indices) < sample_count:
            for idx in range(total_frames):
                if idx not in indices:
                    indices.append(idx)
                if len(indices) >= sample_count:
                    break
            indices = sorted(indices[:sample_count])

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if success:
            frames.append(frame)
    cap.release()
    return frames


def sample_frames_for_setting(video_path, frame_setting):
    if frame_setting == "all":
        return sample_frames_1fps(video_path)
    return sample_frames_uniform(video_path, int(frame_setting))


def extract_frame_embeddings(frames, model, preprocess, device):
    embeddings = []
    with torch.no_grad():
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            image = preprocess(frame_pil).unsqueeze(0).to(device)
            features = model.encode_image(image)
            embeddings.append(features.float().cpu().numpy())
    if not embeddings:
        return np.array([])
    return np.concatenate(embeddings, axis=0)


def mean_pool_and_normalize(embeddings):
    video_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(video_embedding)
    if norm > 0:
        video_embedding = video_embedding / norm
    return video_embedding.reshape(-1)


def parse_reupload_info(filename):
    stem = Path(filename).stem
    for transform in TRANSFORM_TYPES:
        suffix = f"_{transform}"
        if stem.endswith(suffix):
            source_stem = stem[: -len(suffix)]
            return source_stem, transform
    return stem, "unknown"


def load_video_files(video_dir):
    return sorted(
        f
        for f in os.listdir(video_dir)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"))
    )


def extract_embeddings_for_dir(video_dir, frame_setting, model, preprocess, device):
    video_files = load_video_files(video_dir)
    embeddings = {}
    attempted = len(video_files)
    succeeded = 0

    print(f"  Extracting embeddings from {video_dir} ({attempted} videos)")
    for idx, video_name in enumerate(video_files, start=1):
        video_path = os.path.join(video_dir, video_name)
        print(f"    [{idx}/{attempted}] {video_name}")
        try:
            frames = sample_frames_for_setting(video_path, frame_setting)
            if not frames:
                print("      WARNING: no frames sampled; skipping")
                continue
            frame_embeddings = extract_frame_embeddings(frames, model, preprocess, device)
            if len(frame_embeddings) == 0:
                print("      WARNING: no frame embeddings; skipping")
                continue
            video_embedding = mean_pool_and_normalize(frame_embeddings)
            embeddings[video_name] = video_embedding
            succeeded += 1
        except Exception as exc:
            print(f"      WARNING: failed to process {video_name}: {exc}")
            continue

    return embeddings, attempted, succeeded


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


def compute_map(banned_names, banned_matrix, hard_names, hard_matrix, neg_names, neg_matrix):
    candidate_names = hard_names + neg_names
    candidate_matrix = np.concatenate([hard_matrix, neg_matrix], axis=0)

    aps = []
    per_query = []
    for query_idx, query_name in enumerate(banned_names):
        query_stem = Path(query_name).stem
        sims = candidate_matrix @ banned_matrix[query_idx]

        relevance = []
        for cand in candidate_names:
            if cand in hard_names:
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

    return float(np.mean(aps)) if aps else 0.0, per_query


def precision_recall_f1(y_true, scores, threshold):
    y_pred = (scores >= threshold).astype(np.int32)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp, fp, fn


def compute_setting_metrics(frame_setting, model, preprocess, device, run_dir, threshold=0.75):
    print(f"\n=== Frame setting: {frame_setting} ===")
    setting_start = time.perf_counter()

    banned_dir = "data/banned"
    hard_dir = "data/reuploads_hard"
    neg_dir = "data/negatives"

    # Extraction phase
    extract_start = time.perf_counter()
    banned_embeddings, banned_attempted, banned_succeeded = extract_embeddings_for_dir(
        banned_dir, frame_setting, model, preprocess, device
    )
    hard_embeddings, hard_attempted, hard_succeeded = extract_embeddings_for_dir(
        hard_dir, frame_setting, model, preprocess, device
    )
    neg_embeddings, neg_attempted, neg_succeeded = extract_embeddings_for_dir(
        neg_dir, frame_setting, model, preprocess, device
    )
    extract_end = time.perf_counter()

    banned_names = sorted(banned_embeddings.keys())
    hard_names = sorted(hard_embeddings.keys())
    neg_names = sorted(neg_embeddings.keys())

    if not banned_names:
        raise RuntimeError(f"No banned embeddings were created for frame setting {frame_setting}")

    banned_matrix = np.stack([banned_embeddings[name] for name in banned_names], axis=0)

    # Similarity computations
    compare_start = time.perf_counter()
    _, _, hard_best_matches, hard_scores = cosine_max_scores(hard_embeddings, banned_names, banned_matrix)
    _, _, neg_best_matches, neg_scores = cosine_max_scores(neg_embeddings, banned_names, banned_matrix)
    compare_end = time.perf_counter()

    # mAP computation (banned queries vs hard + negatives)
    hard_matrix = np.stack([hard_embeddings[name] for name in hard_names], axis=0) if hard_names else np.empty((0, banned_matrix.shape[1]), dtype=np.float32)
    neg_matrix = np.stack([neg_embeddings[name] for name in neg_names], axis=0) if neg_names else np.empty((0, banned_matrix.shape[1]), dtype=np.float32)
    map_score, _ = compute_map(banned_names, banned_matrix, hard_names, hard_matrix, neg_names, neg_matrix)

    # F1 at threshold for hard reuploads vs negatives
    y_true = np.concatenate([
        np.ones(len(hard_scores), dtype=np.int32),
        np.zeros(len(neg_scores), dtype=np.int32),
    ])
    y_scores = np.concatenate([hard_scores, neg_scores])
    precision, recall, f1, tp, fp, fn = precision_recall_f1(y_true, y_scores, threshold)

    total_attempted = banned_attempted + hard_attempted + neg_attempted
    total_succeeded = banned_succeeded + hard_succeeded + neg_succeeded
    extraction_seconds = extract_end - extract_start
    compare_seconds = compare_end - compare_start
    total_seconds = time.perf_counter() - setting_start

    extract_time_per_video = extraction_seconds / total_attempted if total_attempted > 0 else 0.0
    compare_time_per_video = compare_seconds / max(total_succeeded, 1)
    total_time_per_video = total_seconds / total_attempted if total_attempted > 0 else 0.0

    return {
        "frame_setting": frame_setting,
        "banned_attempted": banned_attempted,
        "banned_succeeded": banned_succeeded,
        "hard_attempted": hard_attempted,
        "hard_succeeded": hard_succeeded,
        "neg_attempted": neg_attempted,
        "neg_succeeded": neg_succeeded,
        "map": map_score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "extract_seconds": extraction_seconds,
        "compare_seconds": compare_seconds,
        "total_seconds": total_seconds,
        "extract_time_per_video": extract_time_per_video,
        "compare_time_per_video": compare_time_per_video,
        "total_time_per_video": total_time_per_video,
        "hard_scores": hard_scores,
        "neg_scores": neg_scores,
        "hard_best_matches": hard_best_matches,
        "neg_best_matches": neg_best_matches,
    }


def frame_label(frame_setting):
    return "all (1fps)" if frame_setting == "all" else str(frame_setting)


def plot_metrics(results, run_dir):
    labels = [frame_label(r["frame_setting"]) for r in results]
    x = np.arange(len(labels))

    map_vals = [r["map"] for r in results]
    f1_vals = [r["f1"] for r in results]

    # Plot 1: mAP vs frames
    plt.figure(figsize=(8, 5))
    plt.plot(x, map_vals, marker="o")
    plt.xticks(x, labels)
    plt.xlabel("Frame sampling setting")
    plt.ylabel("mAP")
    plt.title("mAP vs Frame Sampling Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "plot1_map_vs_frames.png", dpi=200)
    plt.close()

    # Plot 2: F1 @ 0.75 vs frames
    plt.figure(figsize=(8, 5))
    plt.plot(x, f1_vals, marker="o", color="C1")
    plt.xticks(x, labels)
    plt.xlabel("Frame sampling setting")
    plt.ylabel("F1 @ 0.75")
    plt.title("F1 @ 0.75 vs Frame Sampling Density")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "plot2_f1_vs_frames.png", dpi=200)
    plt.close()


def write_summary(results, run_dir):
    summary_path = run_dir / "frame_efficiency_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Frame Efficiency Analysis\n")
        f.write("=" * 70 + "\n\n")
        f.write("Threshold for F1: 0.75\n")
        f.write("Similarity: max cosine similarity to banned set\n")
        f.write("Frame settings: 1, 2, 3, 5, 10, all (1fps)\n\n")

        f.write(
            "setting\tmAP\tprecision\trecall\tF1\textract_time_per_video\ttotal_time_per_video\tbanned_ok\thard_ok\tneg_ok\n"
        )
        for r in results:
            setting = frame_label(r["frame_setting"])
            f.write(
                f"{setting}\t{r['map']:.4f}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}\t"
                f"{r['extract_time_per_video']:.4f}\t{r['total_time_per_video']:.4f}\t"
                f"{r['banned_succeeded']}/{r['banned_attempted']}\t{r['hard_succeeded']}/{r['hard_attempted']}\t{r['neg_succeeded']}/{r['neg_attempted']}\n"
            )

    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Analyze detection performance vs frame sampling density")
    parser.add_argument("--results_dir", default="results", help="Folder to save analysis results")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print("Loading CLIP model once...")
    model, preprocess = load_clip_model(device)

    run_dir = create_next_test_dir(args.results_dir, "frame_efficiency")
    results = []

    for frame_setting in FRAME_SETTINGS:
        result = compute_setting_metrics(frame_setting, model, preprocess, device, run_dir, threshold=0.75)
        results.append(result)
        print(
            f"Completed {frame_label(frame_setting)}: mAP={result['map']:.4f}, F1={result['f1']:.4f}, "
            f"extract_time/video={result['extract_time_per_video']:.4f}s"
        )

    plot_metrics(results, run_dir)
    summary_path = write_summary(results, run_dir)

    print("\nFrame efficiency analysis complete.")
    print(f"Results saved to: {run_dir}")
    print(f"Summary: {summary_path}")
    print(f"Plot 1: {run_dir / 'plot1_map_vs_frames.png'}")
    print(f"Plot 2: {run_dir / 'plot2_f1_vs_frames.png'}")


if __name__ == "__main__":
    main()
