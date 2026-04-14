import argparse
import importlib
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

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
    raise ImportError("CLIP is not available. Install clip or open-clip-torch.")


def sample_frames_1fps(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_interval = max(1, int(round(fps))) if fps > 0 else 1

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


def save_frame_bgr(frame, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def extract_clip_frame_embeddings(frames, model, preprocess, device):
    embeddings = []
    with torch.no_grad():
        for frame in tqdm(frames, desc="Extracting CLIP embeddings", unit="frame"):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            image = preprocess(frame_pil).unsqueeze(0).to(device)
            features = model.encode_image(image)
            embeddings.append(features.float().cpu().numpy())

    if not embeddings:
        return np.array([])
    return np.concatenate(embeddings, axis=0)


def mean_pool_l2_normalize(frame_embeddings):
    video_embedding = np.mean(frame_embeddings, axis=0)
    norm = np.linalg.norm(video_embedding)
    if norm > 0:
        video_embedding = video_embedding / norm
    return video_embedding.reshape(-1)


def load_banned_embeddings(path):
    raw = np.load(path, allow_pickle=True).item()
    names = []
    vectors = []
    for name, vec in raw.items():
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        names.append(name)
        vectors.append(arr)

    if not vectors:
        raise ValueError(f"No embeddings found in {path}")

    return names, np.stack(vectors, axis=0)


def find_video_file(banned_dir, filename):
    direct = banned_dir / filename
    if direct.exists():
        return direct

    stem = Path(filename).stem
    candidates = sorted(banned_dir.glob(f"{stem}.*"))
    if candidates:
        return candidates[0]

    return None


def load_first_frame(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    return frame


def decision_from_score(score):
    if score >= 0.90:
        return "Auto-flag"
    if score >= 0.70:
        return "Human Review"
    return "Allow"


def run_live_demo(video_path, banned_embeddings_path, banned_videos_dir, results_dir):
    start = time.time()

    device = get_device()
    print(f"Device: {device}")
    if str(device) != "mps":
        print("Warning: MPS is not available; using fallback device.")

    print("Loading CLIP model...")
    model, preprocess = load_clip_model(device)

    print(f"Loading banned embeddings from {banned_embeddings_path}...")
    banned_names, banned_matrix = load_banned_embeddings(banned_embeddings_path)

    print(f"Sampling frames at 1fps from: {video_path}")
    frames = sample_frames_1fps(video_path)
    if not frames:
        raise RuntimeError("No frames sampled from input video.")
    print(f"Sampled frames: {len(frames)}")

    uploaded_frame_path = results_dir / "demo_uploaded_frame.jpg"
    save_frame_bgr(frames[0], uploaded_frame_path)
    print(f"Saved uploaded first frame: {uploaded_frame_path}")

    frame_embeddings = extract_clip_frame_embeddings(frames, model, preprocess, device)
    if len(frame_embeddings) == 0:
        raise RuntimeError("No frame embeddings were produced.")

    video_embedding = mean_pool_l2_normalize(frame_embeddings)
    similarities = banned_matrix @ video_embedding

    sorted_idx = np.argsort(-similarities)
    best_idx = int(sorted_idx[0])
    best_name = banned_names[best_idx]
    best_score = float(similarities[best_idx])

    closest_video_path = find_video_file(banned_videos_dir, best_name)
    if closest_video_path is None:
        raise FileNotFoundError(
            f"Closest banned match '{best_name}' not found in {banned_videos_dir}"
        )

    banned_first_frame = load_first_frame(closest_video_path)
    banned_frame_path = results_dir / "demo_banned_frame.jpg"
    save_frame_bgr(banned_first_frame, banned_frame_path)

    decision = decision_from_score(best_score)

    print("\n=== LIVE DEMO RESULT ===")
    print(f"Input video: {video_path}")
    print(f"Closest banned match: {best_name}")
    print(f"Closest banned file: {closest_video_path}")
    print(f"Score: {best_score:.4f}")
    print(f"Decision: {decision}")
    print(f"Saved closest banned first frame: {banned_frame_path}")

    print("\nTop 5 closest banned videos:")
    top_k = min(5, len(banned_names))
    for rank in range(top_k):
        idx = int(sorted_idx[rank])
        print(f"  {rank + 1}. {banned_names[idx]}  ->  {float(similarities[idx]):.4f}")

    elapsed = time.time() - start
    print(f"\nProcessing time: {elapsed:.2f} seconds")



def main():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Run live near-duplicate video detection demo using CLIP and banned embeddings."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--banned_embeddings",
        type=str,
        default=str(project_root / "embeddings" / "banned_clip.npy"),
        help="Path to banned CLIP embeddings (.npy)",
    )
    parser.add_argument(
        "--banned_dir",
        type=str,
        default=str(project_root / "data" / "banned"),
        help="Path to banned video folder",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(project_root / "results"),
        help="Directory to save demo output frames",
    )

    args = parser.parse_args()

    video_path = Path(args.video)
    banned_embeddings_path = Path(args.banned_embeddings)
    banned_videos_dir = Path(args.banned_dir)
    results_dir = Path(args.results_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")
    if not banned_embeddings_path.exists():
        raise FileNotFoundError(f"Banned embeddings not found: {banned_embeddings_path}")
    if not banned_videos_dir.exists():
        raise FileNotFoundError(f"Banned video directory not found: {banned_videos_dir}")

    run_live_demo(video_path, banned_embeddings_path, banned_videos_dir, results_dir)


if __name__ == "__main__":
    main()
