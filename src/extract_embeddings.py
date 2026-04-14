import argparse
import os
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
from tqdm import tqdm

try:
    import clip
    CLIP_AVAILABLE = True
    CLIP_LIB = "openai"
except ImportError:
    try:
        import open_clip
        CLIP_AVAILABLE = True
        CLIP_LIB = "open_clip"
    except ImportError:
        CLIP_AVAILABLE = False
        CLIP_LIB = None


def get_device():
    """Get the best available device (MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def sample_frames_1fps(video_path):
    """Sample frames at 1 fps from a video."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps))  # Sample every N frames for 1 fps
    
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


def extract_embeddings_clip(frames, model, preprocess, device):
    """Extract embeddings from frames using CLIP ViT-B/32."""
    embeddings = []
    with torch.no_grad():
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame_pil = Image.fromarray(frame_rgb)
            # Preprocess and get embedding
            image = preprocess(frame_pil).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            embeddings.append(image_features.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0) if embeddings else np.array([])


def extract_embeddings_resnet(frames, model, device, transform):
    """Extract embeddings from frames using ResNet-50."""
    embeddings = []
    with torch.no_grad():
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            frame_pil = Image.fromarray(frame_rgb)
            # Apply transforms
            image = transform(frame_pil).unsqueeze(0).to(device)
            # Get embedding (2048-dim after removing classification layer)
            features = model(image).reshape(1, -1)
            embeddings.append(features.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0) if embeddings else np.array([])


def mean_pool_and_normalize(embeddings):
    """Mean-pool frame embeddings and L2-normalize."""
    video_embedding = np.mean(embeddings, axis=0)
    # L2 normalization
    norm = np.linalg.norm(video_embedding)
    if norm > 0:
        video_embedding = video_embedding / norm
    return video_embedding.flatten()


def extract_embeddings_from_videos(input_dir, output_path, model_type="clip"):
    """Extract embeddings from all videos in a directory."""
    if model_type == "clip" and not CLIP_AVAILABLE:
        raise ImportError("CLIP library not available. Install with: pip install git+https://github.com/openai/CLIP.git or pip install open-clip-torch")
    
    device = get_device()
    print(f"Using device: {device}")
    print(f"Using model: {model_type}")
    
    # Load model once
    print("Loading model...")
    if model_type == "clip":
        if CLIP_LIB == "openai":
            model, preprocess = clip.load("ViT-B/32", device=device)
            model.eval()
        elif CLIP_LIB == "open_clip":
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai', device=device)
            model.eval()
        else:
            raise ImportError("CLIP library not available")
    elif model_type == "resnet":
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model = model.to(device)
        model.eval()
        # ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    embeddings_dict = {}
    
    # Get all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f"*{ext}"))
        video_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
    
    # Remove duplicates
    video_files = list(set(video_files))
    video_files.sort()
    
    print(f"Found {len(video_files)} videos")
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        try:
            print(f"\nProcessing: {video_path.name}")
            
            # Sample frames at 1 fps
            frames = sample_frames_1fps(str(video_path))
            
            if len(frames) == 0:
                print(f"  Warning: No frames extracted from {video_path.name}")
                continue
            
            print(f"  Sampled {len(frames)} frames")
            
            # Extract embeddings based on model type
            if model_type == "clip":
                frame_embeddings = extract_embeddings_clip(frames, model, preprocess, device)
            elif model_type == "resnet":
                frame_embeddings = extract_embeddings_resnet(frames, model, device, preprocess)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            if len(frame_embeddings) == 0:
                print(f"  Warning: No embeddings extracted from {video_path.name}")
                continue
            
            # Mean-pool and normalize
            video_embedding = mean_pool_and_normalize(frame_embeddings)
            embeddings_dict[video_path.name] = video_embedding
            
            print(f"  Video embedding shape: {video_embedding.shape}")
        
        except Exception as e:
            print(f"  Error processing {video_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save embeddings
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, embeddings_dict)
    print(f"\n✓ Embeddings saved to {output_path}")
    print(f"✓ Total videos processed: {len(embeddings_dict)}")


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from videos using CLIP or ResNet-50")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Path to folder containing videos")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Path to save embeddings .npy file")
    parser.add_argument("--model", type=str, choices=["clip", "resnet"], default="clip",
                       help="Model to use for embeddings (clip or resnet)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    extract_embeddings_from_videos(args.input_dir, args.output_path, args.model)


if __name__ == "__main__":
    main()
