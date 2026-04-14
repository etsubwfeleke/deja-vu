# Deja Vu

Detecting near-duplicate videos that platforms have seen before, even after edits.

Course project for **CSCE 5218 (Deep Learning)** by **Etsub Feleke**.

## Overview

Content moderation systems struggle when banned videos are re-uploaded with edits such as cropping, overlays, color shifts, speed changes, reaction layouts, and combined transformations. In real platforms (e.g., TikTok-style repost patterns), this allows policy-violating content to evade simple duplicate detection. Traditional perceptual hashing is lightweight, but it often degrades under heavy semantic or structural edits.

This project approaches near-duplicate video retrieval using transfer learning with pretrained visual encoders: **CLIP (ViT-B/32)** and **ResNet-50**. Videos are sampled at 1 fps, frame embeddings are extracted, then aggregated into a single video-level embedding via mean pooling and L2 normalization.

At inference time, each uploaded video embedding is compared to a banned-content embedding database using cosine similarity. A tiered decision policy maps similarity into moderation actions: **Auto-flag**, **Human Review**, or **Allow**. The system is evaluated across easy and hard transformation sets, plus negative videos.

## Key Results

| Method | Easy mAP | Hard mAP | Best F1 |
|---|---:|---:|---:|
| CLIP | 0.972 | 0.828 | 0.877 |
| ResNet-50 | 0.979 | 0.905 | 0.901 |
| pHash baseline | - | - | easy: 0.889, hard: 0.857 |

ResNet provides stronger overall discrimination, while CLIP is notably more robust on structural edits (especially reaction layouts and heavy combined transformations).

## Project Structure

```text
deja-vu/
    data/
        banned/              # 52 source 'banned' videos
        reuploads/           # 416 easy transformations
        reuploads_hard/      # 312 hard transformations
        negatives/           # 104 negative videos
    embeddings/              # Precomputed .npy embedding files
    results/                 # Numbered test folders with outputs
    src/
        extract_embeddings.py
        generate_reuploads.py
        generate_hard_reuploads.py
        evaluate.py
        evaluate_hashing.py
        threshold_analysis.py
        frame_efficiency.py
        live_demo.py
        run_all_embeddings.sh
```

## Setup

```bash
conda create -n deja-vu python=3.11 -y
conda activate deja-vu
uv pip install torch torchvision torchaudio openai-clip opencv-python numpy faiss-cpu matplotlib seaborn scikit-learn imagehash
```

Apple Silicon Macs use **MPS** acceleration; NVIDIA systems use **CUDA**.

## Usage

Run the full pipeline in this order:

### 1) Generate transformations
```bash
python src/generate_reuploads.py
python src/generate_hard_reuploads.py
```

### 2) Extract embeddings
```bash
bash src/run_all_embeddings.sh
```

### 3) Run evaluation
```bash
python src/evaluate.py --model clip
python src/evaluate.py --model resnet
python src/evaluate_hashing.py
```

### 4) Run threshold analysis
```bash
python src/threshold_analysis.py
```

### 5) Run frame efficiency
```bash
python src/frame_efficiency.py
```

### 6) Run live demo
```bash
python src/live_demo.py --video data/reuploads_hard/banned_001_reaction.mp4
```

## Dataset

Videos are sourced from **Pexels** and **Pixabay** under Creative Commons-friendly licenses. The dataset includes 14 transformation types total: **8 easy** and **6 hard**. Raw video assets are not included in the repository due to size constraints; recreate them locally using the generation scripts.

## Experiments

1. **Embedding model comparison** — CLIP vs ResNet on near-duplicate retrieval quality.
2. **Hashing baseline comparison** — perceptual hashing performance under easy and hard edits.
3. **Threshold sweep** — precision/recall/F1 behavior across decision thresholds.
4. **Transformation-level robustness** — detection rates by edit type (e.g., heavytext, reaction, heavycombined).
5. **Frame-efficiency tradeoff** — accuracy vs number of sampled frames / runtime cost.

## References

- NDVR-DML: Near-Duplicate Video Retrieval with Deep Metric Learning.
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP).
- He et al., *Deep Residual Learning for Image Recognition* (ResNet).

## License

MIT
