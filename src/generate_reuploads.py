"""
Generate modified re-uploads from banned videos.
8 transformations per source video to simulate real-world re-upload behavior.
"""

import os
import subprocess
import sys


INPUT_DIR = "data/banned"
OUTPUT_DIR = "data/reuploads"


def run_ffmpeg(cmd):
    """Run an FFmpeg command silently. Returns True on success."""
    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return result.returncode == 0


def get_video_info(path):
    """Get video width, height, and duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path
    ]
    import json
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)

    video_stream = next(
        s for s in info["streams"] if s["codec_type"] == "video"
    )
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    duration = float(info["format"]["duration"])
    return width, height, duration


def apply_transformations(input_path, base_name, output_dir):
    """Apply 8 transformations to a single video."""

    w, h, dur = get_video_info(input_path)
    results = []

    # 1. Crop -- remove 15% from each edge
    crop_w = int(w * 0.7)
    crop_h = int(h * 0.7)
    crop_x = int(w * 0.15)
    crop_y = int(h * 0.15)
    out = os.path.join(output_dir, f"{base_name}_crop.mp4")
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}",
        "-c:v", "libx264", "-preset", "fast", "-an", out
    ])
    results.append(("crop", ok))

    # 2. Horizontal mirror
    out = os.path.join(output_dir, f"{base_name}_mirror.mp4")
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "hflip",
        "-c:v", "libx264", "-preset", "fast", "-an", out
    ])
    results.append(("mirror", ok))

    # 3. Color/brightness shift
    out = os.path.join(output_dir, f"{base_name}_color.mp4")
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "eq=brightness=0.1:contrast=1.3:saturation=1.5",
        "-c:v", "libx264", "-preset", "fast", "-an", out
    ])
    results.append(("color", ok))

    # 4. Speed change (1.25x)
    out = os.path.join(output_dir, f"{base_name}_speed.mp4")
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "setpts=0.8*PTS",
        "-c:v", "libx264", "-preset", "fast", "-an", out
    ])
    results.append(("speed", ok))

    # 5. Low quality re-encode (CRF 35 = very noticeable compression)
    out = os.path.join(output_dir, f"{base_name}_lowq.mp4")
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-crf", "35", "-preset", "fast", "-an", out
    ])
    results.append(("lowq", ok))

    # 6. Text overlay (simulating TikTok caption)
    out = os.path.join(output_dir, f"{base_name}_text.mp4")
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", (
            "drawtext=text='REPOSTED':"
            "fontsize=48:fontcolor=white:"
            "borderw=2:bordercolor=black:"
            "x=(w-text_w)/2:y=h-th-40"
        ),
        "-c:v", "libx264", "-preset", "fast", "-an", out
    ])
    results.append(("text", ok))

    # 7. Trim -- cut first and last 15% of duration
    trim_start = dur * 0.15
    trim_end = dur * 0.85
    out = os.path.join(output_dir, f"{base_name}_trim.mp4")
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-ss", str(trim_start), "-to", str(trim_end),
        "-c:v", "libx264", "-preset", "fast", "-an", out
    ])
    results.append(("trim", ok))

    # 8. Combined -- crop + color shift + low quality (hardest case)
    out = os.path.join(output_dir, f"{base_name}_combined.mp4")
    ok = run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", (
            f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},"
            "eq=brightness=0.08:contrast=1.2:saturation=1.3"
        ),
        "-c:v", "libx264", "-crf", "32", "-preset", "fast", "-an", out
    ])
    results.append(("combined", ok))

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    videos = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    ])

    if not videos:
        print(f"No videos found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(videos)} banned videos. Generating transformations...\n")

    total_success = 0
    total_fail = 0

    for i, video in enumerate(videos, 1):
        input_path = os.path.join(INPUT_DIR, video)
        base_name = os.path.splitext(video)[0]

        print(f"[{i}/{len(videos)}] {video}")

        results = apply_transformations(input_path, base_name, OUTPUT_DIR)

        for transform, ok in results:
            if ok:
                total_success += 1
            else:
                total_fail += 1
                print(f"  FAILED: {transform}")

    print(f"\nDone. {total_success} succeeded, {total_fail} failed.")
    print(f"Re-uploads saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()