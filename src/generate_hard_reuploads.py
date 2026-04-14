"""
Generate harder modified re-uploads from banned videos.
Includes 6 transformations that simulate realistic TikTok-style repost behavior.
"""

import json
import os
import subprocess
import sys


INPUT_DIR = "data/banned"
OUTPUT_DIR = "data/reuploads_hard"


def run_ffmpeg(cmd):
    """Run an FFmpeg command silently. Returns True on success."""
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def get_video_info(path):
    """Get video width, height, and duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)

    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    duration = float(info["format"].get("duration", 0.0))
    return width, height, duration


def transform_heavy_text(input_path, output_path):
    """Large top text + diagonal watermark style text overlays."""
    vf = (
        "drawtext=text='follow for more':"
        "fontsize=54:fontcolor=white@0.96:"
        "box=1:boxcolor=black@0.45:boxborderw=14:"
        "x=(w-text_w)/2:y=40,"
        "drawtext=text='@username':"
        "fontsize=50:fontcolor=white@0.55:"
        "box=1:boxcolor=black@0.30:boxborderw=10:"
        "x=w*0.18:y=h*0.28,"
        "drawtext=text='@username':"
        "fontsize=50:fontcolor=white@0.55:"
        "box=1:boxcolor=black@0.30:boxborderw=10:"
        "x=w*0.34:y=h*0.45,"
        "drawtext=text='@username':"
        "fontsize=50:fontcolor=white@0.55:"
        "box=1:boxcolor=black@0.30:boxborderw=10:"
        "x=w*0.50:y=h*0.62"
    )
    return run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-an", output_path,
    ])


def transform_reaction(input_path, output_path, width, height, duration):
    """Picture-in-picture reaction layout with gray lower area."""
    pip_w = int(width * 0.60)
    pip_h = int(height * 0.60)
    top_y = int(height * 0.04)
    bottom_bar_y = int(height * 0.60)
    bottom_bar_h = height - bottom_bar_y
    safe_duration = max(duration, 1.0)

    filter_complex = (
        f"color=c=black:s={width}x{height}:d={safe_duration}[bg];"
        f"[bg]drawbox=x=0:y={bottom_bar_y}:w={width}:h={bottom_bar_h}:color=gray@1.0:t=fill[layout];"
        f"[0:v]scale={pip_w}:{pip_h}[pip];"
        f"[layout][pip]overlay=(W-w)/2:{top_y}"
    )

    return run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-c:a", "copy",
        output_path,
    ])


def transform_grayscale(input_path, output_path):
    """Black and white conversion."""
    return run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", "format=gray,format=yuv420p",
        "-c:v", "libx264", "-preset", "fast", "-an", output_path,
    ])


def transform_heavy_crop_zoom(input_path, output_path):
    """Crop center 40% and scale back to original size."""
    vf = "crop=iw*0.4:ih*0.4:(iw-iw*0.4)/2:(ih-ih*0.4)/2,scale=iw:ih"
    return run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-an", output_path,
    ])


def transform_vintage_filter(input_path, output_path):
    """Vintage style: warm shift, lower saturation, slight blur."""
    vf = (
        "colorbalance=rs=0.35:gs=0.08:bs=-0.20:rm=0.20:gm=0.10:"
        "bm=-0.10:rh=0.08:gh=0.04:bh=-0.05,"
        "eq=saturation=0.6,"
        "gblur=sigma=1.5"
    )
    return run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-an", output_path,
    ])


def transform_heavy_combined(input_path, output_path):
    """Mirror + grayscale + heavy crop + text + low quality encode."""
    vf = (
        "hflip,"
        "crop=iw*0.5:ih*0.5:(iw-iw*0.5)/2:(ih-ih*0.5)/2,"
        "scale=iw:ih,"
        "format=gray,format=yuv420p,"
        "drawtext=text='follow for more':"
        "fontsize=46:fontcolor=white@0.95:"
        "box=1:boxcolor=black@0.45:boxborderw=12:"
        "x=(w-text_w)/2:y=36"
    )
    return run_ffmpeg([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "38", "-preset", "fast", "-an", output_path,
    ])


def apply_hard_transformations(input_path, base_name, output_dir):
    """Apply the 6 hard transformations."""
    width, height, duration = get_video_info(input_path)
    results = []

    out = os.path.join(output_dir, f"{base_name}_heavytext.mp4")
    results.append(("heavy_text", transform_heavy_text(input_path, out)))

    out = os.path.join(output_dir, f"{base_name}_reaction.mp4")
    results.append(("reaction", transform_reaction(input_path, out, width, height, duration)))

    out = os.path.join(output_dir, f"{base_name}_grayscale.mp4")
    results.append(("grayscale", transform_grayscale(input_path, out)))

    out = os.path.join(output_dir, f"{base_name}_heavycropzoom.mp4")
    results.append(("heavy_crop_zoom", transform_heavy_crop_zoom(input_path, out)))

    out = os.path.join(output_dir, f"{base_name}_vintage.mp4")
    results.append(("vintage_filter", transform_vintage_filter(input_path, out)))

    out = os.path.join(output_dir, f"{base_name}_heavycombined.mp4")
    results.append(("heavy_combined", transform_heavy_combined(input_path, out)))

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    videos = sorted(
        f
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
    )

    if not videos:
        print(f"No videos found in {INPUT_DIR}")
        sys.exit(1)

    print(f"Found {len(videos)} banned videos. Generating hard transformations...\n")

    total_success = 0
    total_fail = 0

    for i, video in enumerate(videos, 1):
        input_path = os.path.join(INPUT_DIR, video)
        base_name = os.path.splitext(video)[0]

        print(f"[{i}/{len(videos)}] {video}")

        try:
            results = apply_hard_transformations(input_path, base_name, OUTPUT_DIR)
        except Exception:
            total_fail += 6
            print("  FAILED: unable to process video")
            continue

        for transform, ok in results:
            if ok:
                total_success += 1
            else:
                total_fail += 1
                print(f"  FAILED: {transform}")

    print(f"\nDone. {total_success} succeeded, {total_fail} failed.")
    print(f"Hard re-uploads saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
