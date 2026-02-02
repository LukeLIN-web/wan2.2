"""Concatenate videos in continuous_outputs/ directory in order."""

import glob
import os
import subprocess
import tempfile

INPUT_DIR = "continuous_outputs/20260201_180527"
OUTPUT_FILE = "continuous_outputs/20260201_180527/merged.mp4"


def concat_videos(input_dir, output_file):
    """
    Concatenate all video_*.mp4 files in input_dir into a single output file.
    Uses ffmpeg concat demuxer for fast, lossless concatenation.
    """
    videos = sorted(glob.glob(os.path.join(input_dir, "video_*.mp4")))
    if not videos:
        print(f"No video_*.mp4 files found in {input_dir}/")
        return None

    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(f"  {v}")

    # Write ffmpeg concat demuxer file list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for v in videos:
            f.write(f"file '{os.path.abspath(v)}'\n")
        list_path = f.name

    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_path, "-c", "copy", output_file,
        ]
        print(f"\nRunning: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"\nDone: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg concat failed: {e}")
        return None
    finally:
        os.unlink(list_path)


def main():
    concat_videos(INPUT_DIR, OUTPUT_FILE)


if __name__ == "__main__":
    main()
