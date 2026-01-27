#!/bin/bash

# GPU device list (space-separated)
DEVICE_LIST=(1 3)

# Video file list (space-separated)
VIDEO_LIST=(
    "generatedvideo/dog.mp4"
    "generatedvideo/road.mp4"
)

# Checkpoint directory
CKPT_DIR="./Wan2.2-I2V-A14B"

conda activate wan

# Process videos in parallel
video_idx=0
for video in "${VIDEO_LIST[@]}"; do
    # Select GPU in round-robin fashion
    gpu_idx=$((video_idx % ${#DEVICE_LIST[@]}))
    gpu_id=${DEVICE_LIST[$gpu_idx]}
    
    echo "Processing $video on GPU $gpu_id"
    
    # Run in background
    CUDA_VISIBLE_DEVICES=$gpu_id python generate.py \
        --task v2v-14B \
        --video "$video" \
        --ckpt_dir "$CKPT_DIR" \
        > "logs/gpu${gpu_id}_$(basename $video .mp4).log" 2>&1 &
    
    video_idx=$((video_idx + 1))
    
    # Optional: Add a small delay to avoid race conditions
    sleep 2
done

# Wait for all background processes to complete
echo "Waiting for all processes to complete..."
wait

echo "All videos processed!"
