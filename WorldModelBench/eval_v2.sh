#!/bin/bash
# 8-GPU parallel evaluation for WorldModelBench v2
source /home/user1/miniconda3/etc/profile.d/conda.sh
conda activate vila

cd "$(dirname "$0")"

VIDEO_DIR="${1:-outputs}"
TOTAL="${2:-350}"
MODEL_NAME="${3:-Wan2.2-TI2V-5B}"

NUM_GPUS=8
PER_GPU=$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
JUDGE="vila-ewm-qwen2-1.5b"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="evaloutputs/eval_v2_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "Run dir: $RUN_DIR"
echo "Launching $NUM_GPUS parallel evaluations ($PER_GPU videos each)..."

pids=()
for gpu in $(seq 0 $((NUM_GPUS - 1))); do
    start=$((gpu * PER_GPU))
    end=$(( (gpu + 1) * PER_GPU ))
    [ $end -gt $TOTAL ] && end=$TOTAL

    echo "  GPU $gpu: videos [$start, $end)"
    CUDA_VISIBLE_DEVICES=$gpu python evaluation_v2.py \
        --model_name "$MODEL_NAME" \
        --video_dir "$VIDEO_DIR" \
        --judge "$JUDGE" \
        --cot \
        --save_name "${RUN_DIR}/shard_gpu${gpu}" \
        --start $start --end $end \
        > "${RUN_DIR}/gpu${gpu}.log" 2>&1 &
    pids+=($!)
done

echo "Waiting for all GPUs to finish..."
failed=0
for i in "${!pids[@]}"; do
    wait ${pids[$i]}
    code=$?
    if [ $code -ne 0 ]; then
        echo "  GPU $i FAILED (exit $code), see ${RUN_DIR}/gpu${i}.log"
        failed=1
    else
        echo "  GPU $i done"
    fi
done

if [ $failed -ne 0 ]; then
    echo "Some workers failed. Check logs before merging."
    exit 1
fi

OUTPUT="${RUN_DIR}/results.json"
echo "All done. Merging results..."
python merge_eval_v2.py \
    --prefix "${RUN_DIR}/shard" \
    --num_gpus $NUM_GPUS \
    --output "$OUTPUT"

echo "Final results saved to $OUTPUT"
