#!/bin/bash
# Evaluate GRPO checkpoint by first merging FSDP shards to HuggingFace format

set -e

# Default parameters
CHECKPOINT_DIR="saves/qwen2_5vl-3b/grpo/adaptive_reasoning/global_step_90"
CHECKPOINT_NAME="step_90"
PROJECT_NAME="verl_grpo_adaptive_reasoning"
EXPERIMENT_NAME="qwen2_5vl_3b_adaptive"
VAL_DATA="test/test.parquet"
OUTPUT_DIR="evaluation_results"
MAX_SAMPLES=""
TEMPERATURE=0.8
TOP_P=0.95

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --checkpoint_name)
            CHECKPOINT_NAME="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="--max_samples $2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --checkpoint_dir <path>   Path to GRPO checkpoint directory (default: saves/qwen2_5vl-3b/grpo/adaptive_reasoning/global_step_90)"
            echo "  --checkpoint_name <name>  Checkpoint name for output (default: step_90)"
            echo "  --max_samples <num>       Maximum samples to evaluate (default: all)"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Paths
ACTOR_DIR="$CHECKPOINT_DIR/actor"
MERGED_DIR="$CHECKPOINT_DIR/actor_merged"

echo "========================================"
echo "GRPO Checkpoint Evaluation"
echo "========================================"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Checkpoint Name: $CHECKPOINT_NAME"
echo "Output: $OUTPUT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME/$CHECKPOINT_NAME"
echo "========================================"

# Step 1: Check if merged model already exists
if [ -d "$MERGED_DIR" ] && [ -f "$MERGED_DIR/config.json" ]; then
    echo ""
    echo "✓ Merged HuggingFace model already exists at $MERGED_DIR"
    echo "  Skipping merge step..."
else
    echo ""
    echo "Step 1: Merging FSDP checkpoint to HuggingFace format..."
    echo "  Source: $ACTOR_DIR"
    echo "  Target: $MERGED_DIR"

    cd verl
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "../$ACTOR_DIR" \
        --target_dir "../$MERGED_DIR"
    cd ..

    echo "✓ Merge completed!"
fi

# Step 2: Run evaluation
echo ""
echo "Step 2: Running evaluation on validation set..."
python evaluate_checkpoint.py \
    --checkpoint "$MERGED_DIR" \
    --val_data "$VAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --project_name "$PROJECT_NAME" \
    --experiment_name "$EXPERIMENT_NAME" \
    --checkpoint_name "$CHECKPOINT_NAME" \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    $MAX_SAMPLES

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR/$PROJECT_NAME/$EXPERIMENT_NAME/$CHECKPOINT_NAME"
echo ""
echo "Files:"
echo "  - evaluation_summary.txt (human-readable summary)"
echo "  - evaluation_detailed.json (detailed results)"
echo "  - evaluation_results.csv (CSV format)"
echo "========================================"
