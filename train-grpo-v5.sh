#!/bin/bash
# Training script for Stage 2 GRPO with Adaptive Reasoning V5
# Features: Reversed Format Bonus, Error Penalties, NO Diversity Scaling
# Reward Design: Type 1 > Type 2 > Type 3, with error penalties for risky formats

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default parameters
SFT_MODEL=""
DATA_DIR="grpo_data"
OUTPUT_DIR="saves/qwen2_5vl-3b/grpo/adaptive_reasoning_9k_v5"
NUM_GPUS=8
ENGINE="vllm"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sft_model)
            SFT_MODEL="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --sft_model <path>     Path to SFT checkpoint (required)"
            echo "  --data_dir <path>      GRPO data directory (default: grpo_data)"
            echo "  --output_dir <path>    Output directory (default: saves/qwen2_5vl-3b/grpo/adaptive_reasoning_9k_v5)"
            echo "  --gpus <num>           Number of GPUs (default: 8)"
            echo "  --engine <vllm|sglang> Inference engine (default: vllm)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Auto-detect SFT model if not specified
if [ -z "$SFT_MODEL" ]; then
    # Try to find the latest SFT checkpoint
    if [ -d "LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_9k" ]; then
        SFT_MODEL="LLaMA-Factory/saves/qwen2_5vl-3b/full/sft_9k"
        echo -e "${YELLOW}Auto-detected SFT model: $SFT_MODEL${NC}"
    else
        echo -e "${RED}ERROR: SFT model not specified and no default found${NC}"
        echo "Please specify --sft_model or run SFT training first"
        exit 1
    fi
fi

# Validate paths
if [ ! -d "$SFT_MODEL" ]; then
    echo -e "${RED}ERROR: SFT model not found at $SFT_MODEL${NC}"
    exit 1
fi

if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo -e "${RED}ERROR: GRPO training data not found at $DATA_DIR/train.parquet${NC}"
    echo "Please run: python prepare_grpo_data.py"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GRPO Training - Adaptive Reasoning V5${NC}"
echo -e "${GREEN}Efficient Reasoning with Error Penalties${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "SFT Model: ${YELLOW}$SFT_MODEL${NC}"
echo -e "Data Directory: ${YELLOW}$DATA_DIR${NC}"
echo -e "Output Directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Number of GPUs: ${YELLOW}$NUM_GPUS${NC}"
echo -e "Inference Engine: ${YELLOW}$ENGINE${NC}"
echo -e "${BLUE}Reward Manager: ${YELLOW}adaptive_reasoning_v5${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "${CYAN}=== V5 Reward Design ===${NC}"
echo -e "${BLUE}Format Bonuses (when CORRECT):${NC}"
echo -e "  ${GREEN}Type 1 (direct):${NC}       +0.2 (highest reward)"
echo -e "  ${YELLOW}Type 2 (perception):${NC}  +0.1 (medium reward)"
echo -e "  ${CYAN}Type 3 (full):${NC}         +0.0 (baseline)"
echo -e ""
echo -e "${BLUE}Error Penalties (when INCORRECT):${NC}"
echo -e "  ${RED}Type 1 (direct):${NC}       -0.5 (high risk)"
echo -e "  ${YELLOW}Type 2 (perception):${NC}  -0.3 (medium risk)"
echo -e "  ${GREEN}Type 3 (full):${NC}         +0.0 (safe baseline)"
echo -e ""
echo -e "${BLUE}Key Changes from V4:${NC}"
echo -e "  ${GREEN}✓${NC} Reversed format bonus (Type 1 > Type 2 > Type 3)"
echo -e "  ${GREEN}✓${NC} Added error penalties for Type 1 and Type 2"
echo -e "  ${GREEN}✓${NC} Removed diversity scaling (simplified)"
echo -e "  ${GREEN}✓${NC} Encourages efficient reasoning"
echo -e ""
echo -e "${BLUE}Expected Model Behavior:${NC}"
echo -e "  • Simple questions → Type 1 (if confident)"
echo -e "  • Medium questions → Type 2 or Type 3 (balance)"
echo -e "  • Complex questions → Type 3 (safe baseline)"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "${BLUE}TensorBoard Metrics:${NC}"
echo -e "  • Format distribution (Type 1/2/3 ratios)"
echo -e "  • Per-type accuracy and length"
echo -e "  • Reward component breakdown"
echo -e "  • Overall accuracy"
echo -e ""
echo -e "${BLUE}View metrics:${NC}"
echo -e "  ${YELLOW}tensorboard --logdir $OUTPUT_DIR${NC}"
echo -e "${GREEN}========================================${NC}"

# Install verl if needed
if ! python -c "import verl" 2>/dev/null; then
    echo -e "${YELLOW}Installing verl...${NC}"
    cd verl
    pip install -e .
    cd ..
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/verl"

# NCCL settings for robustness
export NCCL_TIMEOUT=7200  # 2 hours timeout (default is 30 minutes)
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22  # Increase InfiniBand timeout
export NCCL_BLOCKING_WAIT=1  # Use blocking wait for more stable communication

# PyTorch distributed timeout (CRITICAL for fixing the all_gather timeout)
export TORCH_DISTRIBUTED_TIMEOUT=7200  # 2 hours in seconds
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # Better error reporting

# VLLM settings for stability
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo -e "${GREEN}Starting GRPO training with V5 reward function...${NC}"

cd verl

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=../$DATA_DIR/train.parquet \
    data.val_files=../$DATA_DIR/val.parquet \
    data.train_batch_size=256 \
    data.val_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.image_key=images \
    data.prompt_key=prompt \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=../$SFT_MODEL \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=adaptive_reasoning_v5 \
    +reward_model.type1_format_bonus=0.2 \
    +reward_model.type2_format_bonus=0.1 \
    +reward_model.type3_format_bonus=0.0 \
    +reward_model.type1_error_penalty=-0.5 \
    +reward_model.type2_error_penalty=-0.3 \
    +reward_model.type3_error_penalty=0.0 \
    +reward_model.length_threshold=300 \
    +reward_model.ideal_length=300.0 \
    +reward_model.min_scalar=0.3 \
    +reward_model.normalize_answers=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_adaptive_reasoning_9k_v5' \
    trainer.experiment_name='qwen2_5vl_3b_adaptive_9k_v5' \
    trainer.default_local_dir=../$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.resume_mode=auto \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.total_epochs=1

cd ..

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Training completed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Model saved to: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e ""
echo -e "${BLUE}View training metrics in TensorBoard:${NC}"
echo -e "  ${YELLOW}tensorboard --logdir $OUTPUT_DIR${NC}"
echo -e ""
echo -e "${BLUE}Available metrics:${NC}"
echo -e "  • ${GREEN}format/${NC}type{1,2,3}_ratio - Format distribution"
echo -e "  • ${GREEN}format/${NC}type{1,2,3}_correct_rate - Per-type accuracy"
echo -e "  • ${GREEN}format/${NC}type{1,2,3}_avg_length - Per-type length"
echo -e "  • ${GREEN}reward/${NC}base_mean - Base reward (correctness)"
echo -e "  • ${GREEN}reward/${NC}format_bonus_mean - Format bonus/penalty"
echo -e "  • ${GREEN}reward/${NC}length_scalar_mean - Length penalty"
echo -e "  • ${GREEN}accuracy/${NC}overall - Overall accuracy"
echo -e ""
echo -e "${CYAN}V5 Changes:${NC}"
echo -e "  • Reward manager changed to adaptive_reasoning_v5"
echo -e "  • Format bonus reversed (Type 1 > Type 2 > Type 3)"
echo -e "  • Error penalties added for Type 1 (-0.5) and Type 2 (-0.3)"
echo -e "  • Diversity scaling removed (simplified)"
echo -e ""
echo -e "${BLUE}For more information, see:${NC}"
echo -e "  ${YELLOW}reward_functions/REWARD_V4_CHANGES.md${NC}"
echo -e "  ${YELLOW}QUICK_START_METRICS.md${NC}"
echo -e "${GREEN}========================================${NC}"
