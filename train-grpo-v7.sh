#!/bin/bash
# Training script for Stage 2 GRPO with Adaptive Reasoning V7
# Features: Type Ratio Control to prevent format over-exploitation
# Key Innovation: Dynamic ratio-based penalty that activates after warm-up
# Automatically balances type distribution towards target ratios

set -e

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Default parameters
SFT_MODEL=""
DATA_DIR="grpo_data_all"
OUTPUT_DIR="saves/qwen2_5vl-3b/grpo/adaptive_reasoning_9k_v7"
NUM_GPUS=8
ENGINE="vllm"

# V7 Ratio control parameters (can be overridden via command line)
ENABLE_RATIO_PENALTY=true
RATIO_PENALTY_START_STEP=50
TARGET_TYPE1_RATIO=0.6
TARGET_TYPE2_RATIO=0.2
TARGET_TYPE3_RATIO=0.2
RATIO_TOLERANCE=0.10
RATIO_PENALTY_MIN_SCALAR=0.3
RATIO_WINDOW_SIZE=128

# V6 Decay parameters (inherited, can still be used)
ENABLE_DECAY=false  # Usually not needed with ratio control
DECAY_STRATEGY="linear"  # linear, exponential, or cosine
DECAY_START_STEP=10
DECAY_END_STEP=40
TYPE1_BONUS_INITIAL=0.2
TYPE1_BONUS_MIN=0.05
DECAY_RATE=0.95  # For exponential decay

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
        --enable_ratio_penalty)
            ENABLE_RATIO_PENALTY="$2"
            shift 2
            ;;
        --ratio_penalty_start_step)
            RATIO_PENALTY_START_STEP="$2"
            shift 2
            ;;
        --target_type1_ratio)
            TARGET_TYPE1_RATIO="$2"
            shift 2
            ;;
        --target_type2_ratio)
            TARGET_TYPE2_RATIO="$2"
            shift 2
            ;;
        --target_type3_ratio)
            TARGET_TYPE3_RATIO="$2"
            shift 2
            ;;
        --ratio_tolerance)
            RATIO_TOLERANCE="$2"
            shift 2
            ;;
        --ratio_penalty_min_scalar)
            RATIO_PENALTY_MIN_SCALAR="$2"
            shift 2
            ;;
        --ratio_window_size)
            RATIO_WINDOW_SIZE="$2"
            shift 2
            ;;
        --enable_decay)
            ENABLE_DECAY="$2"
            shift 2
            ;;
        --decay_strategy)
            DECAY_STRATEGY="$2"
            shift 2
            ;;
        --decay_start_step)
            DECAY_START_STEP="$2"
            shift 2
            ;;
        --decay_end_step)
            DECAY_END_STEP="$2"
            shift 2
            ;;
        --type1_bonus_initial)
            TYPE1_BONUS_INITIAL="$2"
            shift 2
            ;;
        --type1_bonus_min)
            TYPE1_BONUS_MIN="$2"
            shift 2
            ;;
        --decay_rate)
            DECAY_RATE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Basic Options:"
            echo "  --sft_model <path>         Path to SFT checkpoint (required)"
            echo "  --data_dir <path>          GRPO data directory (default: grpo_data_all)"
            echo "  --output_dir <path>        Output directory (default: saves/qwen2_5vl-3b/grpo/adaptive_reasoning_9k_v7)"
            echo "  --gpus <num>               Number of GPUs (default: 8)"
            echo "  --engine <vllm|sglang>     Inference engine (default: vllm)"
            echo ""
            echo "V7 Ratio Control Options:"
            echo "  --enable_ratio_penalty <true|false>    Enable ratio control (default: true)"
            echo "  --ratio_penalty_start_step <num>       Step to activate ratio penalty (default: 60)"
            echo "  --target_type1_ratio <float>           Target ratio for Type1 (default: 0.3)"
            echo "  --target_type2_ratio <float>           Target ratio for Type2 (default: 0.4)"
            echo "  --target_type3_ratio <float>           Target ratio for Type3 (default: 0.3)"
            echo "  --ratio_tolerance <float>              Tolerance before penalty (default: 0.15)"
            echo "  --ratio_penalty_min_scalar <float>     Min scalar when ratio deviates (default: 0.5)"
            echo "  --ratio_window_size <num>              Window size for tracking (default: 256)"
            echo ""
            echo "V6 Decay Options (optional, inherited from V6):"
            echo "  --enable_decay <true|false>        Enable Type1 bonus decay (default: false)"
            echo "  --decay_strategy <strategy>        Decay strategy: linear, exponential, cosine (default: linear)"
            echo "  --decay_start_step <num>           Step to start decay (default: 10)"
            echo "  --decay_end_step <num>             Step to finish decay (default: 40)"
            echo "  --type1_bonus_initial <float>      Initial Type1 bonus (default: 0.2)"
            echo "  --type1_bonus_min <float>          Minimum Type1 bonus after decay (default: 0.05)"
            echo "  --decay_rate <float>               Decay rate for exponential strategy (default: 0.95)"
            echo ""
            echo "  --help                             Show this help message"
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
echo -e "${GREEN}GRPO Training - Adaptive Reasoning V7${NC}"
echo -e "${MAGENTA}NEW: Type Ratio Control Mechanism${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "SFT Model: ${YELLOW}$SFT_MODEL${NC}"
echo -e "Data Directory: ${YELLOW}$DATA_DIR${NC}"
echo -e "Output Directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Number of GPUs: ${YELLOW}$NUM_GPUS${NC}"
echo -e "Inference Engine: ${YELLOW}$ENGINE${NC}"
echo -e "${BLUE}Reward Manager: ${YELLOW}adaptive_reasoning (V7)${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "${CYAN}=== V7 Ratio Control Mechanism ===${NC}"
echo -e "${MAGENTA}Innovation:${NC} Dynamic penalty based on type distribution"
echo -e "${MAGENTA}Goal:${NC} Prevent any single type from dominating"
echo -e ""
echo -e "${BLUE}Target Type Distribution:${NC}"
echo -e "  ${GREEN}Type 1 (direct):${NC}       ${TARGET_TYPE1_RATIO} (${CYAN}fast, efficient${NC})"
echo -e "  ${YELLOW}Type 2 (perception):${NC}  ${TARGET_TYPE2_RATIO} (${CYAN}balanced${NC})"
echo -e "  ${CYAN}Type 3 (full):${NC}         ${TARGET_TYPE3_RATIO} (${CYAN}thorough${NC})"
echo -e ""
echo -e "${BLUE}Ratio Control Parameters:${NC}"
echo -e "  ${CYAN}Enabled:${NC}              ${ENABLE_RATIO_PENALTY}"
echo -e "  ${CYAN}Activation Step:${NC}      ${RATIO_PENALTY_START_STEP} ${MAGENTA}(warm-up period)${NC}"
echo -e "  ${CYAN}Tolerance:${NC}            ±${RATIO_TOLERANCE} ${MAGENTA}(allowed deviation)${NC}"
echo -e "  ${CYAN}Min Scalar:${NC}           ${RATIO_PENALTY_MIN_SCALAR} ${MAGENTA}(max penalty strength)${NC}"
echo -e "  ${CYAN}Window Size:${NC}          ${RATIO_WINDOW_SIZE} samples"
echo -e ""
echo -e "${BLUE}How It Works:${NC}"
echo -e "  1. ${GREEN}Tracks${NC} recent type distribution in sliding window"
echo -e "  2. ${YELLOW}Compares${NC} current ratio to target ratio for each type"
echo -e "  3. ${RED}Penalizes${NC} over-used types with scalar reduction"
echo -e "  4. ${CYAN}Encourages${NC} under-used types (no penalty)"
echo -e ""
echo -e "${MAGENTA}=== Example Scenario ===${NC}"
echo -e "${YELLOW}Before Ratio Control (Step 50):${NC}"
echo -e "  Type 1: ${RED}80%${NC} (way over target!)"
echo -e "  Type 2: ${YELLOW}15%${NC} (below target)"
echo -e "  Type 3: ${YELLOW}5%${NC} (below target)"
echo -e ""
echo -e "${GREEN}After Ratio Control Activates (Step 60+):${NC}"
echo -e "  ${RED}Type 1 gets penalty scalar${NC} → reward reduced"
echo -e "  ${GREEN}Type 2 and 3 encouraged${NC} → full reward"
echo -e "  ${CYAN}Distribution gradually balances${NC}"
echo -e ""
echo -e "${BLUE}Expected Final Distribution (Step 100+):${NC}"
echo -e "  Type 1: ${GREEN}~30%${NC} ✓"
echo -e "  Type 2: ${GREEN}~40%${NC} ✓"
echo -e "  Type 3: ${GREEN}~30%${NC} ✓"
echo -e ""
echo -e "${MAGENTA}=== V6 Decay Configuration (Optional) ===${NC}"
echo -e "  ${BLUE}Enable Decay:${NC}         ${ENABLE_DECAY}"
if [ "$ENABLE_DECAY" = "true" ]; then
    echo -e "  ${BLUE}Decay Strategy:${NC}       ${DECAY_STRATEGY}"
    echo -e "  ${BLUE}Decay Start Step:${NC}     ${DECAY_START_STEP}"
    echo -e "  ${BLUE}Decay End Step:${NC}       ${DECAY_END_STEP}"
    echo -e "  ${BLUE}Initial Bonus:${NC}        ${TYPE1_BONUS_INITIAL}"
    echo -e "  ${BLUE}Minimum Bonus:${NC}        ${TYPE1_BONUS_MIN}"
    if [ "$DECAY_STRATEGY" = "exponential" ]; then
        echo -e "  ${BLUE}Decay Rate:${NC}           ${DECAY_RATE}"
    fi
fi
echo -e ""
echo -e "${CYAN}=== Key Advantages of V7 ===${NC}"
echo -e "  ${GREEN}✓${NC} Automatic type balancing (no manual tuning needed)"
echo -e "  ${GREEN}✓${NC} Prevents single-format exploitation"
echo -e "  ${GREEN}✓${NC} Adaptive to actual model behavior"
echo -e "  ${GREEN}✓${NC} Smooth transition with warm-up period"
echo -e "  ${GREEN}✓${NC} All V6 features preserved (bonus decay, error penalties, etc.)"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "${BLUE}TensorBoard Metrics:${NC}"
echo -e "  ${GREEN}Format Metrics:${NC}"
echo -e "    • format/type{1,2,3}_ratio - Actual type distribution"
echo -e "    • format/type{1,2,3}_correct_rate - Per-type accuracy"
echo -e "    • format/type{1,2,3}_avg_length - Per-type length"
echo -e ""
echo -e "  ${MAGENTA}V7 Ratio Control Metrics:${NC}"
echo -e "    • ratio_control/enabled - Whether ratio penalty is active"
echo -e "    • ratio_control/type{1,2,3}_scalar - Current penalty scalars"
echo -e "    • ratio_control/target_type{1,2,3}_ratio - Target ratios"
echo -e "    • ratio_control/window_type{1,2,3}_ratio - Current window ratios"
echo -e ""
echo -e "  ${GREEN}Reward Components:${NC}"
echo -e "    • reward/base_mean - Base correctness reward"
echo -e "    • reward/format_bonus_mean - Format bonus/penalty"
echo -e "    • reward/length_scalar_mean - Length penalty scalar"
echo -e "    • reward/ratio_scalar_mean - Ratio penalty scalar (NEW!)"
echo -e "    • reward/total_mean - Final reward"
echo -e ""
echo -e "  ${GREEN}Overall:${NC}"
echo -e "    • accuracy/overall - Overall accuracy"
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

# wandb api key
export WANDB_API_KEY="d669fcba5506b573c26943e6b7904881365b6012"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo -e "${GREEN}Starting GRPO training with V7 ratio control...${NC}"

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
    reward_model.reward_manager=adaptive_reasoning \
    +reward_model.reward_kwargs.reward_version=v7 \
    +reward_model.reward_kwargs.correct_reward=1.0 \
    +reward_model.reward_kwargs.incorrect_reward=0.0 \
    +reward_model.reward_kwargs.type1_format_bonus=$TYPE1_BONUS_INITIAL \
    +reward_model.reward_kwargs.type2_format_bonus=0.3 \
    +reward_model.reward_kwargs.type3_format_bonus=0.15 \
    +reward_model.reward_kwargs.type1_error_penalty=-0.1 \
    +reward_model.reward_kwargs.type2_error_penalty=0.0 \
    +reward_model.reward_kwargs.type3_error_penalty=0.0 \
    +reward_model.reward_kwargs.length_threshold=200 \
    +reward_model.reward_kwargs.ideal_length=200.0 \
    +reward_model.reward_kwargs.min_scalar=0.6 \
    +reward_model.reward_kwargs.enable_bonus_decay=$ENABLE_DECAY \
    +reward_model.reward_kwargs.decay_strategy=$DECAY_STRATEGY \
    +reward_model.reward_kwargs.decay_start_step=$DECAY_START_STEP \
    +reward_model.reward_kwargs.decay_end_step=$DECAY_END_STEP \
    +reward_model.reward_kwargs.type1_bonus_min=$TYPE1_BONUS_MIN \
    +reward_model.reward_kwargs.decay_rate=$DECAY_RATE \
    +reward_model.reward_kwargs.enable_ratio_penalty=$ENABLE_RATIO_PENALTY \
    +reward_model.reward_kwargs.ratio_penalty_start_step=$RATIO_PENALTY_START_STEP \
    +reward_model.reward_kwargs.target_type1_ratio=$TARGET_TYPE1_RATIO \
    +reward_model.reward_kwargs.target_type2_ratio=$TARGET_TYPE2_RATIO \
    +reward_model.reward_kwargs.target_type3_ratio=$TARGET_TYPE3_RATIO \
    +reward_model.reward_kwargs.ratio_tolerance=$RATIO_TOLERANCE \
    +reward_model.reward_kwargs.ratio_penalty_min_scalar=$RATIO_PENALTY_MIN_SCALAR \
    +reward_model.reward_kwargs.ratio_window_size=$RATIO_WINDOW_SIZE \
    +reward_model.reward_kwargs.normalize_answers=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard","wandb"]' \
    trainer.project_name='verl_grpo_adaptive_reasoning_9k_v7' \
    trainer.experiment_name='qwen2_5vl_3b_adaptive_9k_v7_1' \
    +trainer.log_rollout_to_wandb=True \
    trainer.default_local_dir=../$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.resume_mode=auto \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
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
echo -e "${BLUE}V7 Success Criteria:${NC}"
echo -e "  ${GREEN}✓${NC} Type distribution converges to target ratios"
echo -e "  ${GREEN}✓${NC} ratio_control/type{1,2,3}_scalar values shown in metrics"
echo -e "  ${GREEN}✓${NC} No single type dominates (e.g., not 80%+ of any type)"
echo -e "  ${GREEN}✓${NC} reward/total_mean increases over training"
echo -e "  ${GREEN}✓${NC} Overall accuracy stable or improving"
echo -e ""
echo -e "${BLUE}Expected Training Dynamics:${NC}"
echo -e "  ${GREEN}Early Steps (1-60):${NC}"
echo -e "    • Ratio control inactive (warm-up period)"
echo -e "    • Model explores different formats freely"
echo -e "    • May see imbalanced distribution initially"
echo -e ""
echo -e "  ${YELLOW}Mid Training (60-100):${NC}"
echo -e "    • Ratio control activates at step ${RATIO_PENALTY_START_STEP}"
echo -e "    • Over-used types start getting penalized"
echo -e "    • Distribution gradually shifts towards targets"
echo -e ""
echo -e "  ${CYAN}Late Training (100+):${NC}"
echo -e "    • Distribution stabilizes near target ratios"
echo -e "    • Model learns adaptive format selection"
echo -e "    • Balanced performance across all types"
echo -e ""
echo -e "${BLUE}Monitor These Metrics:${NC}"
echo -e "  ${MAGENTA}CRITICAL:${NC}"
echo -e "    • ${GREEN}ratio_control/window_type{1,2,3}_ratio${NC} - Should approach targets"
echo -e "    • ${GREEN}reward/ratio_scalar_mean${NC} - Should be close to 1.0 when balanced"
echo -e ""
echo -e "  ${YELLOW}Important:${NC}"
echo -e "    • format/type{1,2,3}_ratio - Batch-level distribution"
echo -e "    • reward/total_mean - Must increase steadily"
echo -e "    • accuracy/overall - Should remain stable or improve"
echo -e ""
echo -e "${BLUE}For more information, see:${NC}"
echo -e "  ${YELLOW}reward_functions/adaptive_reasoning_reward_v7.py${NC}"
echo -e "  ${YELLOW}reward_functions/adaptive_reasoning_reward_manager.py${NC}"
echo -e "${GREEN}========================================${NC}"
