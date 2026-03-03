#!/bin/bash
# Training script for Stage 2 GRPO with Adaptive Reasoning V6.4
# Features: BALANCED Reward Design to counter length penalty dominance
# Key Fix: Type 2/3 get higher bonuses to compensate for length penalty
# Innovation: Prevents Type 1 over-exploitation even without decay

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
OUTPUT_DIR="saves/qwen2_5vl-3b/grpo/adaptive_reasoning_9k_v6_8"
NUM_GPUS=8
ENGINE="vllm"

# V6.4 Decay parameters (can be overridden via command line)
ENABLE_DECAY=false  # Default to false - balanced rewards should work without decay
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
            echo "  --output_dir <path>        Output directory (default: saves/qwen2_5vl-3b/grpo/adaptive_reasoning_9k_v6_4_balanced)"
            echo "  --gpus <num>               Number of GPUs (default: 8)"
            echo "  --engine <vllm|sglang>     Inference engine (default: vllm)"
            echo ""
            echo "V6.4 Decay Options:"
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
echo -e "${GREEN}GRPO Training - Adaptive Reasoning V6.4${NC}"
echo -e "${MAGENTA}BALANCED Reward Design (Fixes Length Penalty Dominance)${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "SFT Model: ${YELLOW}$SFT_MODEL${NC}"
echo -e "Data Directory: ${YELLOW}$DATA_DIR${NC}"
echo -e "Output Directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Number of GPUs: ${YELLOW}$NUM_GPUS${NC}"
echo -e "Inference Engine: ${YELLOW}$ENGINE${NC}"
echo -e "${BLUE}Reward Manager: ${YELLOW}adaptive_reasoning${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "${CYAN}=== V6.4 BALANCED Reward Design ===${NC}"
echo -e "${MAGENTA}Problem Fixed:${NC} Length penalty was dominating format bonuses"
echo -e "${MAGENTA}Solution:${NC} Type 2/3 get higher bonuses to compensate"
echo -e ""
echo -e "${BLUE}Format Bonuses (when CORRECT):${NC}"
echo -e "  ${GREEN}Type 1 (direct):${NC}       +${TYPE1_BONUS_INITIAL} (efficient)"
echo -e "  ${YELLOW}Type 2 (perception):${NC}  +0.4 ${CYAN}(INCREASED to compensate length!)${NC}"
echo -e "  ${CYAN}Type 3 (full):${NC}         +0.6 ${CYAN}(INCREASED to compensate length!)${NC}"
echo -e ""
echo -e "${BLUE}Error Penalties (when INCORRECT):${NC}"
echo -e "  ${RED}Type 1 (direct):${NC}       -0.8 ${MAGENTA}(higher risk!)${NC}"
echo -e "  ${YELLOW}Type 2 (perception):${NC}  -0.3 (medium risk)"
echo -e "  ${GREEN}Type 3 (full):${NC}         +0.0 (safe baseline)"
echo -e ""
echo -e "${BLUE}Length Penalty (RELAXED):${NC}"
echo -e "  ${CYAN}Threshold:${NC}     500 tokens (was 300)"
echo -e "  ${CYAN}Ideal Length:${NC}  500 tokens (was 300)"
echo -e "  ${CYAN}Min Scalar:${NC}    0.6 (was 0.3)"
echo -e ""
echo -e "${MAGENTA}=== Reward Calculation Examples ===${NC}"
echo -e "${GREEN}Scenario: Correct Answer${NC}"
echo -e "  Type 1 (~80 tokens):  (1.0+0.2) × 1.0  = ${GREEN}1.2${NC}"
echo -e "  Type 2 (~400 tokens): (1.0+0.4) × 0.8  = ${GREEN}1.12${NC} ${CYAN}← Competitive!${NC}"
echo -e "  Type 3 (~600 tokens): (1.0+0.6) × 0.67 = ${GREEN}1.07${NC} ${CYAN}← Viable option!${NC}"
echo -e ""
echo -e "${RED}Scenario: Incorrect Answer${NC}"
echo -e "  Type 1 (~80 tokens):  (0.0-0.8) × 1.0  = ${RED}-0.8${NC}  ${MAGENTA}← High risk!${NC}"
echo -e "  Type 2 (~400 tokens): (0.0-0.3) × 0.8  = ${RED}-0.24${NC} ${YELLOW}← Medium risk${NC}"
echo -e "  Type 3 (~600 tokens): (0.0+0.0) × 0.67 = ${GREEN}0.0${NC}   ${GREEN}← Safe!${NC}"
echo -e ""
echo -e "${MAGENTA}=== V6.4 Decay Configuration ===${NC}"
echo -e "  ${BLUE}Enable Decay:${NC}         ${ENABLE_DECAY} (default off - balanced rewards work without it)"
echo -e "  ${BLUE}Decay Strategy:${NC}       ${DECAY_STRATEGY}"
echo -e "  ${BLUE}Decay Start Step:${NC}     ${DECAY_START_STEP}"
echo -e "  ${BLUE}Decay End Step:${NC}       ${DECAY_END_STEP}"
echo -e "  ${BLUE}Initial Bonus:${NC}        ${TYPE1_BONUS_INITIAL}"
echo -e "  ${BLUE}Minimum Bonus:${NC}        ${TYPE1_BONUS_MIN}"
if [ "$DECAY_STRATEGY" = "exponential" ]; then
    echo -e "  ${BLUE}Decay Rate:${NC}           ${DECAY_RATE}"
fi
echo -e ""
echo -e "${BLUE}Key Changes from V6.3:${NC}"
echo -e "  ${GREEN}✓${NC} Type 2 bonus: 0.0 → 0.4 (compensates for ~400 token length)"
echo -e "  ${GREEN}✓${NC} Type 3 bonus: 0.0 → 0.6 (compensates for ~600 token length)"
echo -e "  ${GREEN}✓${NC} Type 1 penalty: -0.5 → -0.8 (higher risk for errors)"
echo -e "  ${GREEN}✓${NC} Type 2 penalty: 0.0 → -0.3 (medium risk)"
echo -e "  ${GREEN}✓${NC} Length threshold: 300 → 500 (more lenient)"
echo -e "  ${GREEN}✓${NC} Min scalar: 0.3 → 0.6 (less aggressive penalty)"
echo -e "  ${GREEN}✓${NC} Decay disabled by default (balanced rewards sufficient)"
echo -e ""
echo -e "${BLUE}Expected Model Behavior:${NC}"
echo -e "  ${GREEN}Simple questions:${NC}"
echo -e "    • Model chooses Type 1 if confident (highest reward)"
echo -e "    • High risk (-0.8 penalty) discourages guessing"
echo -e ""
echo -e "  ${YELLOW}Medium questions:${NC}"
echo -e "    • Model may use Type 2 (perception helps, competitive reward)"
echo -e "    • Medium risk (-0.3 penalty) makes it safer than Type 1"
echo -e ""
echo -e "  ${CYAN}Complex questions:${NC}"
echo -e "    • Model uses Type 3 (full reasoning, safe baseline)"
echo -e "    • No penalty on errors, encourages thorough thinking"
echo -e ""
echo -e "  ${MAGENTA}Result:${NC} Adaptive format selection based on confidence!"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "${BLUE}TensorBoard Metrics:${NC}"
echo -e "  • ${GREEN}format/${NC}type{1,2,3}_ratio - Should be BALANCED (not 100% Type1!)"
echo -e "  • ${GREEN}format/${NC}type{1,2,3}_correct_rate - Per-type accuracy"
echo -e "  • ${GREEN}format/${NC}type{1,2,3}_avg_length - Per-type length"
echo -e "  • ${GREEN}reward/${NC}* - Reward component breakdown"
echo -e "  • ${GREEN}accuracy/${NC}overall - Overall accuracy"
echo -e "  • ${MAGENTA}decay/type1_bonus_current${NC} - Current Type1 bonus value (if decay enabled)"
echo -e "  • ${MAGENTA}decay/current_step${NC} - Current training step"
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
echo -e "${GREEN}Starting GRPO training with V6.4 BALANCED reward function...${NC}"

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
    +reward_model.reward_kwargs.correct_reward=1.0 \
    +reward_model.reward_kwargs.incorrect_reward=0.0 \
    +reward_model.reward_kwargs.type1_format_bonus=$TYPE1_BONUS_INITIAL \
    +reward_model.reward_kwargs.type2_format_bonus=0.2 \
    +reward_model.reward_kwargs.type3_format_bonus=0.3 \
    +reward_model.reward_kwargs.type1_error_penalty=-0.1 \
    +reward_model.reward_kwargs.type2_error_penalty=0.0 \
    +reward_model.reward_kwargs.type3_error_penalty=0.0 \
    +reward_model.reward_kwargs.length_threshold=150 \
    +reward_model.reward_kwargs.ideal_length=150.0 \
    +reward_model.reward_kwargs.min_scalar=0.6 \
    +reward_model.reward_kwargs.enable_bonus_decay=$ENABLE_DECAY \
    +reward_model.reward_kwargs.decay_strategy=$DECAY_STRATEGY \
    +reward_model.reward_kwargs.decay_start_step=$DECAY_START_STEP \
    +reward_model.reward_kwargs.decay_end_step=$DECAY_END_STEP \
    +reward_model.reward_kwargs.type1_bonus_min=$TYPE1_BONUS_MIN \
    +reward_model.reward_kwargs.decay_rate=$DECAY_RATE \
    +reward_model.reward_kwargs.normalize_answers=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard","wandb"]' \
    trainer.project_name='verl_grpo_adaptive_reasoning_9k_v6' \
    trainer.experiment_name='qwen2_5vl_3b_adaptive_9k_v6_8' \
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
echo -e "${BLUE}Key Metrics to Monitor:${NC}"
echo -e "  ${GREEN}Format Distribution (CRITICAL):${NC}"
echo -e "    • format/type1_ratio - Should NOT be 100%!"
echo -e "    • format/type2_ratio - Should be 20-40%"
echo -e "    • format/type3_ratio - Should be 10-30%"
echo -e ""
echo -e "  ${GREEN}Reward Trend (MOST IMPORTANT):${NC}"
echo -e "    • ${MAGENTA}reward/total_mean${NC} - MUST increase over training!"
echo -e "    • If not increasing, check format distribution"
echo -e ""
echo -e "  ${GREEN}Per-Type Performance:${NC}"
echo -e "    • format/type{1,2,3}_correct_rate - Accuracy by format"
echo -e "    • format/type{1,2,3}_avg_length - Length by format"
echo -e ""
echo -e "  ${GREEN}Reward Components:${NC}"
echo -e "    • reward/base_mean - Base correctness reward"
echo -e "    • reward/format_bonus_mean - Format bonus/penalty"
echo -e "    • reward/length_scalar_mean - Length penalty scalar"
echo -e ""
echo -e "  ${GREEN}Overall:${NC}"
echo -e "    • accuracy/overall - Overall accuracy (may fluctuate)"
echo -e ""
echo -e "${CYAN}V6.4 Success Criteria:${NC}"
echo -e "  ${GREEN}✓${NC} reward/total_mean increases steadily"
echo -e "  ${GREEN}✓${NC} Format distribution is balanced (not 100% Type1)"
echo -e "  ${GREEN}✓${NC} Type 2 and Type 3 are actually used"
echo -e "  ${GREEN}✓${NC} Overall accuracy stable or improving"
echo -e ""
echo -e "${BLUE}Expected Training Dynamics:${NC}"
echo -e "  ${GREEN}Early Steps (1-10):${NC}"
echo -e "    • Model explores all formats"
echo -e "    • Type1 ratio should be 30-50% (not 100%!)"
echo -e "    • reward/total_mean starts increasing"
echo -e ""
echo -e "  ${YELLOW}Mid Training (10-30):${NC}"
echo -e "    • Model learns when to use each format"
echo -e "    • Balanced distribution emerges"
echo -e "    • Accuracy improves for all types"
echo -e ""
echo -e "  ${CYAN}Late Training (30+):${NC}"
echo -e "    • Adaptive format selection by task difficulty"
echo -e "    • Stable, balanced performance"
echo -e "    • High overall reward"
echo -e ""
echo -e "${BLUE}For more information, see:${NC}"
echo -e "  ${YELLOW}reward_functions/adaptive_reasoning_reward_v6.py${NC}"
echo -e "${GREEN}========================================${NC}"
