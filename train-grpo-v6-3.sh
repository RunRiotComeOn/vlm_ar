#!/bin/bash
# Training script for Stage 2 GRPO with Adaptive Reasoning V6
# Features: Type1 Bonus Decay, Error Penalties, Adaptive Exploration
# Reward Design: Type 1 > Type 2 > Type 3 with decaying Type1 bonus to prevent over-exploitation
# Key Innovation: Decay mechanism prevents model from exploiting Type 1 format too much

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
OUTPUT_DIR="saves/qwen2_5vl-3b/grpo/adaptive_reasoning_9k_v6_5"
NUM_GPUS=8
ENGINE="vllm"

# V6 Decay parameters (can be overridden via command line)
ENABLE_DECAY=true
DECAY_STRATEGY="linear"  # linear, exponential, or cosine
DECAY_START_STEP=0
DECAY_END_STEP=10
TYPE1_BONUS_INITIAL=0.1
TYPE1_BONUS_MIN=-0.1
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
            echo "  --data_dir <path>          GRPO data directory (default: grpo_data)"
            echo "  --output_dir <path>        Output directory (default: saves/qwen2_5vl-3b/grpo/adaptive_reasoning_9k_v6)"
            echo "  --gpus <num>               Number of GPUs (default: 8)"
            echo "  --engine <vllm|sglang>     Inference engine (default: vllm)"
            echo ""
            echo "V6 Decay Options:"
            echo "  --enable_decay <true|false>        Enable Type1 bonus decay (default: true)"
            echo "  --decay_strategy <strategy>        Decay strategy: linear, exponential, cosine (default: linear)"
            echo "  --decay_start_step <num>           Step to start decay (default: 0)"
            echo "  --decay_end_step <num>             Step to finish decay (default: 50)"
            echo "  --type1_bonus_initial <float>      Initial Type1 bonus (default: 0.3)"
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
echo -e "${GREEN}GRPO Training - Adaptive Reasoning V6${NC}"
echo -e "${MAGENTA}Type1 Bonus Decay for Balanced Exploration${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "SFT Model: ${YELLOW}$SFT_MODEL${NC}"
echo -e "Data Directory: ${YELLOW}$DATA_DIR${NC}"
echo -e "Output Directory: ${YELLOW}$OUTPUT_DIR${NC}"
echo -e "Number of GPUs: ${YELLOW}$NUM_GPUS${NC}"
echo -e "Inference Engine: ${YELLOW}$ENGINE${NC}"
echo -e "${BLUE}Reward Manager: ${YELLOW}adaptive_reasoning${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "${CYAN}=== V6 Reward Design ===${NC}"
echo -e "${BLUE}Format Bonuses (when CORRECT):${NC}"
echo -e "  ${GREEN}Type 1 (direct):${NC}       +${TYPE1_BONUS_INITIAL} → ${TYPE1_BONUS_MIN} ${MAGENTA}(DECAYING!)${NC}"
echo -e "  ${YELLOW}Type 2 (perception):${NC}  +0.1 (stable)"
echo -e "  ${CYAN}Type 3 (full):${NC}         +0.0 (baseline)"
echo -e ""
echo -e "${BLUE}Error Penalties (when INCORRECT):${NC}"
echo -e "  ${RED}Type 1 (direct):${NC}       -0.5 (high risk)"
echo -e "  ${YELLOW}Type 2 (perception):${NC}  -0.3 (medium risk)"
echo -e "  ${GREEN}Type 3 (full):${NC}         +0.0 (safe baseline)"
echo -e ""
echo -e "${MAGENTA}=== V6 Decay Configuration ===${NC}"
echo -e "  ${BLUE}Enable Decay:${NC}         ${ENABLE_DECAY}"
echo -e "  ${BLUE}Decay Strategy:${NC}       ${DECAY_STRATEGY}"
echo -e "  ${BLUE}Decay Start Step:${NC}     ${DECAY_START_STEP}"
echo -e "  ${BLUE}Decay End Step:${NC}       ${DECAY_END_STEP}"
echo -e "  ${BLUE}Initial Bonus:${NC}        ${TYPE1_BONUS_INITIAL}"
echo -e "  ${BLUE}Minimum Bonus:${NC}        ${TYPE1_BONUS_MIN}"
if [ "$DECAY_STRATEGY" = "exponential" ]; then
    echo -e "  ${BLUE}Decay Rate:${NC}           ${DECAY_RATE}"
fi
echo -e ""
echo -e "${BLUE}Why Decay?${NC}"
echo -e "  ${MAGENTA}Problem:${NC} If Type1 ratio reaches 0.8+ too quickly (e.g., step 12)"
echo -e "  ${MAGENTA}Solution:${NC} Decay Type1 bonus to:"
echo -e "    • Reduce over-exploitation of Type 1 format"
echo -e "    • Encourage exploration of Type 2 and Type 3"
echo -e "    • Maintain balanced format distribution"
echo -e "    • Improve training stability"
echo -e ""
echo -e "${BLUE}Decay Strategies:${NC}"
echo -e "  ${GREEN}linear:${NC}      Smooth linear decrease"
echo -e "  ${GREEN}exponential:${NC} Fast initial decay, slower later"
echo -e "  ${GREEN}cosine:${NC}      Cosine annealing (smooth curve)"
echo -e ""
echo -e "${BLUE}Key Changes from V5:${NC}"
echo -e "  ${GREEN}✓${NC} Added Type1 bonus decay mechanism"
echo -e "  ${GREEN}✓${NC} Three decay strategies (linear/exponential/cosine)"
echo -e "  ${GREEN}✓${NC} Tracks current bonus in metrics"
echo -e "  ${GREEN}✓${NC} All V5 features preserved (error penalties, etc.)"
echo -e ""
echo -e "${BLUE}Expected Model Behavior:${NC}"
echo -e "  • Early training: Model prefers Type 1 (high bonus)"
echo -e "  • Mid training: Type1 bonus decays, explores Type 2/3"
echo -e "  • Late training: Balanced format distribution"
echo -e "  • Result: More robust adaptive reasoning"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "${BLUE}TensorBoard Metrics:${NC}"
echo -e "  • ${GREEN}format/${NC}type{1,2,3}_ratio - Format distribution over time"
echo -e "  • ${GREEN}format/${NC}type{1,2,3}_correct_rate - Per-type accuracy"
echo -e "  • ${GREEN}format/${NC}type{1,2,3}_avg_length - Per-type length"
echo -e "  • ${GREEN}reward/${NC}* - Reward component breakdown"
echo -e "  • ${GREEN}accuracy/${NC}overall - Overall accuracy"
echo -e "  • ${MAGENTA}decay/type1_bonus_current${NC} - Current Type1 bonus value"
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
echo -e "${GREEN}Starting GRPO training with V6 reward function (with decay)...${NC}"

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
    +reward_model.reward_kwargs.type2_format_bonus=0.0 \
    +reward_model.reward_kwargs.type3_format_bonus=0.0 \
    +reward_model.reward_kwargs.type1_error_penalty=-0.5 \
    +reward_model.reward_kwargs.type2_error_penalty=0.0 \
    +reward_model.reward_kwargs.type3_error_penalty=0.0 \
    +reward_model.reward_kwargs.length_threshold=300 \
    +reward_model.reward_kwargs.ideal_length=300.0 \
    +reward_model.reward_kwargs.min_scalar=0.3 \
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
    trainer.experiment_name='qwen2_5vl_3b_adaptive_9k_v6_linear_5' \
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
echo -e "  ${GREEN}Format Distribution:${NC}"
echo -e "    • format/type1_ratio - Should decrease over time with decay"
echo -e "    • format/type2_ratio - Should increase as Type1 decays"
echo -e "    • format/type3_ratio - Baseline exploration"
echo -e ""
echo -e "  ${GREEN}Per-Type Performance:${NC}"
echo -e "    • format/type{1,2,3}_correct_rate - Accuracy by format"
echo -e "    • format/type{1,2,3}_avg_length - Length by format"
echo -e ""
echo -e "  ${GREEN}Reward Components:${NC}"
echo -e "    • reward/base_mean - Base correctness reward"
echo -e "    • reward/format_bonus_mean - Format bonus/penalty"
echo -e "    • reward/length_scalar_mean - Length penalty scalar"
echo -e "    • reward/total_mean - Final reward"
echo -e ""
echo -e "  ${MAGENTA}V6 Decay Metrics:${NC}"
echo -e "    • ${MAGENTA}decay/type1_bonus_current${NC} - Current Type1 bonus value"
echo -e "    • ${MAGENTA}decay/current_step${NC} - Current training step"
echo -e ""
echo -e "  ${GREEN}Overall:${NC}"
echo -e "    • accuracy/overall - Overall accuracy"
echo -e ""
echo -e "${CYAN}V6 Innovation:${NC}"
echo -e "  ${MAGENTA}Decay Mechanism:${NC}"
echo -e "    • Prevents over-exploitation of Type 1 format"
echo -e "    • Encourages balanced exploration across all formats"
echo -e "    • Improves training stability and robustness"
echo -e "    • Configurable decay strategy and schedule"
echo -e ""
echo -e "${BLUE}Expected Training Dynamics:${NC}"
echo -e "  ${GREEN}Phase 1 (Steps 0-${DECAY_START_STEP}):${NC}"
echo -e "    • High Type1 bonus (${TYPE1_BONUS_INITIAL})"
echo -e "    • Model learns efficient Type 1 responses"
echo -e ""
echo -e "  ${YELLOW}Phase 2 (Steps ${DECAY_START_STEP}-${DECAY_END_STEP}):${NC}"
echo -e "    • Type1 bonus decays (${DECAY_STRATEGY} strategy)"
echo -e "    • Model explores Type 2 and Type 3 more"
echo -e "    • Format distribution becomes balanced"
echo -e ""
echo -e "  ${CYAN}Phase 3 (Steps ${DECAY_END_STEP}+):${NC}"
echo -e "    • Type1 bonus stabilizes at ${TYPE1_BONUS_MIN}"
echo -e "    • Model adaptively chooses format by task complexity"
echo -e "    • Robust adaptive reasoning achieved"
echo -e ""
echo -e "${BLUE}For more information, see:${NC}"
echo -e "  ${YELLOW}reward_functions/adaptive_reasoning_reward_v6.py${NC}"
echo -e "  ${YELLOW}REWARD_V6_DECAY_GUIDE.md${NC}"
echo -e "${GREEN}========================================${NC}"
