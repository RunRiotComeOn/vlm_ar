#!/bin/bash
# Setup verl conda environment from scratch
# Python 3.10 + vLLM 0.11.0 + flash-attn 2.8.3 + verl

set -e

# Color codes
GREEN='\033[0;32m'
NC='\033[0m'

# Starting File Settings
git clone --recurse-submodules https://github.com/RunRiotComeOn/vlm_ar.git
cd vlm_ar
REPO_DIR="$PWD"

ENV_NAME=${1:-verl}
TMPDIR=${TMPDIR:-/tmp/verl_setup}
mkdir -p "$TMPDIR"

# 1. Create conda environment
conda create -n "$ENV_NAME" python=3.10 -y
ENV_PATH=$(conda run -n "$ENV_NAME" python -c "import sys; print(sys.prefix)")

# 2. Install vLLM 0.11.0 (pulls torch 2.8.0+cu128 automatically)
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" vllm==0.11.0

# 3. Install flash-attn 2.8.3 precompiled (torch2.8 + cu12 + cxx11abiTRUE + cp310)
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# 4. Install training dependencies
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" \
    wandb hydra-core qwen-vl-utils accelerate datasets peft tensordict \
    pylatexenc liger-kernel dill

# 5. Downgrade transformers to 4.x (5.x removed AutoModelForVision2Seq)
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" \
    "transformers>=4.51.0,<5.0.0"

# 6. Install huggingface_hub into the conda env
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" \
    -U "huggingface_hub[cli]"

# 7. Install verl in editable mode
TMPDIR="$TMPDIR" "$ENV_PATH/bin/pip" install --cache-dir "$TMPDIR/pip-cache" \
    -e "$REPO_DIR/verl"

# 8. Login to wandb
conda run -n "$ENV_NAME" wandb login "${WANDB_API_KEY:-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}"

# 9. Login to Hugging Face
conda run -n "$ENV_NAME" huggingface-cli login --token "${HF_TOKEN:-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}"

echo "Done! Activate with: conda activate $ENV_NAME"

# Download SFT model
mkdir -p models
"$ENV_PATH/bin/huggingface-cli" download yixuH/qwen3_vl_8b_arvlm_sft \
  --local-dir models/qwen3_vl_8b_arvlm_sft \
  --local-dir-use-symlinks False \
  --resume-download

# Download GRPO data and untar
"$ENV_PATH/bin/huggingface-cli" download yixuH/grpo_data \
  --local-dir ./grpo_data \
  --local-dir-use-symlinks False
cd grpo_data
for f in *.tar.gz *.tar; do
    [ -f "$f" ] && tar -xvf "$f"
done
cd ..

SFT_MODEL="models/qwen3_vl_8b_arvlm_sft"
DATA_DIR="grpo_data"
OUTPUT_DIR="saves/qwen3vl-8b/grpo"
ENGINE="vllm"
NUM_GPUS=4

# NCCL settings
export NCCL_TIMEOUT=7200
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=22
export NCCL_BLOCKING_WAIT=1

# PyTorch distributed timeout
export TORCH_DISTRIBUTED_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# VLLM settings
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
echo -e "${GREEN}Starting GRPO training with diversity bonus...${NC}"

cd verl

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=../$DATA_DIR/train.parquet \
    data.val_files=../$DATA_DIR/val.parquet \
    data.train_batch_size=128 \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.02 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$NUM_GPUS \
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
    reward_model.reward_manager=adaptive_reward_v2 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_adaptive_reward' \
    trainer.experiment_name='qwen3vl_4b_adaptive_reward' \
    trainer.default_local_dir=../$OUTPUT_DIR \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.resume_mode=auto \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=2

cd ..

echo -e "${GREEN}Training completed!${NC}"

# =============================================================
# Merge FSDP checkpoint to HuggingFace format
# =============================================================
echo -e "${GREEN}Finding latest checkpoint...${NC}"

LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found in $OUTPUT_DIR"
    exit 1
fi

echo "Latest checkpoint: $LATEST_CKPT"
echo -e "${GREEN}Merging FSDP checkpoint to HuggingFace format...${NC}"

"$ENV_PATH/bin/python3" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LATEST_CKPT}/actor" \
    --target_dir "${LATEST_CKPT}/actor_merged"

echo "Merge completed! Model saved to: ${LATEST_CKPT}/actor_merged"

# =============================================================
# Upload merged model to HuggingFace
# =============================================================
echo -e "${GREEN}Uploading merged model to HuggingFace...${NC}"

"$ENV_PATH/bin/huggingface-cli" upload \
    yixuH/qwen3_vl_8b_arvlm_grpo \
    "${LATEST_CKPT}/actor_merged" \
    . \
    --repo-type model

echo -e "${GREEN}Upload completed! Model available at: https://huggingface.co/yixuH/qwen3_vl_8b_arvlm_grpo${NC}"
