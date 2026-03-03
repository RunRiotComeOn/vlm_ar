#!/bin/bash
# GRPO Training for Qwen2.5-VL-3B with Adaptive Reasoning Reward
#
# This script trains the model to adaptively choose reasoning modes:
# - Type 1 (Direct): Highest reward when correct
# - Type 2 (Perception): Medium reward
# - Type 3 (Full reasoning): Lower reward (but still positive if correct)
#
# The model learns to use the simplest approach that gets the answer right.

set -x

# Configuration
ENGINE=${1:-vllm}  # vllm or sglang
SFT_MODEL_PATH=${2:-"saves/qwen2_5vl-3b/full/sft_all"}  # Path to SFT checkpoint
DATA_DIR=${3:-"grpo_data"}  # GRPO data directory
OUTPUT_DIR=${4:-"saves/qwen2_5vl-3b/grpo/adaptive_reasoning"}

# Check if SFT model exists
if [ ! -d "$SFT_MODEL_PATH" ]; then
    echo "ERROR: SFT model not found at $SFT_MODEL_PATH"
    echo "Please run SFT training first or specify correct path"
    exit 1
fi

# Check if GRPO data exists
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "ERROR: GRPO training data not found at $DATA_DIR/train.parquet"
    echo "Please run prepare_grpo_data.py first"
    exit 1
fi

# Run GRPO training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/val.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.path=reward_functions.adaptive_reasoning_reward \
    reward_model.input_tokenizer=$SFT_MODEL_PATH \
    reward_model.enable_rm_test=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_grpo_adaptive_reasoning' \
    trainer.experiment_name='qwen2_5vl_3b_adaptive' \
    trainer.default_hdfs_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 \
    $@

echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
