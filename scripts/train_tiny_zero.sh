#!/bin/bash
# =============================================================================
# TinyZero PPO 训练启动脚本
# =============================================================================
#
# 【功能说明】
# 此脚本是 TinyZero 训练的主要入口。它通过 Hydra 配置系统
# 启动 PPO 强化学习训练。
#
# 【前置条件】
# 在运行此脚本之前，需要设置以下环境变量：
# - N_GPUS: 每个节点的 GPU 数量（例如：1 或 2）
# - BASE_MODEL: 基座模型路径（例如：~/models/Qwen2.5-3B）
# - DATA_DIR: 数据目录路径（例如：~/data/countdown）
# - ROLLOUT_TP_SIZE: vLLM 张量并行大小（通常等于 GPU 数量）
# - EXPERIMENT_NAME: 实验名称（用于 wandb 日志）
#
# 【使用示例】
# # 单 GPU 训练（适用于 <=1.5B 模型）
# export N_GPUS=1
# export BASE_MODEL=~/models/Qwen2.5-1.5B
# export DATA_DIR=~/data/countdown
# export ROLLOUT_TP_SIZE=1
# export EXPERIMENT_NAME=countdown-qwen2.5-1.5b
# export VLLM_ATTENTION_BACKEND=XFORMERS
# bash ./scripts/train_tiny_zero.sh
#
# # 多 GPU 训练（适用于 3B+ 模型）
# export N_GPUS=2
# export BASE_MODEL=~/models/Qwen2.5-3B
# export DATA_DIR=~/data/countdown
# export ROLLOUT_TP_SIZE=2
# export EXPERIMENT_NAME=countdown-qwen2.5-3b
# export VLLM_ATTENTION_BACKEND=XFORMERS
# bash ./scripts/train_tiny_zero.sh
#
# 【显存优化】
# 如果遇到显存不足 (OOM)，可以添加以下参数：
# - critic.model.enable_gradient_checkpointing=True
# - actor_rollout_ref.model.enable_gradient_checkpointing=True
# - actor_rollout_ref.rollout.gpu_memory_utilization=0.3
#
# =============================================================================

# =============================================================================
# 核心训练命令
# =============================================================================
# 使用 python3 -m 调用 verl.trainer.main_ppo 模块
# 所有参数通过 Hydra 的命令行覆盖语法传递

python3 -m verl.trainer.main_ppo \

# -----------------------------------------------------------------------------
# 数据配置
# -----------------------------------------------------------------------------
# 训练数据文件路径（Parquet 格式）
data.train_files=$DATA_DIR/train.parquet \

# 验证数据文件路径
data.val_files=$DATA_DIR/test.parquet \

# 训练批次大小（每个 epoch 处理的总样本数）
# 较大的值可以提高训练稳定性，但需要更多显存
data.train_batch_size=256 \

# 验证批次大小
data.val_batch_size=1312 \

# 最大提示长度（token 数）
# 超过此长度的提示会被截断
data.max_prompt_length=256 \

# 最大响应长度（token 数）
# 模型生成的响应不会超过此长度
data.max_response_length=1024 \

# -----------------------------------------------------------------------------
# Actor 模型配置
# -----------------------------------------------------------------------------
# Actor 是 PPO 中的策略网络，负责生成响应

# 基座模型路径
actor_rollout_ref.model.path=$BASE_MODEL \

# 使用 remove_padding 优化（减少无效计算）
actor_rollout_ref.model.use_remove_padding=True \

# 使用动态批次大小（根据序列长度自动调整）
# 可以更有效地利用显存
actor_rollout_ref.actor.use_dynamic_bsz=True \

# Actor 优化器学习率
# 通常比 Critic 的学习率小一个数量级
# 较小的学习率可以防止策略更新过快导致不稳定
actor_rollout_ref.actor.optim.lr=1e-6 \

# PPO mini batch 大小
# 在每个更新步骤中处理的样本数
actor_rollout_ref.actor.ppo_mini_batch_size=64 \

# PPO micro batch 大小
# mini batch 被分割成多个 micro batch 进行梯度累积
# 有助于减少显存使用
actor_rollout_ref.actor.ppo_micro_batch_size=8 \

# -----------------------------------------------------------------------------
# Rollout 配置（响应生成）
# -----------------------------------------------------------------------------
# Rollout 是使用 vLLM 引擎生成响应的过程

# 计算 log probability 时的 micro batch 大小
actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \

# vLLM 张量并行大小
# 设置为 GPU 数量以利用多 GPU 推理
# 对于单 GPU，设置为 1
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \

# vLLM GPU 显存利用率
# 值越小，vLLM 占用的显存越少，留给训练的显存越多
# 如果遇到 OOM，可以降低此值（如 0.3 或 0.2）
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \

# -----------------------------------------------------------------------------
# Reference Policy 配置
# -----------------------------------------------------------------------------
# Reference Policy 是不更新的原始策略，用于计算 KL 散度

# 计算 log probability 时的 micro batch 大小
actor_rollout_ref.ref.log_prob_micro_batch_size=4 \

# -----------------------------------------------------------------------------
# Critic 模型配置
# -----------------------------------------------------------------------------
# Critic 是 PPO 中的价值网络，负责估计状态价值

# Critic 优化器学习率
# 通常比 Actor 的学习率大一个数量级
critic.optim.lr=1e-5 \

# Critic 模型路径（通常与 Actor 使用同一个基座模型）
critic.model.path=$BASE_MODEL \

# Critic 的 micro batch 大小
critic.ppo_micro_batch_size=8 \

# -----------------------------------------------------------------------------
# 算法配置
# -----------------------------------------------------------------------------

# KL 散度系数
# 用于控制策略更新不要偏离 reference policy 太远
# 较小的值允许更大的策略变化，但可能导致训练不稳定
algorithm.kl_ctrl.kl_coef=0.001 \

# -----------------------------------------------------------------------------
# 训练器配置
# -----------------------------------------------------------------------------

# 日志记录器（使用 wandb 进行实验跟踪）
trainer.logger=['wandb'] \

# 训练前是否进行验证
# 设置为 False 可以跳过初始验证，节省时间
+trainer.val_before_train=False \

# HDFS 默认目录（设为 null 表示不使用 HDFS）
trainer.default_hdfs_dir=null \

# 每个节点的 GPU 数量
trainer.n_gpus_per_node=$N_GPUS \

# 节点数量（1 表示单机训练）
trainer.nnodes=1 \

# 模型保存频率（每多少步保存一次检查点）
# 设置为 -1 表示不自动保存
trainer.save_freq=100 \

# 验证频率（每多少步进行一次验证）
trainer.test_freq=100 \

# 项目名称（用于 wandb 日志分组）
trainer.project_name=TinyZero \

# 实验名称（用于区分不同的训练运行）
trainer.experiment_name=$EXPERIMENT_NAME \

# 总训练轮数
# 一个 epoch 表示遍历整个训练数据集一次
trainer.total_epochs=15 \

# 将输出重定向到日志文件，同时在终端显示
2>&1 | tee verl_demo.log
