# GRPO Countdown - Qwen3.5-2B-Base 训练实验

## 实验信息
- 日期: 2026-03-29
- 模型: Qwen3.5-2B-Base (1.88B params)
- 算法: GRPO (Group Relative Policy Optimization)
- GPU: 1x 32GB (AutoDL)
- 数据集: Jiayi-Pan/Countdown-Tasks-3to4 (327,680 train / 1,024 test)

## 本轮所有代码改动

### 1. verl/utils/tracking.py
- 新增 `tensorboard` backend 支持
- 添加 `_TensorBoardAdapter` 类，将训练指标写入 TensorBoard

### 2. verl/utils/torch_functional.py
- `logprobs_from_logits_flash_attn()`: 添加 `.cuda()` 确保 tensor 在 GPU 上（修复 param_offload 导致的 CPU tensor 问题）

### 3. verl/utils/fsdp_utils.py
- 当 FSDP wrap policy 中找不到某个 class 时，改为 `continue` 而非 `raise Exception`（修复 Qwen3_5VisionBlock 不存在的问题）

### 4. verl/workers/fsdp_workers.py
- 添加 `from verl.workers.actor import DataParallelPPOActor` 局部导入（修复 NameError）
- 添加 `recompute_log_prob = True` 变量定义（修复 NameError）
- Actor 优化器改为 `bitsandbytes.AdamW8bit`（8-bit AdamW，~2GB vs ~15GB）
- `init_model()` 中添加 dummy optimizer step 预初始化 states 并 offload 到 CPU
- `model_dtype=bf16` 写入 YAML（避免 fp32 加载，省一半显存）

### 5. verl/workers/actor/dp_actor.py
- `flash_attn` 导入改为 `try/except`（可选依赖）
- `compute_policy_loss()` 返回值扩展：新增 ratio_mean, ratio_max, ratio_min
- 新增 TensorBoard 指标：`actor/ratio/mean`, `actor/ratio/max`, `actor/ratio/min`

### 6. verl/workers/critic/dp_critic.py
- `flash_attn` 导入改为 `try/except`（可选依赖）

### 7. verl/workers/rollout/hf_rollout.py
- `generate_sequences()` 添加 `n>1` 支持：将 prompts repeat n 次后生成（HF Rollout 原生不支持多采样）

### 8. verl/third_party/vllm/__init__.py
- vLLM 版本不匹配时 fallback 到 0.6.3 adapter（而非 raise）

### 9. examples/data_preprocess/countdown.py
- `qwen3.5` 模板改为中文（system prompt、instruction、few-shot example）

### 10. verl/trainer/config/ppo_trainer.yaml
- actor fsdp_config 添加 `model_dtype: bf16`

### 11. verl/trainer/ppo/ray_trainer.py (新增详细指标)
- `apply_kl_penalty()`: 新增 KL 散度详细统计
  - `critic/kl/max`, `critic/kl/min`, `critic/kl/std`
- `compute_data_metrics()`: 新增分布统计
  - Score: `std`, `positive_ratio`
  - Rewards: `std`
  - Advantages: `std`, `positive_ratio`
- `compute_advantage()`: GRPO 模式下收集组级指标并传递到 meta_info
- 训练循环中将 `grpo_metrics` 合并到整体 metrics

### 12. verl/trainer/ppo/core_algos.py (新增 GRPO 指标)
- `compute_grpo_outcome_advantage()`: 新增 GRPO 组级别统计指标
  - `grpo/group_size/mean`, `grpo/group_size/min`, `grpo/group_size/max`
  - `grpo/raw_score/mean`, `grpo/raw_score/std`, `grpo/raw_score/positive_ratio`
  - `grpo/num_groups`
- `compute_policy_loss()`: 新增策略比率统计
  - 返回 `ratio_mean`, `ratio_max`, `ratio_min`

## 训练参数（v3 - 最终版）

| 参数 | v1 (验证版) | v2 (调大版) | v3 (当前) | 说明 |
|------|-------------|-------------|-----------|------|
| train_batch_size | 16 | 32 | 64 | 4x |
| val_batch_size | 8 | 16 | 16 | 配套 |
| response_length | 256 | 512 | 1024 | 解决 91% clip ratio |
| n (samples/prompt) | 4 | 6 | 8 | GRPO 更多采样 |
| rollout.micro_batch_size | 2 | 4 | 2 | response 变长后减小 |
| ppo_mini_batch_size | 4 | 8 | 8 | 配合更大 batch |
| ppo_micro_batch_size | 1 | 1 | 1 | 保持 |
| log_prob_micro_batch_size | 2 | 2 | 2 | 保持 |

## 省显存机制

| 机制 | 状态 |
|------|------|
| Gradient Checkpointing | 开启 |
| 8-bit AdamW (bitsandbytes) | 开启 |
| FSDP Mixed Precision (bf16) | 开启 |
| Parameter Offload | 开启 |
| Optimizer Offload | 开启 |
| Gradient Accumulation | 自动 |

## TensorBoard 指标

日志目录: `/root/tf-logs/training_metrics`

### Actor 训练指标
- `actor/pg_loss` - 策略梯度损失
- `actor/kl_loss` - KL 散度损失
- `actor/kl_coef` - KL 系数
- `actor/entropy_loss` - 熵损失
- `actor/pg_clipfrac` - PPO 裁剪比例
- `actor/ppo_kl` - 近似 KL 散度
- `actor/grad_norm` - 梯度范数
- `actor/lr` - 学习率
- `actor/ratio/mean` - 策略比率均值
- `actor/ratio/max` - 策略比率最大值
- `actor/ratio/min` - 策略比率最小值

### KL 散度详情
- `critic/kl` - KL 均值
- `critic/kl_coeff` - KL 惩罚系数
- `critic/kl/max` - KL 最大值
- `critic/kl/min` - KL 最小值
- `critic/kl/std` - KL 标准差

### 奖励/分数/优势分布
- `critic/score/mean|max|min|std|positive_ratio`
- `critic/rewards/mean|max|min|std`
- `critic/advantages/mean|max|min|std|positive_ratio`
- `critic/returns/mean|max|min`

### GRPO 组统计
- `grpo/group_size/mean|min|max` - 每组采样数
- `grpo/raw_score/mean|std|positive_ratio` - 原始分数分布
- `grpo/num_groups` - 组数量

### 序列长度
- `response_length/mean|max|min|clip_ratio`
- `prompt_length/mean|max|min|clip_ratio`

### 时间性能
- `timing_s/gen|ref|update_actor|step`
