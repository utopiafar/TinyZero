# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO算法的核心函数实现。
本文件中的函数用于支持不同分布式策略的trainer来实现PPO算法。
"""

# ==============================================================================
# PPO核心算法模块
# ==============================================================================
# 本模块实现了PPO(Proximal Policy Optimization)算法的核心组件:
#
# 主要组件:
#   1. KL控制器: AdaptiveKLController(自适应), FixedKLController(固定)
#   2. 优势估计: GAE (Generalized Advantage Estimation)
#   3. GRPO优势: Group Relative Policy Optimization (无需Critic)
#   4. 策略损失: PPO clipped surrogate loss
#   5. 价值损失: Value function loss with clipping
#   6. 熵损失:   鼓励探索
#   7. KL惩罚:   多种KL散度计算方式
#
# 参考:
#   - PPO: https://arxiv.org/abs/1707.06347
#   - GAE: https://arxiv.org/abs/1506.02438
#   - GRPO: https://arxiv.org/abs/2402.03300
# ==============================================================================

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    自适应KL控制器

    根据当前KL散度与目标KL散度的差距,动态调整KL系数。
    当实际KL > 目标KL时,增加KL系数以加强约束;
    当实际KL < 目标KL时,减小KL系数以放松约束。

    更新公式: coef = coef * (1 + clip(KL/target-1, -0.2, 0.2) * n_steps/horizon)

    参考: https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """
    固定KL控制器

    使用固定的KL系数,不随训练过程调整。
    适用于KL惩罚较稳定或不需要动态调整的场景。
    """

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """
    使用GAE(Generalized Advantage Estimation)计算优势和回报

    GAE结合了TD方法和蒙特卡洛方法的优势,通过lambda参数控制偏差-方差权衡。
    当lam=0时,等价于TD(0); 当lam=1时,等价于蒙特卡洛方法。

    计算过程:
    1. 从后向前遍历时间步
    2. 计算TD误差: delta = r_t + gamma * V(s_{t+1}) - V(s_t)
    3. 累积优势: A_t = delta_t + gamma * lam * A_{t+1}
    4. 回报 = 优势 + 价值
    5. 对优势进行白化(whiten)归一化

    Args:
        token_level_rewards: 每个token的奖励, shape: (bs, response_length)
        values: Critic预测的价值, shape: (bs, response_length)
        eos_mask: EOS标记掩码, shape: (bs, response_length)
        gamma: 折扣因子
        lam: GAE lambda参数

    Returns:
        advantages: 优势估计, shape: (bs, response_length)
        returns: 回报, shape: (bs, response_length)

    参考: https://arxiv.org/abs/1506.02438
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): 本实现仅考虑结果监督,其中奖励是标量。
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    GRPO(Group Relative Policy Optimization)优势计算

    GRPO是一种无需Critic模型的优势估计方法,核心思想是:
    1. 对每个prompt生成多个responses(组)
    2. 在组内进行相对奖励归一化
    3. 优势 = (reward - group_mean) / (group_std + epsilon)

    这种方法避免了训练单独的Critic模型,减少了内存和计算开销。
    适用于结果监督(每个response只有一个标量奖励)的场景。

    Args:
        token_level_rewards: 每个token的奖励, shape: (bs, response_length)
        eos_mask: EOS标记掩码, shape: (bs, response_length)
        index: 组标识符,相同index的样本属于同一组
        epsilon: 数值稳定性常数

    Returns:
        advantages: 优势估计, shape: (bs, response_length)
        returns: 回报(与advantages相同), shape: (bs, response_length)

    参考: https://arxiv.org/abs/2402.03300
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """
    计算带KL惩罚的奖励

    reward = score - KL(current_policy, ref_policy) * kl_ratio

    Args:
        token_level_scores: 原始奖励分数
        old_log_prob: 当前策略的log概率
        ref_log_prob: 参考策略的log概率
        kl_ratio: KL惩罚系数

    Returns:
        带KL惩罚的奖励
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """
    计算PPO策略损失(Clipped Surrogate Objective)

    PPO通过裁剪概率比率来限制策略更新幅度,防止策略变化过大。
    损失函数: L^CLIP = -E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    其中 r_t = exp(log_prob - old_log_prob) 是新旧策略的概率比

    Args:
        old_log_prob: 旧策略的log概率, shape: (bs, response_length)
        log_prob: 新策略的log概率, shape: (bs, response_length)
        advantages: 优势估计, shape: (bs, response_length)
        eos_mask: EOS标记掩码, shape: (bs, response_length)
        cliprange: PPO裁剪范围ε

    Returns:
        pg_loss: 策略梯度损失(标量)
        pg_clipfrac: 被裁剪的样本比例
        ppo_kl: 近似KL散度

    参考: https://arxiv.org/abs/1707.06347
    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """
    计算策略的熵损失

    熵损失鼓励策略保持探索性,避免过早收敛到确定性策略。
    H(π) = -Σ p(a) * log p(a)

    在PPO中,通常会给熵损失一个较小的负权重(如0.01),作为探索奖励。

    Args:
        logits: 模型输出的logits, shape: (bs, response_length, vocab_size)
        eos_mask: EOS标记掩码, shape: (bs, response_length)

    Returns:
        entropy: 平均熵损失(标量)
    """
    # 计算熵
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """
    计算价值函数损失(Critic Loss)

    使用裁剪的价值损失来稳定Critic训练:
    1. 对预测值进行裁剪: clip(vpreds, values-cliprange, values+cliprange)
    2. 计算两种MSE损失: 原始预测vs裁剪后预测
    3. 取较大者作为最终损失

    这种裁剪机制类似于PPO的策略裁剪,防止Critic更新过快。

    Args:
        vpreds: Critic预测的新价值, shape: (batch_size, response_length)
        values: Critic预测的旧价值, shape: (batch_size, response_length)
        returns: 实际回报(目标), shape: (batch_size, response_length)
        eos_mask: EOS标记掩码
        cliprange_value: 价值裁剪范围

    Returns:
        vf_loss: 价值函数损失(标量)
        vf_clipfrac: 被裁剪的样本比例
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """
    计算KL散度惩罚

    支持多种KL惩罚计算方式:
    - 'kl': 标准KL散度, log_prob - ref_log_prob
    - 'abs': 绝对值KL, |log_prob - ref_log_prob|
    - 'mse': 均方误差, 0.5 * (log_prob - ref_log_prob)^2
    - 'low_var_kl': 低方差KL估计 (Schulman, 2020)

    这些不同的KL估计方式在不同场景下有各自的优劣:
    - 'kl': 标准实现,简单直接
    - 'abs': 对大偏差更敏感
    - 'mse': 更平滑的梯度
    - 'low_var_kl': 方差更低,训练更稳定

    Args:
        logprob: 当前策略的log概率
        ref_logprob: 参考策略的log概率
        kl_penalty: KL惩罚类型

    Returns:
        KL散度值

    参考: http://joschu.net/blog/kl-approx.html
    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. 近似KL散度, 2020.
    # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # 注意:这里logprob和ref_logprob应该包含词汇表中每个token的logits
        raise NotImplementedError

    raise NotImplementedError
