"""
================================================================================
GPU GRPO 最小循环测试 — Countdown 任务
================================================================================

验证 GRPO 训练在 GPU 上的完整流程：
1. 数据加载与预处理
2. 模型推理（生成响应）
3. 奖励计算（Countdown 任务）
4. GRPO 优势估计
5. PPO 策略更新（带梯度）

python gpu_grpo_test.py \
    --model_path ~/models/Qwen3-4B \
    --data_path ~/data/countdown/train.parquet \
    --num_episodes 2 \
    --batch_size 4 \
    --n_samples 2
"""

import argparse
import os
import re
import random
import json
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='GPU GRPO 最小循环测试')
    parser.add_argument('--model_path', type=str, default='~/models/Qwen3-4B')
    parser.add_argument('--data_path', type=str, default='~/data/countdown/train.parquet')
    parser.add_argument('--num_episodes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_samples', type=int, default=2)
    parser.add_argument('--max_prompt_length', type=int, default=256)
    parser.add_argument('--max_response_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--kl_coef', type=float, default=0.001)
    parser.add_argument('--ppo_epochs', type=int, default=1)
    return parser.parse_args()


# =============================================================================
# Countdown 奖励函数
# =============================================================================

def extract_solution(solution_str):
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]
    matches = list(re.finditer(r'<answer>(.*?)</answer>', solution_str))
    return matches[-1].group(1).strip() if matches else None


def validate_equation(equation_str, available_numbers):
    try:
        numbers_in_eq = sorted(int(n) for n in re.findall(r'\d+', equation_str))
        return numbers_in_eq == sorted(available_numbers)
    except Exception:
        return False


def evaluate_equation(equation_str):
    try:
        if not re.match(r'^[\d+\-*/().\s]+$', equation_str):
            return None
        return eval(equation_str, {"__builtins__": None}, {})
    except Exception:
        return None


def compute_score(solution_str, ground_truth):
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    equation = extract_solution(solution_str)
    if equation is None:
        return 0.0
    if not validate_equation(equation, numbers):
        return 0.1
    result = evaluate_equation(equation)
    if result is None:
        return 0.1
    return 1.0 if abs(result - target) < 1e-5 else 0.1


# =============================================================================
# GRPO 核心算法
# =============================================================================

def compute_grpo_advantage(rewards, n_samples, epsilon=1e-6):
    """GRPO 组归一化优势"""
    B = rewards.shape[0] // n_samples
    advantages = torch.zeros_like(rewards)
    for i in range(B):
        g = rewards[i * n_samples:(i + 1) * n_samples]
        advantages[i * n_samples:(i + 1) * n_samples] = (g - g.mean()) / (g.std() + epsilon)
    return advantages


def compute_log_probs(model, input_ids, response_len):
    """
    前向传播计算 response 部分每个 token 的 log_prob。
    input_ids: [1, seq_len]  包含 prompt + response
    返回: sum of log_probs for response tokens (标量，可求导)
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]  # [1, seq_len-1, vocab]
    targets = input_ids[:, 1:]          # [1, seq_len-1]

    # 只取 response 部分的 log_probs
    prompt_len = input_ids.shape[1] - response_len
    response_logits = logits[:, prompt_len - 1:, :]  # 从 prompt 末尾开始
    response_targets = targets[:, prompt_len - 1:]

    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(2, response_targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum()


# =============================================================================
# 数据集
# =============================================================================

class CountdownDataset(Dataset):
    def __init__(self, parquet_path):
        self.data = pd.read_parquet(os.path.expanduser(parquet_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt_text = row['prompt'][0]['content'] if isinstance(row['prompt'], list) else row['prompt']
        gt = row['reward_model']['ground_truth'] if isinstance(row['reward_model'], dict) else json.loads(row['reward_model'])['ground_truth']
        return {'prompt_text': prompt_text, 'ground_truth': gt}


# =============================================================================
# 主训练
# =============================================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"GRPO 最小循环测试 | Device: {device}")
    print("=" * 60)

    # --- 加载模型 ---
    model_path = os.path.expanduser(args.model_path)
    print(f"\n[1/5] 加载模型: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

    # 参考模型（冻结，用于 KL）
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {total_params / 1e9:.2f}B")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # --- 加载数据 ---
    print(f"\n[2/5] 加载数据: {args.data_path}")
    dataset = CountdownDataset(args.data_path)

    def collate_fn(batch):
        return {k: [d[k] for d in batch] for k in batch[0]}

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    print(f"  数据集: {len(dataset)} 样本, batch_size={args.batch_size}, n={args.n_samples}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- 训练循环 ---
    print(f"\n[3/5] 开始 GRPO 训练 ({args.num_episodes} episodes)")
    print("-" * 60)

    for episode in range(args.num_episodes):
        ep_rewards, ep_losses = [], []

        for batch_idx, batch in enumerate(dataloader):
            print(f"\n  Episode {episode+1}/{args.num_episodes}, "
                  f"Batch {batch_idx+1}/{len(dataloader)}")

            prompts = batch['prompt_text']
            ground_truths = batch['ground_truth']

            all_input_ids = []
            all_response_lens = []
            all_old_log_probs = []
            all_ref_log_probs = []
            all_rewards = []

            for p_idx, prompt in enumerate(prompts):
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=args.max_prompt_length).to(device)
                prompt_len = inputs.input_ids.shape[1]

                for _ in range(args.n_samples):
                    # --- 生成 ---
                    model.eval()
                    with torch.no_grad():
                        output = model.generate(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_new_tokens=args.max_response_length,
                            do_sample=True, temperature=1.0, top_p=1.0,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    response_ids = output[0, prompt_len:]
                    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
                    response_len = response_ids.shape[0]

                    # 奖励
                    full_text = prompt + response_text
                    reward = compute_score(full_text, ground_truths[p_idx])
                    all_rewards.append(reward)

                    full_ids = output[:, :prompt_len + response_len].to(device)

                    # 旧策略 log_prob（无梯度）
                    model.eval()
                    with torch.no_grad():
                        old_lp = compute_log_probs(model, full_ids, response_len)
                    all_old_log_probs.append(old_lp.detach())

                    # 参考策略 log_prob（无梯度）
                    with torch.no_grad():
                        ref_lp = compute_log_probs(ref_model, full_ids, response_len)
                    all_ref_log_probs.append(ref_lp.detach())

                    all_input_ids.append(full_ids)
                    all_response_lens.append(response_len)

                    # 打印第一个 prompt 的第一个样本
                    if p_idx == 0 and len(all_rewards) == 1:
                        print(f"    [Prompt]: ...{prompt[-60:]}")
                        print(f"    [Response]: {response_text[:80]}...")
                        print(f"    [Reward]: {reward}")

            # --- GRPO 优势 ---
            rewards_t = torch.tensor(all_rewards, dtype=torch.float32, device=device)
            old_lp_t = torch.stack(all_old_log_probs)
            ref_lp_t = torch.stack(all_ref_log_probs)
            advantages = compute_grpo_advantage(rewards_t, args.n_samples)

            print(f"    Rewards: {all_rewards}")
            print(f"    Advantages: {advantages.tolist()}")

            # --- PPO 更新（带梯度） ---
            model.train()
            for ppo_ep in range(args.ppo_epochs):
                new_log_probs = []
                for ids, rlen in zip(all_input_ids, all_response_lens):
                    lp = compute_log_probs(model, ids, rlen)
                    new_log_probs.append(lp)
                new_lp_t = torch.stack(new_log_probs)

                # PPO clipped loss
                ratio = torch.exp(new_lp_t - old_lp_t)
                clipped = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
                ppo_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

                # KL 惩罚
                kl = args.kl_coef * (new_lp_t - ref_lp_t).mean()

                loss = ppo_loss + kl
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                ep_losses.append(loss.item())

            avg_reward = rewards_t.mean().item()
            ep_rewards.append(avg_reward)
            print(f"    Loss: {loss.item():.4f} (PPO: {ppo_loss.item():.4f}, KL: {kl.item():.6f}), "
                  f"Avg Reward: {avg_reward:.4f}")

            # 清理显存
            del all_input_ids, all_old_log_probs, all_ref_log_probs
            torch.cuda.empty_cache()

        avg_r = sum(ep_rewards) / len(ep_rewards)
        avg_l = sum(ep_losses) / len(ep_losses)
        print(f"\n  >> Episode {episode+1} 完成: Avg Reward={avg_r:.4f}, Avg Loss={avg_l:.4f}")

    # --- 验证 ---
    print(f"\n[4/5] 验证（贪心解码 3 样本）")
    model.eval()
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        inputs = tokenizer(sample['prompt_text'], return_tensors="pt",
                           truncation=True, max_length=args.max_prompt_length).to(device)
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids, attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_response_length, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        resp = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        reward = compute_score(sample['prompt_text'] + resp, sample['ground_truth'])
        print(f"\n  [{i+1}] Target: {sample['ground_truth']}")
        print(f"      Response: {resp[:100]}...")
        print(f"      Reward: {reward}")

    print(f"\n[5/5] 全部测试完成！")
    print("=" * 60)
    print("GRPO GPU 最小循环验证通过")
    print("  数据加载 | 模型推理 | 奖励计算 | GRPO 优势 | PPO 更新")
    print("=" * 60)


if __name__ == '__main__':
    train(parse_args())
