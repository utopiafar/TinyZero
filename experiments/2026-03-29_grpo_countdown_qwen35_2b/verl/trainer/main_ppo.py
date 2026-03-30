# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
================================================================================
TinyZero PPO 训练主入口
================================================================================

本模块是 TinyZero 项目的 PPO（Proximal Policy Optimization）训练的主入口文件。

【核心功能】
1. 初始化 Ray 分布式计算环境
2. 配置和启动 PPO 训练所需的各个组件（Actor、Critic、Reference Policy）
3. 管理奖励函数的计算和分发
4. 协调训练流程的执行

【调用关系】
本文件 (main_ppo.py)
    │
    ├── 调用 RewardManager 类来计算奖励
    │       └── 调用 verl/utils/reward_score/ 下的奖励函数
    │           ├── countdown.py (倒计时任务奖励)
    │           ├── multiply.py (乘法任务奖励)
    │           ├── gsm8k.py (GSM8K数学题奖励)
    │           └── math.py (MATH数据集奖励)
    │
    ├── 调用 RayPPOTrainer 类来执行训练
    │       └── 位于 verl/trainer/ppo/ray_trainer.py
    │
    └── 调用各个 Worker 类来执行具体任务
            ├── ActorRolloutRefWorker (Actor + Rollout + Reference 策略)
            ├── CriticWorker (Critic 价值网络)
            └── RewardModelWorker (可选的奖励模型)

【训练流程概览】
1. 加载数据集（Parquet格式）
2. Actor 模型生成响应（使用 vLLM 推理引擎）
3. 计算奖励分数（基于规则或奖励模型）
4. Critic 模型估计状态价值
5. 使用 PPO 算法更新 Actor 和 Critic 参数
6. 重复步骤 2-5 直到收敛

注意：我们不将 main 与 ray_trainer 合并，因为 ray_trainer 会被其他主程序复用。
"""

# =============================================================================
# 导入依赖
# =============================================================================

# veRL 核心数据结构，用于在各个组件之间传递数据
from verl import DataProto

# PyTorch 深度学习框架
import torch

# 各任务的奖励计算函数
from verl.utils.reward_score import gsm8k, math, multiply, countdown

# PPO 训练器，基于 Ray 的分布式实现
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


# =============================================================================
# 奖励函数选择器
# =============================================================================

def _select_rm_score_fn(data_source):
    """
    根据数据源选择对应的奖励计算函数。

    【功能说明】
    TinyZero 支持多种数学推理任务，每种任务有不同的奖励计算逻辑。
    此函数根据数据集的来源（data_source）字段，返回对应的奖励计算函数。

    【参数】
    data_source: str
        数据来源标识符，通常来自数据集的 'data_source' 字段
        支持的值：
        - 'openai/gsm8k': GSM8K 小学数学题数据集
        - 'lighteval/MATH': MATH 高中数学竞赛题数据集
        - 包含 'multiply' 或 'arithmetic': 乘法/算术任务
        - 包含 'countdown': 倒计时任务（TinyZero 主要使用此任务）

    【返回值】
    function: 对应的奖励计算函数，签名为 (solution_str, ground_truth) -> float

    【调用示例】
    >>> score_fn = _select_rm_score_fn('countdown')
    >>> score = score_fn(solution_str="(1+2)*3", ground_truth={'target': 9, 'numbers': [1,2,3]})

    【调用关系】
    此函数被 RewardManager.__call__() 调用，用于为每个样本选择正确的奖励函数。
    """
    # GSM8K: 小学数学应用题数据集
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score

    # MATH: 高中数学竞赛题数据集
    elif data_source == 'lighteval/MATH':
        return math.compute_score

    # 乘法/算术任务
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score

    # 倒计时任务（TinyZero 的核心任务）
    elif "countdown" in data_source:
        return countdown.compute_score

    # 不支持的数据源，抛出异常
    else:
        raise NotImplementedError(f"不支持的数据源: {data_source}")


# =============================================================================
# 奖励管理器类
# =============================================================================

class RewardManager():
    """
    奖励管理器 - 负责计算和管理强化学习训练中的奖励信号。

    【核心职责】
    1. 接收模型生成的响应数据
    2. 解码 token IDs 为文本字符串
    3. 调用对应的奖励函数计算分数
    4. 返回奖励张量供 PPO 训练使用

    【在 PPO 训练中的位置】
    Rollout 阶段 (生成响应) → RewardManager (计算奖励) → Critic (估计价值) → PPO 更新

    【调用关系】
    - 被 RayPPOTrainer 初始化时创建
    - 在每个训练步骤中被调用，计算当前批次样本的奖励
    - 内部调用 _select_rm_score_fn() 获取具体任务的奖励函数
    """

    def __init__(self, tokenizer, num_examine) -> None:
        """
        初始化奖励管理器。

        【参数】
        tokenizer: PreTrainedTokenizer
            HuggingFace tokenizer，用于将 token IDs 解码为文本
            通常使用 Actor 模型对应的 tokenizer

        num_examine: int
            打印到控制台的解码响应数量（用于调试和监控）
            设为 0 表示不打印任何样本
            设为正数 N 表示每种数据源打印前 N 个样本
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self._tb_writer = None
        self._step = 0

    def _get_tb_writer(self):
        """延迟初始化 TensorBoard writer（避免 Ray 序列化问题）"""
        if self._tb_writer is None:
            from torch.utils.tensorboard import SummaryWriter
            import os
            log_dir = os.environ.get('GRPO_TB_LOG_DIR', '/root/tf-logs/grpo_samples')
            os.makedirs(log_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=log_dir)
        return self._tb_writer

    def __call__(self, data: DataProto):
        """
        计算一批数据的奖励分数（使类实例可调用）。

        【功能说明】
        此方法是奖励计算的核心入口。它接收一个批次的训练数据，
        对每个样本进行解码和奖励计算。

        【处理流程】
        1. 检查是否已有预计算的奖励分数（某些数据集可能预先计算了奖励）
        2. 初始化奖励张量（全零）
        3. 遍历每个样本：
           a. 提取 prompt 和 response 的 token IDs
           b. 使用 attention_mask 过滤有效的 token
           c. 解码为文本字符串
           d. 获取 ground truth（正确答案）
           e. 选择并调用对应的奖励函数
           f. 将奖励值填入奖励张量
        4. 返回奖励张量

        【参数】
        data: DataProto
            veRL 的数据协议对象，包含一个批次的训练数据
            主要字段：
            - batch['prompts']: prompt 的 token IDs [batch_size, prompt_len]
            - batch['responses']: response 的 token IDs [batch_size, response_len]
            - batch['attention_mask']: 注意力掩码 [batch_size, total_len]
            - non_tensor_batch['reward_model']['ground_truth']: 正确答案
            - non_tensor_batch['data_source']: 数据来源标识

        【返回值】
        torch.Tensor: 奖励张量，形状为 [batch_size, response_len]
                      奖励值放在每个样本的最后一个有效 token 位置
                      其他位置为 0（稀疏奖励，只在序列结束时给奖励）
        """
        # ----------------------------------------------------------------------
        # 步骤 1: 检查是否有预计算的奖励分数
        # ----------------------------------------------------------------------
        # 某些场景下（如使用神经网络奖励模型），奖励可能在之前的步骤中已经计算过了
        # 如果存在 'rm_scores' 字段，直接返回，避免重复计算
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # ----------------------------------------------------------------------
        # 步骤 2: 初始化奖励张量
        # ----------------------------------------------------------------------
        # 创建一个与 responses 形状相同的全零张量
        # 使用 float32 类型以支持小数奖励值
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # 用于跟踪每种数据源已打印的样本数
        already_print_data_sources = {}

        # ----------------------------------------------------------------------
        # 步骤 3: 遍历每个样本计算奖励
        # ----------------------------------------------------------------------
        for i in range(len(data)):
            data_item = data[i]  # 获取单个样本

            # ------------------------------------------------------------------
            # 3.1 提取 prompt 相关信息
            # ------------------------------------------------------------------
            # prompt_ids: 输入提示的 token IDs
            prompt_ids = data_item.batch['prompts']

            # prompt_length: prompt 的总长度（包括 padding）
            prompt_length = prompt_ids.shape[-1]

            # 使用 attention_mask 计算 prompt 的有效长度（排除 padding）
            # attention_mask 的前 prompt_length 个位置对应 prompt
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()

            # 提取有效的 prompt token IDs（去除 padding）
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            # ------------------------------------------------------------------
            # 3.2 提取 response 相关信息
            # ------------------------------------------------------------------
            # response_ids: 模型生成的响应 token IDs
            response_ids = data_item.batch['responses']

            # 使用 attention_mask 计算 response 的有效长度（排除 padding）
            # attention_mask 的 prompt_length 之后的位置对应 response
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            # 提取有效的 response token IDs（去除 padding）
            valid_response_ids = response_ids[:valid_response_length]

            # ------------------------------------------------------------------
            # 3.3 解码 token IDs 为文本字符串
            # ------------------------------------------------------------------
            # 将 prompt 和 response 拼接后解码
            # 这样奖励函数可以看到完整的对话内容
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            # ------------------------------------------------------------------
            # 3.4 获取正确答案（ground truth）
            # ------------------------------------------------------------------
            # ground_truth 包含任务特定的正确答案信息
            # 例如对于 countdown 任务：{'target': 100, 'numbers': [25, 4, 3, 1]}
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # ------------------------------------------------------------------
            # 3.5 选择并调用奖励函数
            # ------------------------------------------------------------------
            # 根据数据源选择对应的奖励计算函数
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)

            # 计算奖励分数
            # score 通常是 0（完全错误）、0.1（格式正确但答案错误）、1（完全正确）
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)

            # ------------------------------------------------------------------
            # 3.6 将奖励值填入奖励张量
            # ------------------------------------------------------------------
            # 使用稀疏奖励：只在 response 的最后一个 token 位置设置奖励值
            # 这是强化学习的常见做法，奖励只在序列结束时给出
            reward_tensor[i, valid_response_length - 1] = score

            # ------------------------------------------------------------------
            # 3.7 可选：打印样本用于调试
            # ------------------------------------------------------------------
            # 初始化计数器（如果该数据源第一次出现）
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # 如果还未达到打印数量限制，打印该样本
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # ----------------------------------------------------------------------
        # 步骤 4: 返回奖励张量
        # ----------------------------------------------------------------------
        # TensorBoard text logging: 每 10 步记录 2 个样本预测
        try:
            if self._step % 10 == 0:
                writer = self._get_tb_writer()
                sample_indices = list(range(len(data)))
                import random as _rand
                _rand.shuffle(sample_indices)
                for j in sample_indices[:min(2, len(data))]:
                    di = data[j]
                    p_ids = di.batch['prompts']
                    p_len = p_ids.shape[-1]
                    v_p_len = di.batch['attention_mask'][:p_len].sum()
                    v_p_ids = p_ids[-v_p_len:]
                    r_ids = di.batch['responses']
                    v_r_len = di.batch['attention_mask'][p_len:].sum()
                    v_r_ids = r_ids[:v_r_len]
                    prompt_str = self.tokenizer.decode(v_p_ids, skip_special_tokens=False)
                    resp_str = self.tokenizer.decode(v_r_ids, skip_special_tokens=False)
                    gt = di.non_tensor_batch['reward_model']['ground_truth']
                    ds = di.non_tensor_batch['data_source']
                    sc = reward_tensor[j, v_r_len - 1].item()
                    sample_md = (
                        f"## Step {self._step} Sample {j}\n\n"
                        f"**Data Source:** {ds}  |  **Reward:** {sc:.2f}\n\n"
                        f"**Ground Truth:** `{gt}`\n\n"
                        f"**Prompt (tail):**\n```\n{prompt_str[-200:]}\n```\n\n"
                        f"**Response:**\n```\n{resp_str[:500]}\n```\n"
                    )
                    writer.add_text(f'samples/step_{self._step}', sample_md, self._step)
                writer.flush()
        except Exception as e:
            if self._step < 3:
                print(f"[TB text logging] {e}")
        self._step += 1

        return reward_tensor


# =============================================================================
# Ray 和 Hydra 导入
# =============================================================================

# Ray: 分布式计算框架，用于协调多 GPU 训练
import ray

# Hydra: 配置管理框架，用于加载和管理训练配置
import hydra


# =============================================================================
# 主入口函数
# =============================================================================

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    """
    PPO 训练的主入口函数。

    【功能说明】
    此函数是整个训练流程的起点。它使用 Hydra 装饰器自动加载配置文件，
    然后初始化 Ray 分布式环境，最后将主任务提交给 Ray 执行。

    【Hydra 装饰器说明】
    @hydra.main 装饰器会：
    1. 从 config/ 目录加载 ppo_trainer.yaml 配置文件
    2. 解析命令行参数覆盖配置
    3. 将最终配置作为 config 参数传入函数

    【参数】
    config: DictConfig
        由 Hydra 自动注入的配置对象
        包含训练所需的所有超参数和设置

    【执行流程】
    1. 检查 Ray 是否已初始化
    2. 如果未初始化，启动本地 Ray 集群
    3. 将 main_task 作为远程任务提交给 Ray 执行
    4. 等待任务完成（ray.get 会阻塞）

    【调用关系】
    命令行执行 python -m verl.trainer.main_ppo
        → main() [本函数]
            → ray.init() [初始化 Ray]
            → main_task.remote() [远程执行训练任务]
    """
    # 检查 Ray 是否已经初始化（可能在外部已启动）
    if not ray.is_initialized():
        # 初始化 Ray 运行时环境
        # TOKENIZERS_PARALLELISM: 启用 tokenizer 并行处理以提高效率
        # NCCL_DEBUG: 设置为 WARN 以减少 NCCL（GPU通信库）的调试输出
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    # 将 main_task 作为 Ray 远程任务执行
    # ray.get() 会阻塞等待任务完成
    ray.get(main_task.remote(config))


# =============================================================================
# 主训练任务（Ray 远程任务）
# =============================================================================

@ray.remote
def main_task(config):
    """
    执行 PPO 训练的主任务（作为 Ray 远程 Actor 运行）。

    【功能说明】
    此函数包含训练的核心逻辑，作为 Ray 远程任务执行。
    它负责：
    1. 加载和准备模型/分词器
    2. 配置各个 Worker（Actor、Critic、Reference Policy）
    3. 创建奖励函数
    4. 初始化并启动训练器

    【Ray 远程任务说明】
    @ray.remote 装饰器将此函数转换为 Ray Actor，
    使其可以在集群中的任意节点上执行，并支持分布式计算。

    【详细执行流程】

    参数:
        config: DictConfig - 训练配置对象
    """
    # ==========================================================================
    # 阶段 1: 导入依赖并打印配置
    # ==========================================================================
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # 用于打印配置信息
    from pprint import pprint
    from omegaconf import OmegaConf

    # 打印完整的配置信息（用于调试和记录）
    # resolve=True 会解析配置中的变量引用（如 ${data.max_prompt_length}）
    pprint(OmegaConf.to_container(config, resolve=True))

    # 解析配置中的所有变量引用
    OmegaConf.resolve(config)

    # ==========================================================================
    # 阶段 2: 加载模型和分词器
    # ==========================================================================
    # 如果模型路径是 HDFS 路径，会自动下载到本地
    # 返回本地模型路径
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # 实例化分词器
    # hf_tokenizer 是 veRL 封装的 HuggingFace tokenizer 加载函数
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # ==========================================================================
    # 阶段 3: 配置 Worker 类（根据分布式策略选择）
    # ==========================================================================
    # 根据配置选择使用 FSDP 还是 Megatron-LM 作为分布式后端

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        # ----------------------------------------------------------------------
        # FSDP (Fully Sharded Data Parallel) 后端
        # ----------------------------------------------------------------------
        # FSDP 是 PyTorch 原生的分布式训练策略
        # 优点：易于使用，与 HuggingFace 模型兼容性好
        # 适用场景：中小规模模型（< 10B 参数）
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy

        # 导入 FSDP 后端的 Worker 类
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        # ----------------------------------------------------------------------
        # Megatron-LM 后端
        # ----------------------------------------------------------------------
        # Megatron-LM 是 NVIDIA 开发的大模型训练框架
        # 优点：高度优化，支持张量并行和流水线并行
        # 适用场景：大规模模型（> 10B 参数）
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy

        # 导入 Megatron 后端的 Worker 类
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError(f"不支持的分布式策略: {config.actor_rollout_ref.actor.strategy}")

    # ==========================================================================
    # 阶段 4: 配置角色-工作进程映射
    # ==========================================================================
    # PPO 训练需要多个模型协同工作，每个模型由一个 Worker 管理

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    # 定义角色到 Worker 类的映射
    # Role.ActorRollout:  Actor 模型（生成响应） + Rollout（采样）
    # Role.Critic:        Critic 模型（估计状态价值）
    # Role.RefPolicy:     Reference 策略（用于 KL 散度约束，防止模型偏离太远）
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)  # RefPolicy 复用 Actor 的 Worker 类
    }

    # ==========================================================================
    # 阶段 5: 配置资源池
    # ==========================================================================
    # 定义 GPU 资源如何在各个 Worker 之间分配

    # 资源池标识符
    global_pool_id = 'global_pool'

    # 资源池规格：定义每个节点有多少 GPU
    # [n_gpus_per_node] * nnodes 表示每个节点的 GPU 数量
    # 例如：2 个节点，每节点 4 GPU → [4, 4]
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }

    # 角色到资源池的映射：定义每个角色使用哪个资源池
    # 这里所有角色共享同一个资源池（即所有 GPU）
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # ==========================================================================
    # 阶段 6: 可选 - 配置奖励模型
    # ==========================================================================
    # TinyZero 默认使用基于规则的奖励函数，但也可以启用神经网络奖励模型

    # 关于多源奖励函数的设计说明：
    # - 基于规则的奖励（rule-based RM）：直接调用奖励计算函数（如 countdown.compute_score）
    # - 基于模型的奖励（model-based RM）：调用一个神经网络来预测奖励
    # - 代码相关提示：发送到沙箱环境执行测试用例
    # - 最终将所有奖励组合在一起
    # - 奖励类型取决于数据的标签

    if config.reward_model.enable:
        # 根据分布式策略导入对应的 Reward Model Worker
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError(f"不支持的奖励模型策略: {config.reward_model.strategy}")

        # 将 Reward Model Worker 添加到映射中
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # ==========================================================================
    # 阶段 7: 创建奖励函数
    # ==========================================================================
    # 创建训练时使用的奖励管理器
    # num_examine=0 表示训练时不打印样本（减少输出）
    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # 创建验证时使用的奖励管理器
    # num_examine=1 表示验证时打印每种数据源的 1 个样本（用于调试）
    # 注意：验证时始终使用基于规则的奖励函数
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    # ==========================================================================
    # 阶段 8: 创建并启动训练器
    # ==========================================================================
    # 创建资源池管理器
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    # 创建 PPO 训练器
    # 参数说明：
    # - config: 训练配置
    # - tokenizer: 分词器
    # - role_worker_mapping: 角色-Worker 映射
    # - resource_pool_manager: 资源池管理器
    # - ray_worker_group_cls: Ray Worker Group 类
    # - reward_fn: 训练时的奖励函数
    # - val_reward_fn: 验证时的奖励函数
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)

    # 初始化所有 Worker（加载模型到 GPU）
    trainer.init_workers()

    # 开始训练
    # fit() 方法会执行完整的训练循环，包括：
    # 1. 加载数据
    # 2. 生成响应（Rollout）
    # 3. 计算奖励
    # 4. 估计价值（Critic）
    # 5. 更新策略（PPO）
    # 6. 验证和保存检查点
    trainer.fit()


# =============================================================================
# 程序入口点
# =============================================================================

if __name__ == '__main__':
    """
    程序入口点。

    【执行方式】
    python -m verl.trainer.main_ppo [Hydra 配置覆盖参数]

    【示例】
    # 使用默认配置
    python -m verl.trainer.main_ppo

    # 覆盖配置参数
    python -m verl.trainer.main_ppo trainer.total_epochs=10 trainer.n_gpus_per_node=2

    【通过脚本执行】
    bash ./scripts/train_tiny_zero.sh
    """
    main()
