"""
================================================================================
Countdown 任务数据预处理脚本
================================================================================

【模块概述】
本脚本用于生成和处理 Countdown（倒计时）任务的训练数据。
它将原始数据集转换为 veRL 框架所需的 Parquet 格式。

【Countdown 任务说明】
给定一组数字和一个目标值，使用四则运算构造等于目标值的等式。
这是 TinyZero 项目用于训练模型推理能力的核心任务。

【数据来源】
默认使用 HuggingFace 上的 'Jiayi-Pan/Countdown-Tasks-3to4' 数据集，
该数据集包含预生成的 Countdown 问题。

【输出格式】
生成的 Parquet 文件包含以下字段：
- data_source: 数据来源标识（'countdown'）
- prompt: 对话格式的提示信息
- ability: 能力标签（'math'）
- reward_model: 奖励模型配置（包含正确答案）
- extra_info: 额外元数据

【使用方法】
python examples/data_preprocess/countdown.py \
    --local_dir ~/data/countdown \
    --train_size 327680 \
    --test_size 1024 \
    --template_type base

【调用关系】
本脚本生成的数据 → train.parquet / test.parquet
                           ↓
                 train_tiny_zero.sh (通过 $DATA_DIR 引用)
                           ↓
                 main_ppo.py (data.train_files 参数)
"""

# =============================================================================
# 导入依赖
# =============================================================================

import re           # 正则表达式（本脚本未使用）
import os           # 操作系统接口，用于文件路径操作
from datasets import Dataset, load_dataset  # HuggingFace 数据集库
from random import randint, seed, choice    # 随机数生成
from typing import List, Tuple              # 类型注解
from tqdm import tqdm                       # 进度条显示
from verl.utils.hdfs_io import copy, makedirs  # HDFS 文件操作（分布式存储）
import argparse     # 命令行参数解析


# =============================================================================
# 辅助函数：生成数据集
# =============================================================================

def gen_dataset(
    num_samples: int,
    num_operands: int = 6,
    max_target: int = 1000,
    min_number: int = 1,
    max_number: int = 100,
    operations: List[str] = ['+', '-', '*', '/'],
    seed_value: int = 42,
) -> List[Tuple]:
    """
    生成 Countdown 任务的随机数据集。

    【功能说明】
    随机生成指定数量的 Countdown 问题。
    每个问题包含一个目标值和一组可用数字。

    【注意】
    此函数生成的数据不保证有解！
    当前脚本实际上使用的是外部数据集，此函数可能未被调用。

    【参数】
    num_samples: int
        要生成的样本数量

    num_operands: int, 默认 6
        每个样本中可用数字的数量

    max_target: int, 默认 1000
        目标值的最大值（最小值为 1）

    min_number: int, 默认 1
        可用数字的最小值

    max_number: int, 默认 100
        可用数字的最大值

    operations: List[str], 默认 ['+', '-', '*', '/']
        允许使用的运算符列表（当前未使用）

    seed_value: int, 默认 42
        随机种子，用于可重复性

    【返回值】
    List[Tuple]: 样本列表，每个样本是 (target, numbers) 元组
    """
    # 设置随机种子以确保可重复性
    seed(seed_value)
    samples = []

    # 生成指定数量的样本
    for _ in tqdm(range(num_samples)):
        # 随机生成目标值
        target = randint(1, max_target)

        # 随机生成可用数字
        numbers = [randint(min_number, max_number) for _ in range(num_operands)]

        samples.append((target, numbers))

    return samples


# =============================================================================
# 辅助函数：生成提示模板
# =============================================================================

def make_prefix(dp, template_type):
    """
    根据数据样本生成对话提示前缀。

    【功能说明】
    根据目标值和可用数字，生成完整的问题提示。
    支持不同的模板格式以适配不同的基座模型。

    【参数】
    dp: dict
        数据样本，包含：
        - 'target': 目标值
        - 'nums': 可用数字列表

    template_type: str
        模板类型：
        - 'base': 通用格式，适用于大多数基础模型
        - 'qwen-instruct': Qwen Instruct 模型的对话格式

    【返回值】
    str: 格式化的对话提示字符串

    【输出格式说明】
    提示包含以下关键部分：
    1. 任务描述：说明要做什么
    2. 约束条件：只能使用给定的数字和运算符，每个数字只用一次
    3. 输出格式要求：
       - 使用 <think reasoning>...</think reasoning> 展示思考过程
       - 使用 <answer>...</answer> 给出最终答案
    4. 预填充的开头："Let me solve this step by step." 和 <think reasoning> 标签

    【预填充策略】
    预填充 "Assistant: Let me solve this step by step. <think reasoning>"
    可以引导模型以结构化的方式开始回答，这是常见的提示工程技巧。
    """
    # 从数据样本中提取目标值和数字
    target = dp['target']
    numbers = dp['nums']

    # 注意：如果修改提示格式，也需要同步修改 reward_score/countdown.py 中的解析逻辑

    if template_type == 'base':
        # ----------------------------------------------------------------------
        # 通用模板格式
        # ----------------------------------------------------------------------
        # 适用于大多数基础语言模型
        # 格式："User: ... Assistant: ..."
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think reasoning> </think reasoning> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think reasoning>"""

    elif template_type == 'qwen-instruct':
        # ----------------------------------------------------------------------
        # Qwen Instruct 模板格式
        # ----------------------------------------------------------------------
        # 适用于 Qwen Instruct 系列模型
        # 使用 ChatML 格式：<|im_start|>role\ncontent<|im_end|>
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think reasoning> </think reasoning> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think reasoning>"""

    elif template_type == 'qwen3':
        # ----------------------------------------------------------------------
        # Qwen3 模板格式（中文 + few-shot）
        # ----------------------------------------------------------------------
        # 适用于 Qwen3 系列基座模型
        # 使用 ChatML 格式 + Qwen3 原生 <think/</think token
        # 包含 few-shot 示例引导模型输出 <answer> 标签
        prefix = f"""<|im_start|>system
你是一个数学助手。你会先仔细思考推理过程，然后给出最终答案。<|im_end|>
<|im_start|>user
给定数字 [3, 7, 5, 2] 和目标值 24，请使用加减乘除运算得到目标值。每个数字只能用一次。
先思考推理过程，最终答案放在 <answer> </answer> 标签中。<|im_end|>
<|im_start|>assistant
<think
我需要用 3、7、5、2 得到 24，每个数字必须用一次。
先试乘法：3 * 7 = 21，然后 5 - 2 = 3，21 + 3 = 24！
验证：3 * 7 + 5 - 2 = 21 + 5 - 2 = 24。正确！
</think >
<answer> 3 * 7 + 5 - 2 </answer><|im_end|>
<|im_start|>user
给定数字 {numbers} 和目标值 {target}，请使用加减乘除运算得到目标值。每个数字只能使用一次。
先思考推理过程，最终答案放在 <answer> </answer> 标签中。<|im_end|>
<|im_start|>assistant
<think
"""

    elif template_type == 'qwen3.5':
        # ----------------------------------------------------------------------
        # Qwen3.5 模板格式（中文 + few-shot + ChatML）
        # ----------------------------------------------------------------------
        # 适用于 Qwen3.5-2B-Base 等基座模型
        # 使用 ChatML 格式（<|im_start|> / <|im_end|>）
        # 使用 <think reasoning> 作为思考标记（避免触发特殊 token）
        # 包含 few-shot 示例引导模型输出 <answer> 标签
        prefix = f"""<|im_start|>system
你是一个有帮助的助手。你会先在脑海中思考推理过程，然后给出答案。<|im_end|>
<|im_start|>user
使用数字 [3, 7, 5, 2]，通过基本算术运算（+、-、*、/）构造一个等于 24 的等式。每个数字只能使用一次。请在 <think reasoning> </think reasoning> 标签中展示你的推理过程，并在 <answer> </answer> 标签中返回最终答案，例如 <answer> (1 + 2) / 3 </answer>。<|im_end|>
<|im_start|>assistant
<think reasoning>
3 * 7 = 21，然后 5 - 2 = 3，21 + 3 = 24。
验证：3 * 7 + 5 - 2 = 21 + 5 - 2 = 24。正确！
</think reasoning>
<answer> 3 * 7 + 5 - 2 </answer><|im_end|>
<|im_start|>user
使用数字 {numbers}，通过基本算术运算（+、-、*、/）构造一个等于 {target} 的等式。每个数字只能使用一次。请在 <think reasoning> </think reasoning> 标签中展示你的推理过程，并在 <answer> </answer> 标签中返回最终答案，例如 <answer> (1 + 2) / 3 </answer>。<|im_end|>
<|im_start|>assistant
<think reasoning>
"""

    return prefix


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    # ==========================================================================
    # 阶段 1: 解析命令行参数
    # ==========================================================================
    parser = argparse.ArgumentParser(
        description='生成 Countdown 任务的训练数据'
    )

    # 输出目录参数
    parser.add_argument('--local_dir', default='~/data/countdown',
                        help='本地输出目录')

    parser.add_argument('--hdfs_dir', default=None,
                        help='HDFS 输出目录（可选，用于分布式存储）')

    # 数据生成参数（以下参数在当前实现中未被使用，因为使用的是外部数据集）
    parser.add_argument('--num_samples', type=int, default=100000,
                        help='要生成的样本数量（未使用）')

    parser.add_argument('--num_operands', type=int, default=6,
                        help='每个样本的数字数量（未使用）')

    parser.add_argument('--max_target', type=int, default=1000,
                        help='目标值最大值（未使用）')

    parser.add_argument('--min_number', type=int, default=1,
                        help='数字最小值（未使用）')

    parser.add_argument('--max_number', type=int, default=100,
                        help='数字最大值（未使用）')

    # 数据集划分参数
    parser.add_argument('--train_size', type=int, default=327680,
                        help='训练集大小')

    parser.add_argument('--test_size', type=int, default=1024,
                        help='测试集大小')

    # 模板类型参数
    parser.add_argument('--template_type', type=str, default='base',
                        choices=['base', 'qwen-instruct', 'qwen3', 'qwen3.5'],
                        help='提示模板类型')

    args = parser.parse_args()

    # ==========================================================================
    # 阶段 2: 加载原始数据集
    # ==========================================================================
    # 数据来源标识（用于后续的奖励函数选择）
    data_source = 'countdown'

    # 训练集和测试集大小
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    # 从 HuggingFace 加载预生成的 Countdown 数据集
    # 该数据集包含 3-4 个数字的 Countdown 问题
    raw_dataset = load_dataset('Jiayi-Pan/Countdown-Tasks-3to4', split='train')

    # ==========================================================================
    # 阶段 3: 划分训练集和测试集
    # ==========================================================================
    # 确保数据集足够大
    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE, \
        f"数据集太小：需要 {TRAIN_SIZE + TEST_SIZE}，实际 {len(raw_dataset)}"

    # 划分训练集（前 TRAIN_SIZE 个样本）
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))

    # 划分测试集（接下来的 TEST_SIZE 个样本）
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    # ==========================================================================
    # 阶段 4: 定义数据处理函数
    # ==========================================================================
    def make_map_fn(split):
        """
        创建数据集映射函数的工厂函数。

        【功能说明】
        返回一个处理函数，用于将原始数据转换为 veRL 所需的格式。

        【参数】
        split: str - 数据划分类型 ('train' 或 'test')

        【返回值】
        function: 处理函数，接受 (example, idx) 参数
        """
        def process_fn(example, idx):
            """
            处理单个数据样本。

            【输入格式】
            example: 原始数据样本
                - 'target': 目标值
                - 'nums': 可用数字列表

            【输出格式】
            dict: veRL 数据格式
            """
            # 生成格式化的提示文本
            question = make_prefix(example, template_type=args.template_type)

            # 构造正确答案（ground truth）
            # 这将在奖励计算时使用
            solution = {
                "target": example['target'],
                "numbers": example['nums']
            }

            # 构建 veRL 数据格式
            data = {
                # 数据来源标识（用于选择奖励函数）
                "data_source": data_source,

                # 对话格式的提示
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],

                # 能力标签
                "ability": "math",

                # 奖励模型配置
                "reward_model": {
                    "style": "rule",  # 使用基于规则的奖励
                    "ground_truth": solution  # 正确答案
                },

                # 额外信息（用于调试和日志）
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    # ==========================================================================
    # 阶段 5: 应用数据处理
    # ==========================================================================
    # 使用 map 函数批量处理数据集
    # with_indices=True 表示处理函数会接收样本索引

    print(f"正在处理训练集 ({TRAIN_SIZE} 样本)...")
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)

    print(f"正在处理测试集 ({TEST_SIZE} 样本)...")
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # ==========================================================================
    # 阶段 6: 保存数据集
    # ==========================================================================
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 确保本地目录存在
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    # 保存为 Parquet 格式（高效的列式存储格式）
    train_path = os.path.join(local_dir, 'train.parquet')
    test_path = os.path.join(local_dir, 'test.parquet')

    print(f"保存训练集到: {train_path}")
    train_dataset.to_parquet(train_path)

    print(f"保存测试集到: {test_path}")
    test_dataset.to_parquet(test_path)

    # ==========================================================================
    # 阶段 7: 可选 - 复制到 HDFS
    # ==========================================================================
    # 如果指定了 HDFS 目录，将数据复制到分布式存储
    if hdfs_dir is not None:
        print(f"复制数据到 HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

    print("数据处理完成！")
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
