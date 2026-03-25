"""
================================================================================
Countdown 任务奖励计算模块
================================================================================

【模块概述】
本模块实现了 Countdown（倒计时）任务的奖励计算逻辑。
Countdown 是一个数学推理游戏：给定一组数字和一个目标值，
玩家需要使用四则运算（+、-、*、/）构造一个等于目标值的等式。

【任务示例】
输入：
  - 目标值 (target): 100
  - 可用数字 (numbers): [25, 4, 3, 1]

正确答案示例：
  - 25 * 4 = 100
  - (25 + 3) * 4 - 12 = 100  ← 错误！使用了不存在的数字12

【奖励设计】
本模块使用分级奖励策略：
  - 1.0 (score): 完全正确 - 答案格式正确且计算结果等于目标值
  - 0.1 (format_score): 格式正确但答案错误 - 能提取出方程，但结果不对或使用了错误的数字
  - 0: 格式错误 - 无法从模型输出中提取出有效答案

这种分级奖励设计可以帮助模型逐步学习：
1. 首先学会正确的输出格式（使用 <answer> 标签）
2. 然后学会使用正确的数字
3. 最后学会计算出正确的结果

【调用关系】
main_ppo.py
    └── RewardManager.__call__()
        └── _select_rm_score_fn()
            └── countdown.compute_score()  [本模块]
                ├── extract_solution()     提取答案
                ├── validate_equation()    验证数字使用
                └── evaluate_equation()    计算结果
"""

# =============================================================================
# 导入依赖
# =============================================================================

import re       # 正则表达式，用于提取答案
import random   # 随机数，用于控制调试输出的频率
import ast      # 抽象语法树（本模块未使用，可删除）
import operator # 运算符（本模块未使用，可删除）


# =============================================================================
# 核心函数：提取答案
# =============================================================================

def extract_solution(solution_str):
    """
    从模型生成的解决方案字符串中提取数学方程。

    【功能说明】
    模型生成的输出通常包含思考过程和最终答案。
    此函数从输出中提取 <answer>...</answer> 标签内的方程。

    【处理流程】
    1. 定位 Assistant 的回复部分（去除 User 的提问）
    2. 提取最后一个 <answer> 标签中的内容
    3. 清理并返回方程字符串

    【支持的对话格式】
    - 通用格式: "User: ... Assistant: ..."
    - Qwen 格式: "<|im_start|>user...<|im_end|><|im_start|>assistant..."

    【参数】
    solution_str: str
        完整的对话文本，包含 User 的提问和 Assistant 的回答
        例如：
        "User: Using the numbers [25, 4, 3, 1], create an equation...
         Assistant: Let me solve this step by step.
         <think reasoning>
         ...
         </think reasoning>
         <answer>25 * 4</answer>"

    【返回值】
    str | None
        成功时返回提取的方程字符串（如 "25 * 4"）
        失败时返回 None（找不到 <answer> 标签或对话格式不正确）

    【示例】
    >>> extract_solution("User: ... Assistant: <answer>25 * 4</answer>")
    '25 * 4'
    >>> extract_solution("没有答案的文本")
    None
    """
    # --------------------------------------------------------------------------
    # 步骤 1: 定位 Assistant 的回复部分
    # --------------------------------------------------------------------------
    # 不同模型使用不同的对话格式，需要分别处理

    # 通用格式："User: ... Assistant: ..."
    if "Assistant:" in solution_str:
        # 分割并取 Assistant 之后的部分
        solution_str = solution_str.split("Assistant:", 1)[1]

    # Qwen Instruct 格式："<|im_start|>user...<|im_start|>assistant..."
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]

    # 无法识别的格式
    else:
        return None

    # --------------------------------------------------------------------------
    # 步骤 2: 提取答案
    # --------------------------------------------------------------------------
    # 模型可能在思考过程中写出多个尝试，我们取最后一个 <answer> 作为最终答案
    # 取最后一行是为了获取模型认为的最终答案

    solution_str = solution_str.split('\n')[-1]

    # 使用正则表达式匹配 <answer>...</answer> 标签
    # 使用非贪婪匹配 .*? 以支持标签内有换行的情况
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)

    if matches:
        # 取最后一个匹配（模型的最终答案）
        # 使用 strip() 去除首尾空白
        final_answer = matches[-1].group(1).strip()
    else:
        # 没有找到 <answer> 标签
        final_answer = None

    return final_answer


# =============================================================================
# 核心函数：验证方程
# =============================================================================

def validate_equation(equation_str, available_numbers):
    """
    验证方程是否只使用了给定的数字，且每个数字只使用一次。

    【功能说明】
    Countdown 任务的规则要求：
    1. 只能使用给定的数字
    2. 每个数字只能使用一次（但不必全部使用）

    此函数检查方程是否满足这些规则。

    【参数】
    equation_str: str
        从 <answer> 标签提取的方程字符串
        例如："25 * 4" 或 "(25 + 3) * 4"

    available_numbers: List[int]
        可用的数字列表
        例如：[25, 4, 3, 1]

    【返回值】
    bool
        True: 方程使用的数字与可用数字完全匹配（允许不全部使用）
        False: 方程使用了不在列表中的数字，或重复使用了数字

    【验证逻辑】
    方程中使用的数字（排序后）必须与可用数字的某个子集完全匹配。
    注意：当前实现要求完全匹配（所有数字都必须使用），这可能需要根据任务要求调整。

    【示例】
    >>> validate_equation("25 * 4", [25, 4, 3, 1])
    False  # 当前实现要求使用所有数字
    >>> validate_equation("25 * 4 * 3 * 1", [25, 4, 3, 1])
    True
    >>> validate_equation("(25 + 3) * 4", [25, 4, 3, 1])
    False  # 缺少数字 1
    """
    try:
        # ----------------------------------------------------------------------
        # 步骤 1: 从方程中提取所有数字
        # ----------------------------------------------------------------------
        # 使用正则表达式找出所有连续的数字序列
        # 例如："25 * 4 + 3" → ["25", "4", "3"] → [25, 4, 3]
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]

        # ----------------------------------------------------------------------
        # 步骤 2: 排序并比较
        # ----------------------------------------------------------------------
        # 将两个列表都排序，然后比较是否相等
        # 排序是为了忽略数字的使用顺序
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # 检查方程中使用的数字是否与可用数字完全一致
        # 注意：这要求必须使用所有给定的数字
        # 如果允许不使用所有数字，应该改为：set(numbers_in_eq).issubset(set(available_numbers))
        return numbers_in_eq == available_numbers

    except Exception:
        # 任何解析错误都视为验证失败
        return False


# =============================================================================
# 核心函数：计算方程结果
# =============================================================================

def evaluate_equation(equation_str):
    """
    安全地计算数学方程的结果。

    【功能说明】
    使用 Python 的 eval() 函数计算方程结果。
    由于 eval() 可以执行任意代码，必须进行严格的安全检查。

    【安全措施】
    1. 使用正则表达式限制允许的字符（只允许数字、运算符、括号）
    2. 禁用 eval() 的内置函数和命名空间

    【参数】
    equation_str: str
        要计算的方程字符串
        例如："25 * 4" 或 "(25 + 3) * 4"

    【返回值】
    float | int | None
        成功时返回计算结果
        失败时返回 None（方程包含非法字符或计算错误）

    【支持的运算】
    - 加法: +
    - 减法: -
    - 乘法: *
    - 除法: /
    - 括号: ( )
    - 整数和小数

    【示例】
    >>> evaluate_equation("25 * 4")
    100
    >>> evaluate_equation("(25 + 3) * 4")
    112
    >>> evaluate_equation("25 * 4 + print('hack')")  # 包含非法字符
    None
    """
    try:
        # ----------------------------------------------------------------------
        # 步骤 1: 安全检查 - 验证字符白名单
        # ----------------------------------------------------------------------
        # 定义允许的字符模式：
        # \d     - 数字 0-9
        # +\-*/  - 四则运算符（注意 - 在字符类中需要转义或放在末尾）
        # ()     - 括号
        # .      - 小数点
        # \s     - 空白字符（空格、制表符等）
        allowed_pattern = r'^[\d+\-*/().\s]+$'

        if not re.match(allowed_pattern, equation_str):
            # 方程包含非法字符，拒绝执行
            raise ValueError("Invalid characters in equation.")

        # ----------------------------------------------------------------------
        # 步骤 2: 安全执行 - 使用受限的命名空间
        # ----------------------------------------------------------------------
        # eval(expression, globals, locals)
        # - globals 设为 {"__builtins__": None} 禁用所有内置函数
        # - locals 设为空字典 {} 禁用所有局部变量
        # 这样即使有人尝试注入代码，也无法访问任何危险函数
        result = eval(equation_str, {"__builtins__": None}, {})

        return result

    except Exception as e:
        # 任何计算错误都返回 None
        return None


# =============================================================================
# 主函数：计算奖励分数
# =============================================================================

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """
    计算 Countdown 任务的主奖励函数。

    【功能说明】
    此函数是奖励计算的主入口。它整合了提取、验证和计算三个步骤，
    根据模型输出的质量返回相应的奖励分数。

    【奖励分级说明】
    ┌─────────────────────────────────────────────────────────────────┐
    │  奖励值  │  含义              │  触发条件                       │
    ├─────────────────────────────────────────────────────────────────┤
    │  1.0     │  完全正确          │  方程正确且结果等于目标值        │
    │  0.1     │  格式正确但错误    │  有答案但数字错误或计算不匹配    │
    │  0       │  格式错误          │  无法提取有效答案                │
    └─────────────────────────────────────────────────────────────────┘

    【参数】
    solution_str: str
        完整的对话文本（包含 User 问题 + Assistant 回答）
        例如："User: Using the numbers [25, 4, 3, 1]...
               Assistant: <answer>25 * 4</answer>"

    ground_truth: dict
        正确答案信息，包含：
        - 'target': int, 目标值
        - 'numbers': List[int], 可用数字列表
        例如：{'target': 100, 'numbers': [25, 4, 3, 1]}

    method: str, 默认 'strict'
        提取方法（当前仅支持 'strict'）

    format_score: float, 默认 0.1
        格式正确但答案错误时的奖励值

    score: float, 默认 1.0
        完全正确时的奖励值

    【返回值】
    float
        奖励分数（0, 0.1, 或 1.0）

    【处理流程图】
    solution_str
         │
         ▼
    ┌─────────────────┐
    │ extract_solution│ ← 提取 <answer> 中的方程
    └────────┬────────┘
             │
             ▼
        equation = None?
         │          │
        Yes        No
         │          │
         ▼          ▼
       返回 0    ┌──────────────────┐
                 │ validate_equation│ ← 检查数字使用是否正确
                 └────────┬─────────┘
                          │
                          ▼
                    验证通过?
                     │      │
                    No     Yes
                     │      │
                     ▼      ▼
                 返回 0.1  ┌───────────────────┐
                          │ evaluate_equation │ ← 计算方程结果
                          └─────────┬─────────┘
                                    │
                                    ▼
                              结果 == 目标?
                               │      │
                              No     Yes
                               │      │
                               ▼      ▼
                           返回 0.1  返回 1.0

    【调用关系】
    此函数由 RewardManager.__call__() 调用
    """
    # --------------------------------------------------------------------------
    # 步骤 1: 提取参数
    # --------------------------------------------------------------------------
    target = ground_truth['target']      # 目标值
    numbers = ground_truth['numbers']    # 可用数字列表

    # --------------------------------------------------------------------------
    # 步骤 2: 从解决方案字符串中提取方程
    # --------------------------------------------------------------------------
    equation = extract_solution(solution_str=solution_str)

    # 随机抽样打印调试信息（1/64 的概率）
    # 这样可以在不产生过多输出的情况下监控训练过程
    do_print = random.randint(1, 64) == 1

    # --------------------------------------------------------------------------
    # 步骤 3: 调试输出
    # --------------------------------------------------------------------------
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    # --------------------------------------------------------------------------
    # 步骤 4: 检查是否成功提取方程
    # --------------------------------------------------------------------------
    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0  # 格式错误：无法提取答案

    # --------------------------------------------------------------------------
    # 步骤 5: 验证方程是否使用了正确的数字
    # --------------------------------------------------------------------------
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score  # 格式正确但数字使用错误

    # --------------------------------------------------------------------------
    # 步骤 6: 计算方程结果并比较
    # --------------------------------------------------------------------------
    try:
        result = evaluate_equation(equation)

        if result is None:
            # 方程计算失败（可能包含无效操作）
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        # 比较计算结果与目标值
        # 使用 1e-5 的容差来处理浮点数精度问题
        if abs(result - target) < 1e-5:
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score  # 完全正确！
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score  # 格式正确但结果错误

    except Exception:
        if do_print:
            print(f"Error evaluating equation")
        return format_score
