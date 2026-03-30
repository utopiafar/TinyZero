
## HuggingFace 镜像配置

服务器无法直接访问 HuggingFace，使用镜像站：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或者在 Python 中设置：
```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```


服务器使用autodl环境，相关文档见：https://www.autodl.com/docs

## 环境相关
- 如果需要安装 FlashAttention 之类的框架，优先使用编译好的版本。（https://github.com/mjun0812/flash-attention-prebuild-wheels）

## 项目操作规范



### 1. 实验文档规范
- 每次实验需留下完整的**实验文档**和**实验效果**
- 实验相关内容放在**单独的目录**中

### 2. 目录结构
```
TinyZero/
├── docs/               # 项目文档
├── experiments/        # 实验记录目录
│   ├── YYYY-MM-DD_exp_name/   # 单次实验目录（命名规范：日期_实验名）
│   │   ├── README.md          # 实验说明文档
│   │   ├── config.yaml        # 实验配置
│   │   ├── results/           # 实验结果
│   │   └── logs/              # 实验日志
├── scripts/            # 脚本文件
├── data/               # 数据文件
└── CLAUDE.md          # 项目规范文档
```

### 3. 实验目录命名规范
```
experiments/YYYY-MM-DD_{实验名称}/
```
- 日期格式：`YYYY-MM-DD`
- 实验名称：简短描述实验内容，使用下划线连接
- 示例：`2025-03-27_ppo_training_v1`

## 数据存储规范

### 存储路径说明
- **系统盘 `/`**：用于存储所有数据和模型（30GB，当前使用量低）

### 需要持久化存储的内容
1. **日志文件/tensorboard** → `/root/tf-logs/`
2. **模型文件** → `~/models/`
3. **实验数据** → `~/data/`
4. **临时文件** → `~/tmp/`
5. **实验记录** → `~/experiments/`
6. **持久化（模型文件、checkpoint等）** -> `/root/autodl-fs`

### 推荐的目录结构
```
~/
├── tf-logs/            # 训练日志文件（TensorBoard 等）
├── models/             # 训练好的模型和下载的基座模型
├── data/               # 数据集
├── experiments/        # 实验记录和结果
└── tmp/                # 临时文件
```
