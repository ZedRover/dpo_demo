# Qwen2.5-8B DPO Training Guide

使用 Qwen2.5-8B 在 SLURM 集群上训练 DPO 模型的完整指南。

## 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **GPU** | 24GB VRAM (V100, RTX 3090) | 40GB+ (A100-40GB/80GB) |
| **系统内存** | 64GB RAM | 128GB+ RAM |
| **存储** | 100GB 可用空间 | 500GB+ SSD |
| **CPU** | 8 核心 | 16+ 核心 |

### 软件要求

- CUDA 11.8+ 或 12.1+
- Python 3.10+
- PyTorch 2.0+
- transformers, datasets, tqdm

## 快速开始

### 第一步: 准备环境

```bash
# 在集群上克隆项目
cd /path/to/your/workspace
# (假设你已经上传了代码)

# 确保 uv 环境已配置
uv sync
```

### 第二步: 快速测试 (推荐!)

在运行完整训练前,先测试配置:

```bash
# 提交测试任务 (50 steps, ~30分钟)
sbatch slurm_qwen_8b_test.sh
```

检查输出:
```bash
# 查看日志
tail -f logs/dpo_qwen_test_<JOB_ID>.out

# 如果成功,你会看到:
# ✓ Test passed! Ready for full training.
```

### 第三步: 完整训练

测试通过后,运行完整训练:

```bash
# 提交完整训练任务
sbatch slurm_qwen_8b.sh
```

## 配置说明

### Qwen2.5-8B 模型信息

- **参数量**: 8B (8 billion)
- **架构**: Transformer decoder
- **上下文长度**: 最大 32K (我们使用 512)
- **词表大小**: ~152K tokens
- **模型大小**: ~16GB (FP16), ~8GB (INT8)

### 训练超参数

在 `slurm_qwen_8b.sh` 中配置:

```bash
MODEL_NAME="Qwen/Qwen2.5-8B"
BETA=0.1                    # DPO 温度参数
LEARNING_RATE=5e-7          # 学习率 (8B模型用较小的LR)
BATCH_SIZE=4                # 批次大小 (根据GPU调整)
MAX_STEPS=10000             # 训练步数
MAX_LENGTH=512              # 序列长度
```

### GPU 内存优化

如果遇到 OOM (Out of Memory) 错误:

#### 选项 1: 减小批次大小

```bash
# 在脚本中修改
BATCH_SIZE=2  # 或者 1
```

#### 选项 2: 使用 8-bit 量化

```bash
# 在脚本中设置
LOAD_IN_8BIT=true
```

这会将内存需求从 ~16GB 降到 ~8GB,但可能略微影响性能。

#### 选项 3: 启用梯度检查点

在 `main.py` 中添加:
```python
model.gradient_checkpointing_enable()
```

#### 选项 4: 减小序列长度

```bash
MAX_LENGTH=256  # 从 512 降到 256
```

### 推荐的 GPU 配置

| GPU 型号 | VRAM | 推荐批次大小 | 8-bit量化 |
|---------|------|------------|----------|
| RTX 3090 | 24GB | 1-2 | 推荐 |
| V100 | 32GB | 2-4 | 可选 |
| A100-40GB | 40GB | 4-8 | 不需要 |
| A100-80GB | 80GB | 8-16 | 不需要 |

## 监控训练

### 查看实时日志

```bash
# 查看标准输出
tail -f logs/dpo_qwen_<JOB_ID>.out

# 查看错误输出
tail -f logs/dpo_qwen_<JOB_ID>.err
```

### 关键指标

训练过程中关注:
- **Loss**: 应该逐渐下降
- **Accuracy**: chosen vs rejected 的准确率
- **Reward margin**: chosen 和 rejected 的奖励差
- **KL divergence**: 与参考模型的 KL 散度

### 使用 W&B (可选)

启用 Weights & Biases 监控:

```bash
# 1. 设置 API key
export WANDB_API_KEY="your_key_here"

# 2. 在脚本中取消注释
# CMD="$CMD --use_wandb"
```

## 训练时间估计

基于 Anthropic HH-RLHF 数据集 (~170K 样本):

| 配置 | GPU | 批次大小 | 预计时间 |
|------|-----|---------|---------|
| 快速测试 | A100 | 4 | ~30分钟 (50步) |
| 中等训练 | A100 | 4 | ~8小时 (5K步) |
| 完整训练 | A100 | 4 | ~16-20小时 (10K步) |
| 完整训练 | V100 | 2 | ~30-36小时 (10K步) |

## 故障排查

### 问题 1: CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决**:
```bash
# 方法 1: 减小批次
BATCH_SIZE=1

# 方法 2: 启用 8-bit
LOAD_IN_8BIT=true

# 方法 3: 减小序列长度
MAX_LENGTH=256
```

### 问题 2: 模型下载失败

**症状**: `Connection timeout` 或 `403 Forbidden`

**解决**:
```bash
# 预先下载模型
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-8B')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-8B')
"

# 或使用 HF 镜像 (中国用户)
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题 3: 训练不稳定 / Loss 爆炸

**症状**: Loss 突然增大或变成 NaN

**解决**:
```bash
# 降低学习率
LEARNING_RATE=1e-7  # 从 5e-7 降到 1e-7

# 增加 warmup
WARMUP_STEPS=500  # 从 150 增加到 500

# 检查梯度裁剪 (在 trainer.py 中)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 问题 4: 任务被取消

**症状**: Job killed by SLURM

**解决**:
```bash
# 增加时间限制
#SBATCH --time=72:00:00  # 72 hours

# 增加内存
#SBATCH --mem=128G

# 检查是否超出分区限制
sinfo  # 查看可用资源
```

## 高级配置

### 多 GPU 训练 (DDP)

如果有多个 GPU:

```bash
#SBATCH --gres=gpu:2  # 使用 2 个 GPU

# 使用 torchrun
CMD="torchrun --nproc_per_node=2 main.py ..."
```

### 断点续训

如果训练中断,从检查点恢复:

```bash
# main.py 支持自动从最新检查点恢复
# 只需使用相同的 output_dir
```

### 自定义数据集

如果想用自己的数据:

```python
# 修改 dpo/data.py
def load_custom_dataset(split="train"):
    # 你的数据加载逻辑
    # 返回格式: {"chosen": str, "rejected": str}
    pass
```

## 评估和可视化

训练完成后:

### 1. 生成可视化

```bash
# 在登录节点或本地运行
uv run python visualize.py \
    --tracker_path outputs/qwen2.5_8b_*/reward_kl_tracker.json \
    --output_dir plots/qwen
```

### 2. 计算 Win Rate

```bash
uv run python evaluate.py \
    --model_path outputs/qwen2.5_8b_*/final_model \
    --num_samples 500 \
    --output_dir eval_results/qwen
```

### 3. 对比其他方法

```bash
# 训练 Preferred-FT baseline
sbatch slurm_qwen_baseline.sh  # (需要创建)

# 运行对比
uv run python evaluate_comparison.py \
    --dpo_path outputs/qwen2.5_8b_*/final_model \
    --base_model Qwen/Qwen2.5-8B
```

## 最佳实践

1. **先运行快速测试**: 避免在完整训练时才发现配置问题
2. **监控 GPU 利用率**: 使用 `nvidia-smi` 确保 GPU 被充分利用
3. **保存检查点**: 定期保存,防止训练中断
4. **记录超参数**: 使用 W&B 或记录在日志中
5. **验证数据**: 确保数据加载正确,检查第一个 batch

## 示例工作流

```bash
# 1. 提交测试任务
sbatch slurm_qwen_8b_test.sh
# 等待完成 (~30分钟)

# 2. 检查测试结果
cat logs/dpo_qwen_test_*.out

# 3. 如果测试通过,提交完整训练
sbatch slurm_qwen_8b.sh

# 4. 监控训练 (在另一个终端)
watch -n 60 'tail -20 logs/dpo_qwen_*.out'

# 5. 训练完成后,生成可视化
uv run python visualize.py --tracker_path outputs/qwen2.5_8b_*/reward_kl_tracker.json

# 6. 评估模型
uv run python evaluate.py --model_path outputs/qwen2.5_8b_*/final_model
```

## 预期结果

基于论文,成功的 DPO 训练应该显示:

- ✅ **Loss 下降**: 从 ~0.7 降到 ~0.3-0.4
- ✅ **Accuracy 提高**: 从 50% 升到 60-70%
- ✅ **Reward margin 增加**: Chosen rewards > Rejected rewards
- ✅ **KL divergence 稳定**: 保持在合理范围内 (0-20)

## 联系和支持

遇到问题?
1. 检查日志文件
2. 参考本指南的故障排查部分
3. 查看 GitHub Issues
4. 阅读 DPO 原论文

## 引用

```bibtex
@article{qwen2.5,
  title={Qwen2.5: A Party of Foundation Models},
  author={Qwen Team},
  year={2024}
}
```
