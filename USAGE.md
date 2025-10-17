# DPO 使用指南

## 快速开始

### 1. 本地测试(小数据集 + 小模型)

最快速的测试方式,使用 sanity check 模式只训练 100 个样本:

```bash
# 运行预配置的测试脚本
bash examples/local_test.sh

# 或者手动运行
uv run python main.py \
  --model_name "gpt2" \
  --batch_size 2 \
  --max_steps 50 \
  --sanity_check \
  --output_dir "./outputs/test"
```

### 2. 本地训练(完整数据集 + 小模型)

在 macOS 或单 GPU 机器上训练:

```bash
bash examples/train_small_model.sh
```

推荐的小模型(可在普通笔记本上运行):
- `gpt2` (124M 参数) - 最小,最快
- `facebook/opt-125m` (125M 参数)
- `EleutherAI/pythia-160m` (160M 参数)
- `EleutherAI/pythia-410m` (410M 参数)

### 3. SLURM 超算训练(大模型)

提交到 SLURM 集群进行大规模训练:

```bash
# 使用默认配置(gpt2)
sbatch slurm_job.sh

# 使用大模型
export MODEL_NAME="meta-llama/Llama-2-7b-hf"
export BATCH_SIZE=8
export MAX_STEPS=10000
sbatch slurm_job.sh

# 或使用示例脚本
bash examples/slurm_large_model.sh
```

## 命令行参数详解

### 模型参数

```bash
--model_name TEXT          # 预训练模型名称或路径
                           # 例如: "gpt2", "facebook/opt-125m", "meta-llama/Llama-2-7b-hf"

--load_in_8bit            # 使用 8-bit 量化加载模型(需要 bitsandbytes)
                           # 可以大幅减少显存占用
```

### DPO 参数

```bash
--beta FLOAT              # DPO 的 β 参数,控制 KL 散度约束强度
                           # 默认: 0.1
                           # 较大的值会使模型更接近参考模型
                           # 论文中使用 0.1-0.5
```

### 训练参数

```bash
--learning_rate FLOAT     # 学习率,默认: 1e-6
                           # 论文使用 1e-6 with RMSprop

--batch_size INT          # 训练批量大小,默认: 4
                           # 根据 GPU 显存调整

--max_steps INT           # 最大训练步数,默认: 1000
                           # 论文中对不同任务使用不同步数:
                           # - 情感生成: 20000 步
                           # - 摘要生成: 3000 步
                           # - 对话: 3000 步

--warmup_steps INT        # 学习率预热步数,默认: 150

--eval_steps INT          # 每隔多少步评估一次,默认: 100

--save_steps INT          # 每隔多少步保存检查点,默认: 500
```

### 数据参数

```bash
--max_length INT          # 最大序列长度,默认: 512
                           # 更长的序列需要更多显存

--max_prompt_length INT   # 最大提示词长度,默认: 256

--sanity_check            # 只使用 100 个样本进行快速测试
```

### 系统参数

```bash
--output_dir PATH         # 输出目录,默认: "./outputs"

--num_workers INT         # 数据加载线程数,默认: 4
                           # macOS 上可能需要设为 0

--use_wandb               # 启用 Weights & Biases 日志记录

--seed INT                # 随机种子,默认: 42
```

## 完整示例

### 示例 1: 使用 GPT-2 在本地训练

```bash
uv run python main.py \
  --model_name "gpt2" \
  --beta 0.1 \
  --learning_rate 1e-6 \
  --batch_size 4 \
  --max_steps 1000 \
  --eval_steps 100 \
  --save_steps 500 \
  --output_dir "./outputs/gpt2_dpo" \
  --num_workers 4
```

### 示例 2: 使用 OPT-125M 并启用 WandB

```bash
uv run python main.py \
  --model_name "facebook/opt-125m" \
  --beta 0.5 \
  --batch_size 8 \
  --max_steps 3000 \
  --use_wandb \
  --output_dir "./outputs/opt125m_dpo"
```

### 示例 3: 使用 Pythia-410M 在 SLURM 上训练

```bash
export MODEL_NAME="EleutherAI/pythia-410m"
export BETA=0.1
export BATCH_SIZE=16
export MAX_STEPS=5000
export OUTPUT_DIR="./outputs/pythia410m_dpo"

sbatch slurm_job.sh
```

## 监控训练

### 1. 查看输出日志

```bash
# 实时查看训练日志
tail -f outputs/gpt2_dpo/training.log

# 或者查看 SLURM 日志
tail -f logs/dpo_<job_id>.out
```

### 2. 使用 WandB

如果启用了 `--use_wandb`:

1. 首次使用需要登录:
   ```bash
   wandb login
   ```

2. 在浏览器中查看: https://wandb.ai/

3. 监控的指标包括:
   - `train/loss`: 训练损失
   - `train/accuracy`: 模型选择偏好答案的准确率
   - `train/reward_margin`: 偏好答案和非偏好答案的奖励差
   - `train/chosen_rewards`: 偏好答案的平均隐式奖励
   - `train/rejected_rewards`: 非偏好答案的平均隐式奖励
   - `eval/*`: 评估集上的相同指标

## 训练完成后

### 1. 加载训练好的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./outputs/gpt2_dpo/final_model")
tokenizer = AutoTokenizer.from_pretrained("./outputs/gpt2_dpo/final_model")

# 生成文本
prompt = "Human: How do I make pizza?\n\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

### 2. 继续训练

```bash
uv run python main.py \
  --model_name "./outputs/gpt2_dpo/checkpoint-1000" \
  --max_steps 2000 \
  --output_dir "./outputs/gpt2_dpo_continued"
```

## 常见问题

### Q: 显存不足怎么办?

A: 尝试以下方法:
1. 减小 `--batch_size`
2. 减小 `--max_length`
3. 使用 `--load_in_8bit` 加载模型
4. 使用更小的模型

### Q: macOS 上训练很慢?

A:
1. 设置 `--num_workers 0` (macOS 的多进程有时会很慢)
2. 使用 MPS 加速(PyTorch 会自动检测并使用 M1/M2 芯片)

### Q: 如何选择 beta 值?

A:
- 论文建议范围: 0.1 - 0.5
- 较小的 beta (如 0.1): 允许模型更多地偏离参考模型
- 较大的 beta (如 0.5): 保持更接近参考模型
- 建议从 0.1 开始尝试

### Q: 训练需要多长时间?

A: 取决于:
- 模型大小: GPT-2 (124M) 在单 GPU 上约 1-2 小时完成 1000 步
- 批量大小: 更大的批量更快,但需要更多显存
- 序列长度: 更短的序列训练更快

## 性能优化提示

1. **梯度累积**: 如果显存不足但想要更大的有效批量大小,可以修改代码添加梯度累积

2. **混合精度训练**: 代码已自动使用 `torch.cuda.amp.autocast()` 进行混合精度训练

3. **数据加载优化**:
   - 使用 `--num_workers > 0` 并行加载数据
   - 在 SLURM 上设置 `--num_workers` 等于 CPU 核心数

4. **检查点**: 定期保存检查点以防训练中断:
   ```bash
   --save_steps 500  # 每 500 步保存一次
   ```
