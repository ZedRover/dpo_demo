# Qwen3-8B 训练前检查清单

在提交 SLURM 任务前,请确认以下事项:

## ✅ 环境准备

- [ ] Python 环境已配置 (`uv sync` 完成)
- [ ] CUDA 可用 (`nvidia-smi` 正常)
- [ ] 存储空间充足 (>100GB 可用)
- [ ] logs 目录存在 (`mkdir -p logs`)

## ✅ GPU 资源

- [ ] 确认 GPU 型号和显存
  ```bash
  nvidia-smi --query-gpu=name,memory.total --format=csv
  ```
- [ ] 根据显存调整批次大小:
  - 24GB → batch_size=1-2, load_in_8bit=true
  - 40GB → batch_size=4-8
  - 80GB → batch_size=8-16

## ✅ 模型访问

- [ ] 确认可以访问 Hugging Face
  ```bash
  python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen3-8B')"
  ```
- [ ] 如在中国,设置镜像:
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

## ✅ 数据集

- [ ] 确认可以下载 Anthropic/hh-rlhf
  ```bash
  python -c "from datasets import load_dataset; load_dataset('Anthropic/hh-rlhf', split='train[:10]')"
  ```

## ✅ SLURM 配置

检查脚本中的配置是否符合你的集群:

- [ ] `#SBATCH --partition=gpu` - 分区名称正确
- [ ] `#SBATCH --gres=gpu:1` - GPU 请求格式正确
- [ ] `#SBATCH --time=48:00:00` - 时间限制足够且不超过分区限制
- [ ] `#SBATCH --mem=64G` - 内存请求不超过节点限制
- [ ] 模块加载命令正确 (如需要)

查看集群配置:
```bash
sinfo -o "%20P %5a %10l %16F"
```

## ✅ 快速测试 (强烈推荐!)

在完整训练前运行测试:

```bash
sbatch slurm_qwen_8b_test.sh
```

等待测试完成 (~30分钟),确认:
- [ ] 模型成功加载
- [ ] 训练正常运行
- [ ] GPU 内存充足
- [ ] 日志输出正常

## ✅ 完整训练准备

测试通过后:

- [ ] 检查超参数设置
  - BETA=0.1 (DPO 温度)
  - LEARNING_RATE=5e-7 (学习率)
  - MAX_STEPS=10000 (训练步数)

- [ ] 设置输出目录
  - 确保有足够空间存储检查点

- [ ] (可选) 配置 W&B
  ```bash
  export WANDB_API_KEY="your_key"
  ```

## ✅ 提交任务

一切准备就绪后:

```bash
sbatch slurm_qwen_8b.sh
```

记录任务 ID:
```bash
JOB_ID=<your_job_id>
```

## ✅ 监控训练

- [ ] 查看任务状态
  ```bash
  squeue -u $USER
  ```

- [ ] 监控日志
  ```bash
  tail -f logs/dpo_qwen_${JOB_ID}.out
  ```

- [ ] 检查 GPU 利用率 (如果可以 SSH 到计算节点)
  ```bash
  watch -n 5 nvidia-smi
  ```

## 🚨 常见问题快速修复

### OOM (显存不足)

```bash
# 编辑 slurm_qwen_8b.sh
BATCH_SIZE=1
LOAD_IN_8BIT=true
MAX_LENGTH=256
```

### 任务一直在排队

```bash
# 检查队列情况
squeue -p gpu

# 查看你的任务优先级
sprio -j <JOB_ID>
```

### 模型下载失败

```bash
# 预先下载模型
srun --gres=gpu:1 --pty bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-8B')
"
```

## 📊 训练完成后

- [ ] 检查最终模型
  ```bash
  ls -lh outputs/qwen2.5_8b_*/final_model/
  ```

- [ ] 生成可视化
  ```bash
  uv run python visualize.py --tracker_path outputs/qwen2.5_8b_*/reward_kl_tracker.json
  ```

- [ ] 评估模型
  ```bash
  uv run python evaluate.py --model_path outputs/qwen2.5_8b_*/final_model
  ```

## 🎯 预期训练时间

| GPU | 批次大小 | 预计时间 (10K steps) |
|-----|---------|---------------------|
| A100-80GB | 8 | ~12-16小时 |
| A100-40GB | 4 | ~16-20小时 |
| V100-32GB | 2 | ~30-36小时 |
| RTX 3090 | 1 (8bit) | ~40-48小时 |

## 📝 记录信息

训练任务信息:
```
任务 ID: _________________
开始时间: _________________
预计完成: _________________
GPU 型号: _________________
批次大小: _________________
学习率: _________________
其他备注: _________________
```

---

**准备好了吗?** 如果所有复选框都已勾选,运行:

```bash
sbatch slurm_qwen_8b.sh
```

祝训练顺利! 🚀
