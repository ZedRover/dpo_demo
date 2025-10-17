# DPO (Direct Preference Optimization) Implementation

完整实现论文 "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" ([arxiv:2305.18290](https://arxiv.org/abs/2305.18290))

## 功能特性

- ✅ **核心 DPO 算法**: 完整实现论文中的 DPO loss (方程 7)
- ✅ **数据加载**: 支持 Anthropic/hh-rlhf 数据集
- ✅ **训练框架**: 包含完整的训练循环和评估
- ✅ **论文指标**: Reward-KL frontier, win rate 等评估指标
- ✅ **可视化**: 复现论文中的 Figure 2 和 Figure 3
- ✅ **方法对比**: DPO vs Preferred-FT vs Best-of-N vs 基础模型
- ✅ **SLURM 支持**: 可在超算集群上运行

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv 管理环境
uv sync
```

### 2. 本地测试训练

```bash
# 使用小模型快速测试 (50步)
bash examples/local_test.sh
```

这将:
- 使用 GPT-2 模型
- 在 100 个样本上训练 50 步
- 保存模型到 `outputs/local_test/`
- 生成 reward-KL 追踪数据

### 3. 生成可视化

```bash
# 创建论文风格的图表
uv run python visualize.py \
    --tracker_path outputs/local_test/reward_kl_tracker.json \
    --output_dir plots
```

生成的图表:
- `reward_kl_frontier.png` - Reward-KL 边界 (Figure 2 左)
- `training_curves.png` - 训练曲线

## 方法对比 (Figure 3)

复现论文 Figure 3,对比多种方法:

```bash
# 一键运行所有对比实验
bash examples/run_comparison.sh
```

这将训练和评估:
1. **DPO** - 我们的主要方法
2. **Preferred-FT** - 仅在 preferred completions 上微调
3. **Best of 128** - 采样128个响应选最佳
4. **Pythia-2.8B** - 基础模型

详细说明见 [COMPARISON_GUIDE.md](COMPARISON_GUIDE.md)

## 完整训练

### 在本地训练

```bash
uv run python main.py \
    --model_name EleutherAI/pythia-410m \
    --beta 0.1 \
    --learning_rate 1e-6 \
    --max_steps 5000 \
    --batch_size 4 \
    --output_dir ./outputs/dpo_training \
    --sanity_check  # 使用小数据集
```

### 在 SLURM 集群训练

```bash
# 编辑 slurm_job.sh 配置
# 然后提交任务
sbatch slurm_job.sh
```

## 评估

### 计算 Win Rate

```bash
uv run python evaluate.py \
    --model_path outputs/local_test/final_model \
    --num_samples 100 \
    --beta 0.1 \
    --output_dir ./eval_results
```

### 可视化结果

```bash
# 单个方法
uv run python visualize.py \
    --tracker_path outputs/dpo_training/reward_kl_tracker.json \
    --output_dir plots

# 对比多个方法
uv run python visualize.py \
    --tracker_path outputs/dpo_training/reward_kl_tracker.json \
    --compare PPO:outputs/ppo_training/reward_kl_tracker.json \
    --output_dir plots
```

## 项目结构

```
dpo_demo/
├── dpo/                          # DPO 核心实现
│   ├── __init__.py
│   ├── loss.py                   # DPO loss 函数
│   ├── data.py                   # 数据加载和预处理
│   ├── trainer.py                # 训练器
│   ├── metrics.py                # 评估指标
│   └── plotting.py               # 可视化工具
├── main.py                       # 训练入口
├── evaluate.py                   # 评估脚本
├── visualize.py                  # 可视化脚本
├── train_baselines.py            # 训练baseline方法
├── evaluate_comparison.py        # 多方法对比评估
├── config.py                     # 配置文件
├── slurm_job.sh                  # SLURM 任务脚本
├── examples/
│   ├── local_test.sh             # 本地测试脚本
│   └── run_comparison.sh         # 对比实验脚本
├── COMPARISON_GUIDE.md           # 对比实验指南
└── README.md                     # 本文件
```

## 核心算法

### DPO Loss

DPO 直接优化偏好,无需显式的reward model:

```python
L_DPO(π_θ; π_ref) = -E[(x,y_w,y_l)~D][log σ(β log(π_θ(y_w|x)/π_ref(y_w|x)) - β log(π_θ(y_l|x)/π_ref(y_l|x)))]
```

其中:
- `y_w`: preferred completion
- `y_l`: rejected completion
- `β`: KL惩罚强度
- `π_ref`: reference model (冻结)

### Implicit Reward

DPO 隐式学习一个reward函数:

```python
r(x, y) = β log(π(y|x) / π_ref(y|x))
```

## 实验结果

论文中的关键发现:

1. **Figure 2 (左)**: DPO 在 reward-KL frontier 上优于 PPO
2. **Figure 2 (右)**: DPO 在摘要任务上超过 PPO
3. **Figure 3**: DPO 在对话任务上是唯一改进 chosen responses 的高效方法

## 配置说明

### 关键超参数

- `--beta`: DPO 温度参数,控制 KL 惩罚强度 (默认: 0.1)
- `--learning_rate`: 学习率 (默认: 1e-6)
- `--max_steps`: 最大训练步数
- `--batch_size`: 批次大小
- `--warmup_steps`: warmup 步数 (默认: 150)

### 模型选择

支持任何 HuggingFace 模型:
- 小模型 (测试): `gpt2`, `EleutherAI/pythia-410m`
- 中模型: `EleutherAI/pythia-1.4b`, `EleutherAI/pythia-2.8b`
- 大模型: `EleutherAI/pythia-6.9b`, `facebook/opt-6.7b`

## 论文复现

### Figure 2 (左): Reward-KL Frontier

```bash
# 训练多个 beta 值
for beta in 0.05 0.1 0.5 1.0; do
    uv run python main.py --beta $beta --output_dir outputs/beta_$beta
done

# 绘制对比图
uv run python visualize.py --compare beta_0.05:outputs/beta_0.05/...
```

### Figure 2 (右): Summarization Win Rate

```bash
uv run python evaluate.py \
    --model_path outputs/dpo_model \
    --output_dir eval_results
```

### Figure 3: Method Comparison

```bash
bash examples/run_comparison.sh
```

## 性能优化

### 使用 GPU

```bash
uv run python main.py --device cuda
```

### 减少内存占用

```bash
uv run python main.py \
    --batch_size 2 \
    --max_length 256 \
    --sanity_check
```

### 启用混合精度

编辑 `dpo/trainer.py`,使用 `torch.amp`

## 常见问题

### Q: 训练很慢怎么办?
A:
1. 减少 `--max_steps`
2. 使用 `--sanity_check` (仅100个样本)
3. 使用更小的模型
4. 启用 GPU

### Q: Out of memory?
A:
1. 减少 `--batch_size`
2. 减少 `--max_length`
3. 使用更小的模型

### Q: 如何使用自己的数据集?
A:
修改 `dpo/data.py` 中的 `load_hh_rlhf_dataset` 函数

### Q: 如何使用 GPT-4 评估?
A:
参考 `evaluate.py`,需要 OpenAI API key

## 引用

如果使用本实现,请引用原论文:

```bibtex
@inproceedings{rafailov2023direct,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```

## 贡献

欢迎提交 Issue 和 Pull Request!

## License

MIT License

## 相关资源

- 📄 [原论文](https://arxiv.org/abs/2305.18290)
- 🤗 [Anthropic HH-RLHF 数据集](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- 📊 [论文结果复现指南](COMPARISON_GUIDE.md)
