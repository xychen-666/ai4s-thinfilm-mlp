# 基于 MLP 的多层介质薄膜光谱预测与辅助设计

《薄膜技术》AI4S 课程大作业代码。作者：陈新堉，学号：2023270069。个人目标波长 **520 nm**；`seed = 270069`，`design_seed = 270070`。

本仓库从传输矩阵法（TMM）生成数据开始，完成四种训练规模的 MLP 光谱预测实验，并在新候选中按 **520 nm 处反射率越高越好** 的目标进行筛选。MLP 只负责候选排序；最终 Top 5 按 TMM 重新计算的反射率排序。作业原文未明说高反射或低反射，本实验明确选用高反射目标。

## 研究对象与理论

- 结构：Air / H / L / H / L / Glass，四层待设计介质膜。
- 正入射、无吸收、忽略色散；`n_air = 1.00`、`nH = 2.30`、`nL = 1.45`、`n_glass = 1.52`。玻璃为半无限基底。
- 每层膜厚连续均匀抽样于 `[40, 180)` nm。
- 反射光谱采样点：400、410、…、800 nm，共 41 点。

每层的相位厚度为 `δ_j = 2π n_j d_j / λ`，正入射特征矩阵为

```text
M_j = [[cos(δ_j),       i sin(δ_j) / n_j],
       [i n_j sin(δ_j), cos(δ_j)        ]]
M = M_H(d1) M_L(d2) M_H(d3) M_L(d4)
B = M_11 + n_glass M_12;  C = M_21 + n_glass M_22
r = (n_air B - C) / (n_air B + C);  R = |r|²
```

代码也计算透射率 `T`，物理校验要求无吸收时 `R + T = 1`。另用独立的菲涅耳界面递推实现作为测试参照。这里的模型是课程给定的简化物理模型，未把 H/L 折射率解释为真实材料全波段的测量值。

## 实验流程

1. `np.random.default_rng(270069)` 生成 5000 组膜厚和 TMM 光谱，再随机排列一次，依次划分为训练 4000、验证 500、测试 500。划分在所有实验中保持不变。
2. 使用同一训练池的前 500、1000、2000、4000 组分别训练 MLP。每组从同一种子初始化，训练轮数、批大小、学习率、优化器和验证集一致。
3. 网络为 `4–128–128–64–41`，隐藏层 ReLU，输出层线性；膜厚按 `[40,180] nm → [-1,1]` 缩放。损失为 MSE，优化器为 Adam。每轮记录训练和验证 MSE，选取最低验证 MSE 的模型，最后仅在固定测试集上评估。
4. `np.random.default_rng(270070)` 独立生成 10000 组新候选；用 4000 组训练的模型预测，在 520 nm 按预测反射率选 Top 10，用 TMM 验证后按真实反射率重排，得到 Top 5。
5. 为审计筛选效果，**额外**对全部 10000 组只在 520 nm 做一次 TMM 计算，求得候选池真实最优值和所选设计在候选池中的名次。这些结果不参与 MLP 的 Top 10 筛选或训练。

`seed` 同时控制数据生成、划分、网络初始化和每轮训练洗牌；固定 CPU 计算和确定性 PyTorch 算法。脚本不会读取测试集来选模型、调超参数或筛设计。

## 安装与完整复现

推荐 Python 3.12。进入仓库根目录后运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m thinfilm.experiment --output results
```

macOS/Linux 可把上面后两条命令中的 Python 路径替换为 `.venv/bin/python`。默认训练参数：300 epochs、batch size 128、学习率 `0.001`。脚本会重新生成数据、四个模型、指标表、Top 10 验证表、Top 5 表、汇总 JSON 和全部图。可用 `--epochs`、`--batch-size`、`--learning-rate` 改训练参数；论文中使用的正式结果应始终和实际运行参数一致。

物理与可复现性检查：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m thinfilm.verify --output results
```

`verify` 会重新计算全部 5000 条 TMM 光谱并检查能量守恒、四组模型测试指标、Top 10/Top 5 顺序以及 10000 组候选的审计结果。

## 已运行的正式结果

这组结果由仓库中的默认设置实跑产生，完整精度见 `results/summary.json` 和 CSV 表。

| 训练样本数 | 固定测试集 MSE | 固定测试集 MAE |
| ---: | ---: | ---: |
| 500 | 0.0010753 | 0.022646 |
| 1000 | 0.0004907 | 0.015822 |
| 2000 | 0.0002151 | 0.010383 |
| 4000 | 0.0001040 | 0.007421 |

最终选中膜厚 `(d1,d2,d3,d4) = (168.031, 89.466, 168.173, 68.976) nm`；在 520 nm 处，MLP 预测 `R=0.662927`，TMM 验证 `R=0.654256`。该设计在全部 10000 个候选的 TMM 真值中排第 5；全池最佳为 `R=0.656081`，所选设计达到其约 99.72%。这是**固定随机候选池内**的比较，不是全局最优证明。

本机计时：MLP 预测全部 10000 个候选的 41 点光谱用时约 0.014 秒；TMM 直接计算 10000 个候选的单个目标波长约 0.008 秒。任务规模很小，不能据此宣称 MLP 在该场景有通用速度优势；本实验验证的是它能有效辅助筛选，而最终数值仍由 TMM 给出。

## 代码与结果文件

| 路径 | 内容 |
| --- | --- |
| `thinfilm/config.py` | 课程参数、个人波长和随机种子 |
| `thinfilm/physics.py` | TMM 的反射率与透射率 |
| `thinfilm/data.py` | 固定数据集与独立候选池 |
| `thinfilm/model.py` | MLP、训练、推理与误差指标 |
| `thinfilm/experiment.py` | 一键完成所有实验并保存结果 |
| `thinfilm/verify.py` | 对已保存实验结果做独立复核 |
| `thinfilm/plots.py` | 从真实数组绘制论文图 |
| `tests/` | 解析极限、独立菲涅耳算法、能量守恒、种子复现测试 |
| `results/config.json` | 实际物理设置与学生参数 |
| `results/training_size_metrics.csv` | 四种训练规模的验证、测试误差与训练时间 |
| `results/top10_tmm_verification.csv` | MLP Top 10 的膜厚与 TMM 复核 |
| `results/top5_designs.csv` | 按 TMM 结果重排的最终 Top 5 |
| `results/summary.json` | 论文引用的机器可读汇总与运行环境 |
| `results/figures/` | Figure 1–8 及辅助筛选审计图 |

`results/data/`、`results/models/` 和 `results/histories/` 在重新运行时生成，仓库中不必保存。数据和模型均可由固定种子与脚本复现。

## 图表索引

| 论文位置 | 本仓库图片 |
| --- | --- |
| Figure 1 总体流程 | `results/figures/figure_1_workflow.png` |
| Figure 2 膜系与 TMM 数据 | `results/figures/figure_2_tmm_data.png` |
| Figure 3 MLP 结构 | `results/figures/figure_3_mlp_architecture.png` |
| Figure 4 训练与验证损失 | `results/figures/figure_4_training_losses.png` |
| Figure 5 三个测试样本 | `results/figures/figure_5_test_spectra.png` |
| Figure 6 数据量效应 | `results/figures/figure_6_training_size.png` |
| Figure 7 最终设计的双方法光谱 | `results/figures/figure_7_selected_design.png` |
| Figure 8 最大误差测试案例 | `results/figures/figure_8_failure_case.png` |
| 辅助图：Top 10 预测与验证 | `results/figures/screening_audit.png` |

模型输出层为线性层，因此少量 MLP 预测值理论上可能越出 `[0,1]`；结果表记录了这一数量。物理结论均基于 TMM 验证。计时随机器和软件环境变化，不能直接当作通用加速比；评估筛选质量请看 TMM 核验值和全候选池审计。
