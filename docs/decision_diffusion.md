# 行业条件 Decision Diffusion 与局部动作探索

## 项目与实际架构

这是自动出价离线训练基线，包含流量数据转轨迹、Decision Transformer / Decision Diffusion 训练，以及固定历史竞争价格下的离线回放。每个投放周期最多 48 个时间步。DD 使用 16 维状态（剩余时间、预算比例、历史出价／转化／流量等），输出出价倍率 `alpha`，每条流量的出价为 `alpha * pValue`。

代码位置与数据流：

| 环节 | 文件 | 行为 |
| --- | --- | --- |
| 轨迹构造 | `bidding_train_env/dataloader/rl_data_generator.py` | 聚合流量，记录行业、CPA 约束、真实总花费／转化和终止状态 |
| 数据处理 | `bidding_train_env/baseline/dd/dataset.py` | 按投放周期和广告主分组、校验完整性、标准化状态并补齐轨迹 |
| 生成模型 | `bidding_train_env/baseline/dd/DFUSER.py` | Temporal U-Net 去噪生成状态，再由条件逆动力学模型输出动作 |
| 探索增强 | `bidding_train_env/baseline/dd/exploration.py` | 离线 Q 集成拟合、局部扰动筛选、逆动力学伪标签 |
| 训练入口 | `main/main_decision_diffuser.py` | CLI 配置，训练、保存模型与配置 |
| 策略执行 | `bidding_train_env/strategy/dd_bidding_strategy.py` | 行业与 CPA 透传，状态标准化，生成并裁剪倍率 |

模型学习的是 `p(未来状态序列 | 已观测前缀, 总回报, CPA约束, 合规状态, 行业)`。一次采样覆盖规划窗口，但内部仍需多轮去噪；每个真实时间步重新规划并执行当前动作。它不是直接联合扩散全部状态与动作，也不能保证生成的是全局最优序列。逆动力学的输入顺序为 `[s(t-1), s(t-2), s(t), s(t+1), prompt, category_embedding]`；不存在的前序状态和终止后继用标准化空间的零向量表示。

## 条件表征与 CFG

行业原始 ID 映射到训练集的连续 ID，`nn.Embedding(K+1, D)` 最后一项为 NULL，未见行业也映射至 NULL。训练时默认以 15% 概率独立丢弃行业与整个提示。行业向量与扩散时间向量相加，再拼接提示向量，通过每个 U-Net 残差块注入；逆动力学也有独立的可训练行业嵌入和行业丢弃。

提示为 `[R / R_scale, CPA_constraint / CPA_scale, compliant]`。其中 R 为轨迹总转化，比例因子仅由训练 CSV 计算并随模型保存；验证数据必须复用训练元数据。合规标签为 `realAllCost <= CPAConstraint * realAllConversion`。零花费零转化标签为 1，正花费零转化为 0。逆动力学额外用 `-1` 表示未知合规，并以 15% 概率训练此分支，供无法验证 CPA 的扰动样本使用。

推理公式：

```text
epsilon = epsilon_null
        + w_prompt * (epsilon_prompt - epsilon_null)
        + w_category * (epsilon_full - epsilon_prompt)
```

`epsilon_null` 同时去掉提示与行业，`epsilon_prompt` 保留提示且行业为 NULL。默认 `w_prompt=1.2, w_category=1.0`；两者都为 1 时得到全条件预测，行业权重为 0 时去掉行业引导。丢弃训练提供有／无条件分支，不保证模型一定使用行业信息，也不保证未知行业的零样本泛化。

策略默认要求归一化总回报 1.0 和合规标签 1；`DFUSER.forward` 可传 `target_return` 和 `target_compliance`。大于 1 的回报属于训练范围外要求，不能解释为实际可达回报。CPA 提示是软条件，真实合规仍需回放／环境验证。

## Q 估值、扰动与伪标签的边界

保留真实状态扩散训练，探索只作为逆动力学的辅助目标，默认关闭。启用后先训练 3 个独立 MLP 组成的 Q 集成（共享行业嵌入、使用独立 bootstrap 样本掩码），然后冻结 Q。

```text
Q_i(s_t, a_t, category, CPA_constraint) ≈ sum(reward[t:]) / R_scale
candidate = clip(expert_action + Normal(0, perturb_std²), action_min, action_max)
delta = min_i Q_i(s, candidate) - max_i Q_i(s, expert_action)
```

这里 Q 拟合观测轨迹的 Monte Carlo return-to-go，估计行为策略后续动作下的回报；没有 Bellman 自举或动作最大化。不能把它称为 Q*，更不能声称它准确评估任意 OOD 动作。动作单位是原始 `alpha` 倍率，`perturb_std=0.1` 不是状态扩散的噪声强度。

候选需同时满足：抽样概率、保守回报差大于阈值、原动作和候选动作的集成标准差均小于阈值。被接受的候选得到局部估计标签 `R_pseudo = R + min(delta, max_return_delta)`；CPA 约束保持不变，合规状态改为未知 `-1`。干净专家损失始终保留，增强损失系数为 0.25。拒绝的候选不进入辅助损失。

**为何不把扰动后整条轨迹重新打标送进扩散模型？** 原日志中 `s(t+1)` 是原动作的结果，改变动作后不能直接当成新动作的真实后继。因此这里是局部逆动力学近似增强，仍借用了专家状态上下文，不能视为物理一致的反事实轨迹。若要让状态扩散学习更优 OOD 轨迹，应先用可回放的流量环境或经验证的动力学模型重新生成后继状态、总回报和真实 CPA，再接入数据集。Q 集成一致也不能排除共同外推误差。

## 数据与运行

CSV 必须包含：`deliveryPeriodIndex, advertiserNumber, timeStepIndex, state, action, reward, advertiserCategoryIndex, CPAConstraint, realAllCost, realAllConversion`。推荐由仓库生成器生成，并包含 `done`。短轨迹必须显式 `done=1`；重复、跳步、未闭合或字段冲突的轨迹直接报错。没有 `done` 时只接受完整 48 步。多个扩增数据源如果重复使用投放周期／广告主 ID，应分别训练或先显式建立唯一 episode ID，不能直接拼接混为一条轨迹。

仅安装 DD 开发依赖：

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dd.txt
.venv/bin/python -m pytest -q tests/test_conditional_dd.py
```

基线条件训练：

```bash
.venv/bin/python main/main_decision_diffuser.py \
  --train-data-path data/trajectory/trajectory_data.csv \
  --save-path saved_model/DDtest --train-epoch 20 --batch-size 64
```

开启局部探索：

```bash
.venv/bin/python main/main_decision_diffuser.py \
  --train-data-path data/trajectory/trajectory_data.csv \
  --save-path saved_model/DDtest --train-epoch 20 --batch-size 64 \
  --category-dropout 0.15 --exploration --q-epochs 10 \
  --perturb-std 0.1 --perturb-probability 0.5 \
  --uncertainty-max 0.1 --improvement-min 0.01 --max-return-delta 0.1
```

倍率上限默认取训练集最大动作的 1.1 倍（至少 1），可用 `--action-max` 明确指定；小于专家最大动作会报错。这是数值支持范围，不是预算／CPA 保证。推理严格加载相同的行业映射、状态均值／标准差、回报和 CPA 比例，不再把所有标准化状态裁剪到 `[-1,1]`。

输出：`diffuser.pt`（版本 2、结构配置、归一化参数、网络与优化器状态）、`training_config.json`（数据路径、种子、超参、Q 训练损失、接受候选数与接受率），启用探索时额外输出 `q_ensemble.pt`。DD v1 权重不兼容新增结构，加载时明确提示重新训练；原 `save_model` 的 TorchScript 导出未保留，当前策略使用 Python 模型加载。

离线回放前，将 `bidding_train_env/strategy/__init__.py` 中的策略选择切换到 `DdBiddingStrategy`。仓库默认策略仍是原 `PlayerBiddingStrategy`。原 `run_evaluate.py` 仅抽取第一个测试键且使用默认广告主参数，是接口演示，不能作为行业泛化或 CPA 效果报告；正式评估应遍历测试 episode 并传真实预算、行业和 CPA。

## 验证与后续实验

本次自动测试覆盖轨迹分组／终止／归一化、未知行业 NULL、行业丢弃与梯度、补齐掩码、候选动作边界及回报／未知合规伪标签、Q 与 DD 反向传播、单步短轨迹、最终时间步、保存加载一致性及带探索的完整训练入口。测试数据是合成数据，不构成收益、CPA 或 OOD 性能证据。

真实实验按投放周期划分训练／验证／测试，禁止同轨迹片段泄漏，复用训练归一化。至少比较：条件 DD（行业始终 NULL）、行业 DD、行业 DD + Q 扰动；保持数据、种子集合、训练预算、提示和采样步数一致。报告转化数、实际 CPA、合规率、带 CPA 惩罚的竞赛分数，以及行业分组结果。另做留一行业和稀疏行业实验；检查 Q 留出误差、校准、扰动幅度、接受率与回放收益的关系。只有回放／真实环境回报提升，才能支持“超越专家”或“OOD 鲁棒性增强”的结论。


本地验证结果：`13 passed`，无警告；训练 CLI `--help` 和 `git diff --check` 通过。

本地验证环境：Python 3.11.14、PyTorch 2.14.0、NumPy 2.4.6、Pandas 3.0.6，macOS CPU。旧 `requirements.txt` 的 PyTorch 1.12 环境未验证本扩展；请使用单独的 DD 依赖文件。训练去噪采用随机观测前缀，并使用与前向扩散一致的标准高斯采样；此前固定随机种子、半幅采样噪声与余弦时间表端点也已修正。
