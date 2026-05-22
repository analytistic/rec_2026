# Rec2026

CTR/CVR 精排模型，序列特征 + 非序列特征混合建模。

## Time Feature Design

时间特征从两个维度独立建模，采用不同分桶策略。

### Seq Time Features

序列内每个事件附带时间戳，从中提取 **hour/dow/weekend** 作为虚拟 sideinfo 特征拼入序列（`add_seq_time_attrs`），让模型感知序列中每个事件发生在"周日晚上 10 点"还是"周一凌晨 3 点"。序列跨度长（数天~数周），train 中覆盖了所有 hour×dow 组合，因此 test 不会遇到未见过的分布。

此外，可选使用 Time Buckets（相对时间差经非线性分桶后加到 token 上）和 Fourier 时间编码，增强模型对事件远近关系的感知。

分桶边界（`BUCKET_BOUNDARIES`）从 5 秒到 1 年非均匀分布：

| 范围         | 步长     |
| ------------ | -------- |
| 5s ~ 60s     | 每 5s    |
| 2min ~ 10min | 每 1min  |
| 15min ~ 1h   | 每 15min |
| 1.5h ~ 6h    | 每 30min |
| 9h ~ 24h     | 每 3h    |
| 2d ~ 7d      | 每天     |
| 13d ~ 30d    | —       |
| 50d ~ 180d   | —       |
| ~1yr         | 兜底     |

`bucket_id = searchsorted(BUCKET_BOUNDARIES, time_diff) + 1`（0=padding），`token_emb += time_embedding[bucket_id]`。

序列时间跨度长（数天~数周），不会出现 row timestamp 中 train/test 时间段不交叉的问题，因此可以按天/小时/分钟细分而不用担心泛化。

### Row Time Features

行级 timestamp 是当前预测时刻，train 和 test 的时间分布存在系统性偏移：

| 分集  | 时间范围         | 主要分布                     |
| ----- | ---------------- | ---------------------------- |
| Train | 周四~周日        | 82% 周日，晚高峰 22-24 点    |
| Eval  | 周日最后 27 分钟 | 周日深夜                     |
| Test  | 周一凌晨         | 65% 为 hour=1（周一 1-2 点） |

直接按小时 embedding 的问题：test 的"周一凌晨"与 train 的"周日深夜"是不同的 embedding，模型学到的是周日模式，无法泛化到周一。为此设计了 **is_workday × hour_segment 组合分桶**：

1. **Hour → 8 段**：| segment | hours      | 含义 |
   | ------- | ---------- | ---- |
   | 0       | 3-6        | 凌晨 |
   | 1       | 7-10       | 早上 |
   | 2       | 11-12      | 中午 |
   | 3       | 13-14      | 午间 |
   | 4       | 15-16      | 下午 |
   | 5       | 17-19      | 下班 |
   | 6       | 20-22      | 晚上 |
   | 7       | 23-24, 1-2 | 深夜 |
2. **深夜（seg=7）继承前一天的 DOW 状态**：凌晨 1-2 点属于前一天的深夜，`effective_dow = (dow - 2) % 7 + 1`
3. **is_workday = (effective_dow ≤ 5)**，即 Mon-Fri 为 workday，Sat-Sun 为 weekend
4. **time_code = is_workday × 8 + seg**，单一 Embedding(16, d_model)

周日 23-24（weekend+深夜）和周一 1-2（继承 weekend+深夜）映射到同一个 time_code，test 直接复用训练中学到的 weekend 深夜 embedding。

Row time 作为独立的 NS token（排在 item_ns 之后），避免被 user/item 的大量特征稀释。

### Domain Embedding

`use_domain_emb` 为 user/item/query 三类 token 各附加一个学习到的 domain 偏置向量。user/item/query 本身来自异构特征空间，经过 NS tokenizer 投影到同一 d_model 空间后由共享 FFN 处理——domain embedding 相当于"类型位置编码"，让共享 FFN 区分当前处理的是哪类 token。

### 效果实测

- **Seq time buckets**：序列内时间编码，有效。
- **Row time 独立 NS token**：之前时间特征混在 user int 特征里一起经 tokenizer，信号被稀释；独立 token 后 eval 持平、test 提升，说明时间信号得到了有效利用。
- **is_workday × segment 分桶**：解决 train/test 时间分布不一致问题，test 提升。

最后AUC: 0.822

## Multi-Hash Embedding

### 背景

Item int 特征的 train/test 分布存在严重偏移：

| FID | Train _other | Test _other | 问题                                |
| --- | ------------ | ----------- | ----------------------------------- |
| 7   | 78.7%        | 73.9%       | 长尾极大，test top rank 漂移        |
| 8   | 75.5%        | 68.2%       | 大量 test-only novel 值             |
| 12  | 78.7%        | 73.9%       | 同 f7，高度相关                     |
| 16  | 91.8%        | 88.4%       | 最稀疏，train/test top-5 完全无重叠 |

这些特征原始值域很大，但经过 mod 编码压缩到 21 个取值，碰撞严重，标准 Embedding(21, emb_dim) 无法区分碰撞到同一桶的不同 ID。Seq 侧也存在 vocab 极大的特征（seq_c/f47: 86.3M、f29: 5.8M），直接建 Embedding 不现实。

### 方案

多值 hash：k 个独立 hash 函数，各映射到 H 个桶，每桶对应 `Embedding(H, emb_dim/k)`，concat 回 `emb_dim`。

```
hash_idx_j = (_HASH_PRIMES[j] × (fid_idx + 1) + val) % (H - 1) + 1
hash_idx_j = 0 if val == 0   # padding
```

`_HASH_PRIMES = [100003, 200003, 300007, 500009]`，val=0 始终映射到 idx=0（可学习 padding）。

### 选定特征

**Item NS**（mod 21 碰撞最严重）：f16/512/4
**Seq**（vocab 过大，会被 emb_skip_threshold=1M 跳过）：seq_b/f69/512/4(64.7M), seq_c/f29/512/4(5.8M), f34/512/4(1.0M), f47/512/4(86.3M)

### 效果

infer 正确加载 hash 后 test AUC 0.824→**0.829**。

## Paired Int+Float Processor

### 背景

User 侧 f62-66（int+float 配对）和 f89-91（定长 10 的配对特征）同时提供类别 ID 和对应统计量。int 仅 21 个取值（f62 仅 10 个），但 float 分布极丰富（f66 有 1,389 种 int-float 组合）。简单做法是把 int 作为普通特征 embedding，float 作为 dense 特征拼入，两者关联信息丢失。

### 方案

`PairedProcessor`：每个 paired fid 独立处理，多槽位 int 值经 Embedding 后以 log1p(float_val) 为权重做加权求和，输出一个 emb_dim 向量，最后 concat 所有 fid 的向量投影为一个 d_model token。

两类 float 不同处理：

- **Count 型**（f62-66 的 count/sum/duration）：weight = log1p(float) / log1p(global_max[int])，用该 int 值在全局中的最大 float 归一化
- **Score 型**（f89-91 的匹配度得分）：weight = float，原始得分直接作为权重

同时将 paired int 从 user int 特征中分离，不再经过 NS tokenizer，避免 int+float 关联信息被 group embedding 平均池化稀释。

### 效果

同时配合 padding 修正，test AUC 0.822→0.824（paired processor 单独收益无法量化）。

## Effective Rank

Effective Rank 衡量隐藏表示矩阵 $\mathbf{H} \in \mathbb{R}^{N \times d}$ 的有效维度数，用于分析模型是否发生了表示坍缩（collapse）。

设 $\sigma_1, \ldots, \sigma_k$ 为 $\mathbf{H}$ 的奇异值（$k = \min(N, d)$），归一化奇异值分布为 $p_i = \sigma_i / \sum_{j=1}^k \sigma_j$，则：

$$
\text{erank}(\mathbf{H}) = \exp\left(-\sum_{i=1}^k p_i \ln p_i\right)
$$

即归一化奇异值分布的 Shannon 熵的指数。

- **erank ≈ 1**：表示坍缩到单个主成分，$d$ 维 embedding 只用了 1 维有效信息
- **erank ≈ k**：信息均匀分布在各正交维度，全部维度被充分利用
