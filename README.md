# Rec2026

CTR/CVR 精排模型，序列特征 + 非序列特征混合建模。

## Time Feature Design

时间特征从两个维度独立建模，采用不同分桶策略。

### Seq Time Features

序列内每个事件附带时间戳，从中提取 **hour/dow/weekend** 作为虚拟 sideinfo 特征拼入序列（`add_seq_time_attrs`），让模型感知序列中每个事件发生在"周日晚上 10 点"还是"周一凌晨 3 点"。序列跨度长（数天~数周），train 中覆盖了所有 hour×dow 组合，因此 test 不会遇到未见过的分布。

此外，可选使用 Time Buckets（相对时间差经非线性分桶后加到 token 上）和 Fourier 时间编码，增强模型对事件远近关系的感知。

分桶边界（`BUCKET_BOUNDARIES`）从 5 秒到 1 年非均匀分布：

| 范围 | 步长 |
|------|------|
| 5s ~ 60s | 每 5s |
| 2min ~ 10min | 每 1min |
| 15min ~ 1h | 每 15min |
| 1.5h ~ 6h | 每 30min |
| 9h ~ 24h | 每 3h |
| 2d ~ 7d | 每天 |
| 13d ~ 30d | — |
| 50d ~ 180d | — |
| ~1yr | 兜底 |

`bucket_id = searchsorted(BUCKET_BOUNDARIES, time_diff) + 1`（0=padding），`token_emb += time_embedding[bucket_id]`。

序列时间跨度长（数天~数周），不会出现 row timestamp 中 train/test 时间段不交叉的问题，因此可以按天/小时/分钟细分而不用担心泛化。

### Row Time Features

行级 timestamp 是当前预测时刻，train 和 test 的时间分布存在系统性偏移：

| 分集 | 时间范围 | 主要分布 |
|------|---------|---------|
| Train | 周四~周日 | 82% 周日，晚高峰 22-24 点 |
| Eval | 周日最后 27 分钟 | 周日深夜 |
| Test | 周一凌晨 | 65% 为 hour=1（周一 1-2 点） |

直接按小时 embedding 的问题：test 的"周一凌晨"与 train 的"周日深夜"是不同的 embedding，模型学到的是周日模式，无法泛化到周一。为此设计了 **is_workday × hour_segment 组合分桶**：

1. **Hour → 8 段**：

   | segment | hours | 含义 |
   |---------|-------|------|
   | 0 | 3-6 | 凌晨 |
   | 1 | 7-10 | 早上 |
   | 2 | 11-12 | 中午 |
   | 3 | 13-14 | 午间 |
   | 4 | 15-16 | 下午 |
   | 5 | 17-19 | 下班 |
   | 6 | 20-22 | 晚上 |
   | 7 | 23-24, 1-2 | 深夜 |

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

## Model Architecture

PCVRHyFormer，核心流程：

```
features → NS tokens（user / item / time）
seqs     → Seq tokens（每域独立 transformer）
queries  → Learnable queries × seq tokens → cross-attention
↓
HyFormer Blocks（Token Mixer + Token FFN）× N
↓
classifier → logits
```

## Data

详见 `data/EDA.md`。
