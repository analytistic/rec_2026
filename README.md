# Rec2026 - CTR/CVR 精排模型

CTR/CVR 联合建模，序列特征 + 非序列特征混合建模，探索 Scaling Law。

## Feature Tokenization Design

每条样本最终表示为 token 序列，所有特征统一转为 token，与序列特征一起输入模型进行交互。

### Tokenization 策略

| 特征类型 | 数据源 | Token 方式 | 输出维度 |
|---------|-------|-----------|---------|
| **int_value** | item_feat 6-16, user_feat 1-4, 50-105 等 | embedding lookup (vocab_size = map_range) | 1 × d |
| **int_array** | item_feat 14, user_feat 5,18,53,54,67,74 | 每个元素 embedding → sum/mean pool | 1 × d |
| **float_array** | user_feat 68(256d), 81(320d) | MLP 投影到 d 维 | 1 × d |
| **int_array_and_float_array** (变长, array_len=None) | user_feat 69-73 | int embedding + float weight → weighted sum | 1 × d |
| **int_array_and_float_array** (定长, array_len=10) | user_feat 83-85 | int 固定为位置索引, float 是值 → 拼接投影 | 1 × d |
| **seq_feature** | action_seq(228)×10, item_seq(181)×12, content_seq(841)×9 | 每步各 feat concat 后 projection → N × d | seq_len × d |

### Token 序列组成

```
token_seq = [
  # 非序列特征 tokens（逐 feature）
  [item_feat_6_token],        # int_value
  [item_feat_7_token],        # int_value
  ...                         # 其余 item_feature
  [user_feat_1_token],        # int_value
  [user_feat_68_token],       # float_array (MLP proj)
  [user_feat_69_token],       # int+float weighted sum
  ...
  [user_feat_83_token],       # int+float 定长拼接
  ...
  [itemid_token],             # item_id embedding
  [userid_token],             # user_id embedding
  # 序列特征 tokens（每组序列一串 token）
  [action_seq_0, action_seq_1, ..., action_seq_227],   # 228 tokens
  [item_seq_0, item_seq_1, ..., item_seq_180],          # 181 tokens
  [content_seq_0, content_seq_1, ..., content_seq_840], # 841 tokens
]
```

非序列特征 ~80 个 token，序列特征 ~1250 个 token，总计 ~1330 tokens。

### Token 化细节

#### int_value
```
token = Embedding(int_value, vocab_size=map_range)
```

#### int_array
```
tokens = [Embedding(v) for v in int_array]  # (n, d)
token = MeanPool(tokens)                     # (1, d)
```

#### float_array
```
token = Linear(in_dim=array_len, out_dim=d)  # (1, d)
```

#### int_array_and_float_array (变长)
```
embed = Embedding(int[i])                    # (1, d)
weight = float[i] / sum(float_array)         # 归一化
token = sum(embed[i] * weight[i])            # (1, d)
```

#### seq_feature (等长 int_array 序列)
```
# 每步：该位置所有 feat 的 embedding 拼接
step_tokens = [Embedding(feat_value) for feat in seq_step]  # (num_feats, d)
step_token = Linear(step_tokens.flatten())                   # (1, d)
# 整条序列
seq_tokens = [step_0, step_1, ..., step_{seq_len-1}]         # (seq_len, d)
```

## Scaling Law 探索方向

- **主维度**: embedding dim (d), hidden dim, num_layers
- **次维度**: 特征选择、序列长度截断

## Data

详见 `data/README.md`
