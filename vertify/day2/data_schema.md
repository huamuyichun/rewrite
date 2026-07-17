# Day 2 最小数据结构定义

日期：2026-05-25

## 1. 目的

本文档定义 Day 2 阶段最小可用的数据结构。

当前阶段只需要描述一个可重复提取的 Transformer MLP block 及其 Torch FX 图表示，不定义 rewrite candidates，不定义 latency label。

## 2. Block 样本

一个 block 样本由以下配置唯一确定：

| 字段 | 含义 | 当前值 |
| --- | --- | --- |
| `block_type` | block 类型 | `transformer_mlp_swiglu` |
| `batch_size` | batch size | `1` |
| `seq_len` | sequence length | `128` |
| `hidden_dim` | hidden dimension | `1024` |
| `intermediate_dim` | MLP intermediate dimension | `4096` |
| `dtype` | tensor dtype | `fp16` |
| `seed` | 初始化与输入 seed | `20260525` |

当前 block 结构：

```text
gate = gate_proj(x)
up = up_proj(x)
hidden = silu(gate) * up
out = down_proj(hidden)
```

## 3. FX Node 表

`fx_nodes.csv` 和 `fx_nodes.json` 使用同一套字段。

| 字段 | 含义 |
| --- | --- |
| `node_idx` | 节点在 FX graph 中的拓扑顺序 |
| `name` | FX node 名称 |
| `op` | FX op 类型，例如 `placeholder`、`call_module`、`call_function`、`output` |
| `target` | 调用目标，例如 `gate_proj`、`silu`、`mul` |
| `args` | 当前节点依赖的输入节点 |
| `users` | 使用当前节点输出的后继节点 |
| `shape` | `ShapeProp` 推出的 tensor shape |
| `dtype` | `ShapeProp` 推出的 tensor dtype |

当前节点表样例：

```text
x -> gate_proj
x -> up_proj
gate_proj -> silu
silu + up_proj -> mul
mul -> down_proj
down_proj -> output
```

## 4. Graph 表示

当前最小图表示为：

```text
G = (V, E, X)
```

其中：

- `V`：FX nodes；
- `E`：由 `args` / `users` 推出的 data dependency；
- `X`：节点特征，当前包括 `op`、`target`、`shape`、`dtype`。

当前不包含：

- rewrite candidate id；
- candidate plan label；
- latency；
- ranking label；
- FLOPs / bytes；
- backend compile artifact。

这些字段属于 Day 3 以后再定义。

## 5. Day 2 验收对应关系

Roadmap 中 Day 2 的三个交付物对应为：

| Roadmap 交付物 | 当前文件 |
| --- | --- |
| 1 个可重复提取的 block 样本 | `extract_mlp_fx.py`、`metadata.json` |
| 1 份图表示样例 | `fx_graph_readable.txt`、`fx_nodes.csv`、`fx_nodes.json` |
| 1 个最小数据结构定义 | `data_schema.md` |

## 6. 结论

Day 2 的最小数据结构已经固定。

下一步 Day 3 可以在该结构上定义 candidate plan 数据结构，例如：

```text
candidate_id
rewrite_family
plan_desc
fx_graph_variant
semantic_equivalence_check
```
