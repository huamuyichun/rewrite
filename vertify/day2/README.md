# Day 2: Torch FX Block Extraction

日期：2026-05-25

## 目的

Day 2 的目标是把 Day 1 固定的 `Transformer MLP block` 抽成可重复的 Torch FX graph。

本阶段不做：

- candidate rewrite 枚举；
- latency profiling；
- heuristic baseline；
- GNN 或其他学习模型；
- chitu 接入。

## 固定对象

本日使用一个最小 SwiGLU-like MLP block：

```text
x -> gate_proj(x)
x -> up_proj(x)
hidden = silu(gate) * up
out = down_proj(hidden)
```

这是 LLM Transformer MLP 中常见的 gate/up/down projection 结构，适合作为后续 fusion-related rewrite candidate 的最小对象。

## 环境策略

不修改 `chitu_clean`。

Day 2 使用已有专用环境：

```text
/pub/data/hjwz/miniconda3/envs/rewrite_miniexp
```

原因：

- 该环境已有 PyTorch；
- `torch.fx` 可用；
- CUDA 可用；
- 避免改动 `chitu_clean`。

## 输出文件

运行 `extract_mlp_fx.py` 后生成：

- `fx_graph_code.py`：Torch FX 生成的 Python forward 代码；
- `fx_graph_readable.txt`：FX graph 的 readable IR；
- `fx_nodes.csv`：节点表；
- `fx_nodes.json`：节点表 JSON 版本；
- `metadata.json`：运行环境、shape、dtype、节点数、eager 与 FX 输出差异；
- `LOG.md`：Day 2 日志。

## Day 2 验收标准

Day 2 完成时应确认：

1. MLP block 可以被 `torch.fx.symbolic_trace` 抽取；
2. FX graph 中保留 gate/up/down projection、silu、mul 等关键节点；
3. shape 和 dtype 信息可通过 `ShapeProp` 写入节点元数据；
4. FX GraphModule 与 eager module 输出一致；
5. 产出结构化节点表，供 Day 3 定义 rewrite candidate space。
