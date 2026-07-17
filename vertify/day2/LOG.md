# Day 2 日志

日期：2026-05-25

## 1. 做了什么

创建 Day 2 工作目录：

```text
/pub/data/hjwz/rewrite/vertify/day2
```

创建脚本：

```text
extract_mlp_fx.py
```

该脚本定义一个最小 SwiGLU-like Transformer MLP block，并使用 Torch FX 抽取图结构。

## 2. 目的是什么

Day 2 的目的不是开始实验测 latency，而是确认：

```text
Day 1 选定的 Transformer MLP block 能够被 Torch FX 稳定表示。
```

只有先得到稳定的 FX graph，Day 3 才能定义 candidate rewrite plans，Day 4 才能实现枚举器。

## 3. 环境决策

默认 `base` Python 没有安装 torch。

用户要求尽量不要动 `chitu_clean`，因此 Day 2 不使用 `chitu_clean` 作为工作环境。

当前选择已有专用环境：

```text
/pub/data/hjwz/miniconda3/envs/rewrite_miniexp
```

原因：

- PyTorch 可用；
- CUDA 可用；
- `torch.fx` 可用；
- 不需要修改 `chitu_clean`。

## 4. 固定 block

当前 block：

```text
gate = gate_proj(x)
up = up_proj(x)
hidden = silu(gate) * up
out = down_proj(hidden)
```

默认 shape：

```text
batch_size = 1
seq_len = 128
hidden_dim = 1024
intermediate_dim = 4096
dtype = fp16
```

这是 Day 2 的单 shape 抽图配置，不代表最终完整 shape sweep。

## 5. 预期输出

脚本运行后应生成：

```text
fx_graph_code.py
fx_graph_readable.txt
fx_nodes.csv
fx_nodes.json
metadata.json
```

## 6. 实际运行结果

运行命令：

```text
conda activate rewrite_miniexp
python /pub/data/hjwz/rewrite/vertify/day2/extract_mlp_fx.py --out-dir /pub/data/hjwz/rewrite/vertify/day2
```

运行环境：

```text
python env: /pub/data/hjwz/miniconda3/envs/rewrite_miniexp
torch: 2.10.0+cu129
device: NVIDIA A100 80GB PCIe
```

输出结果：

```text
num_fx_nodes = 7
max_abs_diff_eager_vs_fx = 0.0
status = ok
```

生成文件：

```text
fx_graph_code.py
fx_graph_readable.txt
fx_nodes.csv
fx_nodes.json
metadata.json
```

FX graph 关键节点：

```text
x -> gate_proj
x -> up_proj
gate_proj -> silu
silu + up_proj -> mul
mul -> down_proj
down_proj -> output
```

节点表中已保留：

- op 类型；
- target；
- args；
- users；
- shape；
- dtype。

## 7. 当前结论

Day 2 验收通过。

已经确认：

1. Day 1 选定的 Transformer MLP / SwiGLU-like block 可以被 Torch FX 抽取；
2. FX graph 保留了 gate/up/down projection、silu、mul 等后续 rewrite 所需关键结构；
3. `ShapeProp` 能写入 shape 和 dtype；
4. FX GraphModule 与 eager module 输出一致；
5. 当前已有结构化节点表，可作为 Day 3 定义 candidate rewrite space 的输入。

下一步 Day 3 应做：

```text
基于这个 FX graph，定义 4-8 个语义等价 candidate rewrite plans。
```


## 8. Roadmap Day 2 交付物核对

对应 `/pub/data/hjwz/rewrite/docs/verified_execution_roadmap.md` 中 Day 2 要求：

1. 1 个可重复提取的 block 样本：已完成，对应 `extract_mlp_fx.py` 和 `metadata.json`。
2. 1 份图表示样例：已完成，对应 `fx_graph_readable.txt`、`fx_nodes.csv`、`fx_nodes.json`。
3. 1 个最小数据结构定义：已完成，对应 `data_schema.md`。

结论：Day 2 的 roadmap 交付物已补齐。
