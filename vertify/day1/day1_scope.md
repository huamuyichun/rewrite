# Day 1 固定实验边界

## 1. 当前任务

本文件固定前两周验证任务的最小实验边界。

前两周的核心不是证明 GNN 有效，而是证明 rewrite plan selection 这个问题值得继续做：

- 有候选可选；
- 有稳定 latency 差异；
- 简单规则不能完全解决；
- 选择开销足够小。

## 2. 固定设定

### 2.1 Backend

固定为：

```text
PyTorch eager + torch.compile(inductor)
```

理由：

- PyTorch block 构造和执行最直接；
- Torch FX 抽图和改写方便；
- Inductor 提供真实 backend lowering / fusion 环境；
- 比直接接 chitu 更适合两周内验证问题存在性。

注意：

已有 `rewrite_miniexp` 第一版用的是 PyTorch eager CUDA。后续 Day 2-5 应逐步补上 `torch.compile(inductor)`，否则实验说服力仍偏弱。

### 2.2 IR

固定为：

```text
Torch FX
```

理由：

- 和 PyTorch block 自然对齐；
- 容易做候选图构造、替换和导出；
- 前两周不需要 ONNX / Relay 的跨框架通用性。

### 2.3 Block

固定为：

```text
Transformer MLP block
```

第一版推荐结构：

```text
x -> gate_proj / up_proj -> activation -> multiply -> down_proj
```

即 SwiGLU-like MLP。

理由：

- 结构简单；
- 高频出现在 LLM block 中；
- 有 gate/up/down projection 和逐点链，适合构造 fusion-related candidates；
- 比 attention 更容易控制 shape、mask、layout 和 kernel 路径。

### 2.4 Rewrite Family

固定第一阶段只做一类：

```text
Linear/GEMM 后逐点链的等价重排，优先 fusion-related gate/up projection variants
```

允许的候选方向：

- separate gate/up projection vs fused gate/up projection；
- `chunk` vs `split`；
- `F.silu(gate)` vs manual `gate * sigmoid(gate)`；
- out-of-place multiply vs inplace multiply；
- 轻量 view/reshape 周边变体，但必须保证语义明确。

暂不引入：

- 复杂 layout movement；
- 大量 transpose sinking；
- 需要复杂代数证明的 reassociation；
- 跨 block rewrite。

### 2.5 初始 Shape 与执行条件

固定初始条件：

```text
GPU: 单卡
batch_size: 1
dtype: fp16 优先
seq_len: 先单值跑通，再扩展到 128 / 512 / 2048
```

hidden / intermediate 维度：

- 先选择 LLM MLP 常见量级；
- 不在 Day 1 固定完整扫描表；
- Day 2-3 根据显存和运行时间确定最小可跑配置。

## 3. 第一阶段不做项

为避免范围扩散，第一阶段明确不做：

- attention 主实验；
- 多 backend；
- 多机多卡；
- 全模型 end-to-end rewrite；
- 大规模 e-graph 搜索；
- online serving 集成；
- chitu 深度接入；
- 复杂 GNN；
- 大规模 shape 泛化；
- 论文完整版数据集。

## 4. Day 1 判断

当前 Day 1 判断：

这两周的正确推进顺序是：

```text
固定范围 -> 抽取 FX block -> 定义 candidate plans -> profile -> simple baseline -> 判断是否值得上图模型
```

如果候选 latency 差异不稳定，优先改 profiling protocol 或换 rewrite family。

如果 simple baseline 接近 oracle，优先换更有图级交互的 candidate space，而不是硬上 GNN。

如果 profiling 稳定且 simple baseline 做不完，才进入结构化模型阶段。
