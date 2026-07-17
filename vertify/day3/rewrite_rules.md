# Day 3 Rewrite Rule List

日期：2026-05-25

## 1. 目的

Day 3 的目标是定义一个有限、合法、可落地的 rewrite candidate space。

本阶段不做：

- 自动枚举器；
- latency profiling；
- heuristic baseline；
- GNN；
- chitu 接入。

## 2. 输入图

Day 3 基于 Day 2 的 Torch FX graph：

```text
x -> gate_proj
x -> up_proj
gate_proj -> silu
silu + up_proj -> mul
mul -> down_proj
down_proj -> output
```

对应 block：

```text
gate = gate_proj(x)
up = up_proj(x)
hidden = silu(gate) * up
out = down_proj(hidden)
```

## 3. Rewrite Family

第一阶段只保留一类 rewrite family：

```text
fusion_related_gate_up_projection
```

它属于 Day 1 固定的范围：

```text
Linear/GEMM 后逐点链的等价重排，优先 fusion-related gate/up projection variants
```

## 4. Rule A: Separate Gate/Up Projection

保持 baseline 结构：

```text
gate = gate_proj(x)
up = up_proj(x)
```

该规则用于保留默认计划，并作为其他 candidate 的对照。

## 5. Rule B: Fused Gate/Up Projection

将两个 projection 合并为一个 projection：

```text
gate_up = gate_up_proj(x)
gate, up = split_or_chunk(gate_up)
```

语义条件：

```text
gate_up_proj.weight = concat(gate_proj.weight, up_proj.weight) along output dimension
```

该规则改变 GEMM 数量和后续 split/chunk 结构，可能影响 backend fusion 和 kernel 调度。

## 6. Rule C: Activation Form

两种 activation 表达：

```text
hidden_gate = F.silu(gate)
hidden_gate = gate * sigmoid(gate)
```

数学上：

```text
silu(x) = x * sigmoid(x)
```

注意：

fp16 下两种表达可能由于 kernel 和舍入路径不同产生微小数值差异，Day 4 必须做 tolerance-based equivalence check。

## 7. Rule D: Split Form

fused projection 输出拆分方式：

```text
gate, up = torch.chunk(gate_up, 2, dim=-1)
gate, up = torch.split(gate_up, intermediate_dim, dim=-1)
```

两者语义等价，但 FX 结构和 backend 处理路径可能不同。

## 8. Rule E: Multiply Form

两种 multiply 表达：

```text
hidden = hidden_gate * up
hidden_gate.mul_(up)
```

语义条件：

```text
hidden_gate 没有除 multiply 之外的其他 user
```

如果该条件不成立，inplace rewrite 不合法。

## 9. 当前 Candidate Set

当前固定 6 个 candidate plans：

| Candidate | Gate/Up | Split | Activation | Multiply | Baseline |
| --- | --- | --- | --- | --- | --- |
| `p0_baseline_separate_silu` | separate | none | `F.silu` | out-of-place | yes |
| `p1_separate_manual_silu` | separate | none | manual silu | out-of-place | no |
| `p2_separate_inplace_silu_mul` | separate | none | `F.silu` | inplace | no |
| `p3_fused_chunk_silu` | fused | chunk | `F.silu` | out-of-place | no |
| `p4_fused_split_silu_inplace` | fused | split | `F.silu` | inplace | no |
| `p5_fused_chunk_manual_silu` | fused | chunk | manual silu | out-of-place | no |

## 10. Day 3 结论

当前 candidate space 满足 Day 3 最小要求：

- 只包含 1 个 rewrite family；
- 包含 1 个 baseline plan；
- 包含 5 个非 baseline plans；
- 每个 candidate 有明确结构差异；
- 每个 candidate 有语义条件；
- 每个 candidate 后续可落成 PyTorch/FX 可执行变体。

Day 4 应做：

```text
把这 6 个 candidate plans 程序化落成可执行 module / FX graph，并做去重和等价性检查。
```
