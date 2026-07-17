# Minimal Rewrite Plan Selection Experiment
## 1. 实验设定
- Block 类型：Transformer MLP / SwiGLU-like block
- Rewrite family：fusion-related gate/up projection variants
- Graph 数量：3
- 每图候选数：6
- Backend：PyTorch eager CUDA
- Device：NVIDIA A100-PCIE-40GB
- dtype：fp16
- batch_size：1
- seq_len：[128, 512, 2048]
- hidden_dim：[512, 768]
- intermediate_dim：[1024, 2304, 3072]
- warmup / measure：2 / 3
- selector decision repeats：10
- 偏差说明：第一版未接真实编译器 pass manager；用等价 PyTorch MLP 计算图变体代表 candidate rewrite plans，用于最小体验实验。

## 2. 主表 1：整体结果
| Selector | Median Speedup | Geomean Speedup | Median Regret | P90 Regret | Win Rate | P50 Decision Time (ms) | P95 Decision Time (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1.0000 | 1.0000 | 0.0083 | 0.0906 | 0.0000 | 0.0001 | 0.0002 |
| Heuristic | 1.0083 | 1.0372 | 0.0000 | 0.0032 | 0.3333 | 0.0053 | 0.0058 |
| Oracle | 1.0083 | 1.0386 | 0.0000 | 0.0000 | 0.3333 | 0.0009 | 0.0010 |

## 3. 主表 2：按 shape 分组结果
| Shape | Selector | Median Speedup | Median Regret | Win Rate | P50 Decision Time (ms) |
| --- | --- | --- | --- | --- | --- |
| s0_seq128 | Baseline | 1.0000 | 0.1111 | 0.0000 | 0.0001 |
| s0_seq128 | Heuristic | 1.1111 | 0.0000 | 1.0000 | 0.0054 |
| s0_seq128 | Oracle | 1.1111 | 0.0000 | 1.0000 | 0.0009 |
| s1_seq512 | Baseline | 1.0000 | 0.0083 | 0.0000 | 0.0001 |
| s1_seq512 | Heuristic | 1.0083 | 0.0000 | 0.0000 | 0.0053 |
| s1_seq512 | Oracle | 1.0083 | 0.0000 | 0.0000 | 0.0009 |
| s2_seq2048 | Baseline | 1.0000 | 0.0000 | 0.0000 | 0.0001 |
| s2_seq2048 | Heuristic | 0.9961 | 0.0040 | 0.0000 | 0.0052 |
| s2_seq2048 | Oracle | 1.0000 | 0.0000 | 0.0000 | 0.0009 |

## 4. Candidate 差异与稳定性
- Median candidate spread：0.1852
- P90 candidate spread：0.2394
- Graphs with >2% candidate spread：1.0000
- Median latency CV：0.0074
- P90 latency CV：0.0165

## 5. 图文件
- speedup_distribution.png：Heuristic selector 的 speedup 分布
- regret_distribution.png：Heuristic selector 的 regret 分布

## 6. 结论
- Candidate plans 是否存在稳定 latency 差异：是。候选 spread 中位数 0.1852，P90 CV 0.0165。
- Heuristic selector 是否优于 baseline：否/不明显。Median speedup 1.0083，win rate 0.3333。
- Regret 是否可接受：是。Median regret 0.0000，P90 regret 0.0032。
- Decision time 是否足够低：是。P50 0.0053 ms，P95 0.0058 ms。
- 是否值得进入下一阶段：未达到第一版成功门槛，下一步应先诊断 candidate diversity、profiling 稳定性或 heuristic 特征，而不是直接上 GNN。
