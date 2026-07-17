# Minimal Rewrite Plan Selection Experiment
## 1. 实验设定
- Block 类型：Transformer MLP / SwiGLU-like block
- Rewrite family：fusion-related gate/up projection variants
- Graph 数量：60
- 每图候选数：6
- Backend：PyTorch eager CUDA
- Device：NVIDIA A100-PCIE-40GB
- dtype：fp16
- batch_size：1
- seq_len：[128, 512, 2048]
- hidden_dim：[384, 512, 768, 1024, 1280, 1536, 2048]
- intermediate_dim：[768, 1024, 1280, 1536, 2048, 2304, 2560, 3072, 3328, 3840, 4096, 4608, 5120, 6144, 8192]
- warmup / measure：10 / 30
- selector decision repeats：100
- 偏差说明：第一版未接真实编译器 pass manager；用等价 PyTorch MLP 计算图变体代表 candidate rewrite plans，用于最小体验实验。

## 2. 主表 1：整体结果
| Selector | Median Speedup | Geomean Speedup | Median Regret | P90 Regret | Win Rate | P50 Decision Time (ms) | P95 Decision Time (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1.0000 | 1.0000 | 0.0307 | 0.1871 | 0.0000 | 0.0001 | 0.0001 |
| Heuristic | 1.0132 | 1.0296 | 0.0000 | 0.0774 | 0.4000 | 0.0050 | 0.0053 |
| Oracle | 1.0307 | 1.0555 | 0.0000 | 0.0000 | 0.5500 | 0.0008 | 0.0009 |

## 3. 主表 2：按 shape 分组结果
| Shape | Selector | Median Speedup | Median Regret | Win Rate | P50 Decision Time (ms) |
| --- | --- | --- | --- | --- | --- |
| s0_seq128 | Baseline | 1.0000 | 0.0680 | 0.0000 | 0.0001 |
| s0_seq128 | Heuristic | 1.0632 | 0.0000 | 0.8500 | 0.0050 |
| s0_seq128 | Oracle | 1.0680 | 0.0000 | 0.8500 | 0.0008 |
| s1_seq512 | Baseline | 1.0000 | 0.0333 | 0.0000 | 0.0001 |
| s1_seq512 | Heuristic | 1.0052 | 0.0096 | 0.2000 | 0.0050 |
| s1_seq512 | Oracle | 1.0333 | 0.0000 | 0.6000 | 0.0008 |
| s2_seq2048 | Baseline | 1.0000 | 0.0114 | 0.0000 | 0.0001 |
| s2_seq2048 | Heuristic | 1.0092 | 0.0000 | 0.1500 | 0.0050 |
| s2_seq2048 | Oracle | 1.0114 | 0.0000 | 0.2000 | 0.0008 |

## 4. Candidate 差异与稳定性
- Median candidate spread：0.1711
- P90 candidate spread：0.3178
- Graphs with >2% candidate spread：1.0000
- Median latency CV：0.0204
- P90 latency CV：0.0427

## 5. 图文件
- speedup_distribution.png：Heuristic selector 的 speedup 分布
- regret_distribution.png：Heuristic selector 的 regret 分布

## 6. 结论
- Candidate plans 是否存在稳定 latency 差异：是。候选 spread 中位数 0.1711，P90 CV 0.0427。
- Heuristic selector 是否优于 baseline：否/不明显。Median speedup 1.0132，win rate 0.4000。
- Regret 是否可接受：是。Median regret 0.0000，P90 regret 0.0774。
- Decision time 是否足够低：是。P50 0.0050 ms，P95 0.0053 ms。
- 是否值得进入下一阶段：未达到第一版成功门槛，下一步应先诊断 candidate diversity、profiling 稳定性或 heuristic 特征，而不是直接上 GNN。
