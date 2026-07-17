# Related Work Matrix

| Work | Decision object | Label | Evaluation implication |
| --- | --- | --- | --- |
| TASO (SOSP 2019) | tensor graph substitutions | analytical/measured cost | rewrite search is prior art |
| TENSAT (MLSys 2021) | e-graph extraction | estimated runtime | merged shared-input matmul is prior art |
| Kaufman et al. (MLSys 2021) | TPU fusion/configuration | real TPU runtime | GraphSAGE learned fusion cost is direct prior art |
| X-RLflow (MLSys 2023) | sequential graph rewrites | end-to-end latency | GNN rewrite selection is direct prior art |
| TpuGraphs (NeurIPS 2023) | same-graph configurations | real TPU runtime | group split/ranking/top-k are required baselines |
| Cross-config attention (2025) | candidate configuration set | pairwise ranking | set-aware pairwise GNN alone is not novel |
| DNNFusion / Welder | fusion plans | analytical + measured cost | strong non-learning baselines are required |
| PyTorch Inductor | graph lowering/autotune | measured kernel runtime | default and max-autotune are system baselines |
| vLLM / Chitu | production LLM MLP | production kernels | merged gate/up + fused activation is production baseline |

Bibliographic links and the full novelty discussion are maintained in
`docs/rewrite_research_plan.md` sections 2 and 18.
