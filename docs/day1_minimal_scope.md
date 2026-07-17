# Day 1：最小实验边界

## 1. 结论

我建议你现在就把第一阶段实验边界定死为下面这一版，不要再摇摆。

- backend：`PyTorch eager + torch.compile(inductor)`
- IR：`Torch FX`
- block：`Transformer MLP block`
- rewrite family：`elementwise reorder / fusion-related`，先聚焦在 `Linear/GEMM 后逐点链` 这一类

这是我当前最推荐的一版，不推荐你一开始选 ONNX / Relay，也不推荐先做 attention。

## 2. 为什么推荐这一版

### 2.1 backend 选 `PyTorch eager + torch.compile(inductor)`

原因：

1. 你的前两周目标不是做系统接入，而是尽快证明 selection 问题存在。
2. PyTorch 路线最容易完成：
   - block 抽取简单
   - 图抓取容易
   - profiling 成熟
   - 后续可直接比较 eager 与编译后执行
3. `torch.compile` 至少给你一个真实 backend lowering / fusion 环境，而不是纯静态代理。

为什么不推荐现在就绑死 `chitu`：

- 你前面已经确认 `chitu` 当前不像现成的 rewrite pass playground。
- 现在硬接入只会把问题从“研究闭环”拖成“系统工程”。
- 现阶段 `chitu` 更适合作为后续验证平台，不适合作为 Day 1 主实验宿主。

### 2.2 IR 选 `Torch FX`

原因：

1. 它和 PyTorch block 抽取天然对齐。
2. 你要做 candidate graph 枚举，FX 比 ONNX 更容易直接改写和重建子图。
3. 前两周最重要的是快速可控，不是 IR 的通用性。

为什么不推荐现在用 ONNX：

- ONNX 导出链条会引入额外不确定性。
- 很多 rewrite 先在 FX 上做更顺手。
- 你当前不需要跨框架通用 IR 才能证明问题成立。

为什么不推荐现在用 Relay：

- 学习和工程门槛更高。
- 会把时间花在 TVM/Relay 细节，不利于两周闭环。

### 2.3 block 先只做 `Transformer MLP block`

建议具体实例：

`x -> Linear -> activation -> elementwise ops -> Linear`

优先考虑来自 LLaMA / Qwen / Mistral 的 FFN/MLP 子块，但第一周只要先固定一种结构化模板，不必一开始就混多个模型族。

原因：

1. MLP block 结构简单，提取和改写都比 attention 更稳。
2. 它天然带有 `GEMM + bias + activation + multiply/add` 这种适合 fusion / reorder 的局部结构。
3. 如果你连 MLP block 上都造不出 candidate diversity，那题本身就要重审。

为什么暂不优先 attention：

- attention 的 shape、mask、layout、kernel 路径更复杂。
- 太早做 attention，容易把问题复杂度和系统噪声混在一起。
- attention 更适合作为第二阶段扩展，而不是第一阶段最小闭环。

### 2.4 rewrite family 先做 `elementwise reorder / fusion-related`

建议第一版只保留下面一小类：

`Linear/GEMM 后逐点链的等价重排`

例如目标空间围绕这些变化：

- bias/add 与 activation 的相对位置变化（仅限语义严格等价情形）
- 多个逐点 op 的 regroup / reassociation
- 可影响 fusion 边界的逐点顺序变化
- reshape/view/contiguous/transpose 周边可消去或可下沉的轻量变形

这里要严格控制：

- 只保留语义明确、数值稳定、容易验证等价性的 rewrite
- 不要一开始碰太多 layout-heavy 规则
- 不要一开始引入需要复杂代数证明的 rewrite

原因：

1. 这类 rewrite 最容易形成“局部看差不多，但编译后执行未必一样”的候选。
2. 它最适合验证你的核心命题：图级上下文是否影响最终选择。
3. 它比 transpose-heavy family 更容易先跑通枚举与 profile。

## 3. 当前不推荐的选择

下面这些我建议现在不要选。

### 3.1 不推荐 Day 1 就做 attention block

原因不是它不重要，而是它会拖慢闭环。

只有当下面任一情况成立时，才值得切到 attention：

- MLP block 的 candidate diversity 太弱
- simple heuristic 几乎等于 oracle
- 你确认当前 MLP family 不足以体现图级交互

### 3.2 不推荐 Day 1 就做 transpose/layout movement 为主

原因：

- 这类规则常常更依赖 backend 和 memory layout 细节。
- 早期很容易出现“候选很多，但大量被 lowering 抹平”的问题。
- 这类 family 可以做，但更适合作为第二优先 family。

### 3.3 不推荐 Day 1 就追求通用多模型、多 backend

原因：

- 这是论文后期的泛化问题，不是现在的问题存在性验证。
- 过早扩范围，会直接损害两周内的可交付性。

## 4. 最小实验对象定义

建议把第一周的对象严格定成：

- 输入：一个固定 shape 的 MLP block FX graph
- 计划空间：5–20 个合法 candidate plans
- 每个 candidate：能还原为可执行 PyTorch 模块或 FX graph
- 执行环境：单卡 GPU，固定 dtype，固定 batch size，固定 seq len
- 输出：每个 candidate 的稳定 latency 与排序

推荐的固定条件第一版：

- dtype：`fp16` 或 `bf16`，二选一，按你机器当前最稳的那种
- batch size：先固定 1
- sequence length：先固定 1 个值，不要扫太多
- hidden/intermediate size：先模仿一个典型 LLM MLP 配置

这里重点不是“真实覆盖所有设定”，而是先造出一个稳定、可测、可选的问题实例。

## 5. Day 1 必须明确写死的不做项

第一阶段不做：

- 不做 attention 主实验
- 不做多 backend
- 不做全模型 end-to-end rewrite
- 不做 e-graph 全空间搜索
- 不做 online serving 集成
- 不做 `chitu` 深度接入
- 不做复杂 GNN
- 不做大规模 shape 泛化

这是必须写死的。否则范围一定会散。

## 6. Day 1 验收标准

今天结束前，你应该能把下面 6 件事写成固定设定：

1. backend 只用 `PyTorch eager + torch.compile(inductor)`
2. 图表示只用 `Torch FX`
3. block 只用 `Transformer MLP block`
4. rewrite family 只做 `Linear/GEMM 后逐点链的等价重排`
5. 实验环境先固定单卡、单 dtype、单 shape 设定
6. 明确列出第一阶段不做项

如果这 6 条还没定死，就不要进入 Day 2。

## 7. 我对这版边界的判断

这版边界的优点是：

- 足够小，两周内真有机会闭环
- 足够硬，仍然保留 rewrite selection 的研究味道
- 足够稳，不会一开始就被系统集成拖死

它的缺点也要说清：

- 新意上偏保守
- 如果 candidate diversity 不够，可能很快要换 family
- 如果 inductor 把大部分候选都抹平，第一周就会暴露问题

但即使这样，我仍然推荐这一版。

因为现在最需要的不是“最华丽的问题设定”，而是“最快证明这个题值不值得继续投”。

## 8. 下一步

按这版边界，下一步最该做的是：

1. 选一个具体 MLP block 模板
2. 定它的固定 shape
3. 用 FX 抽出来
4. 开始定义第一批 rewrite rule

如果你要，我下一步可以直接继续给你写 `Day 2-3` 的可执行规格：
- 具体选哪种 MLP block
- block graph 最小数据结构怎么定
- 第一批 rewrite rule 应该具体到什么粒度