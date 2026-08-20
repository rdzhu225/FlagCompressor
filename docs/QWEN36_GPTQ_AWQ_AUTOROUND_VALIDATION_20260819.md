# FlagOS-Compressor Qwen3.6 GPTQ/AWQ/AutoRound 完整验证报告

## 1. 结论

本次在 8 张 NVIDIA H20 上完成了 Qwen3.6-27B 和 Qwen3.6-35B-A3B MoE 的 GPTQ、AWQ、AutoRound 三种 W4A16 量化验证。六个量化产物均完成：

1. 校准和权重量化；
2. 原生 packed checkpoint 导出；
3. checkpoint 结构检查；
4. vLLM packed kernel 加载；
5. 实际文本生成；
6. GSM8K 前 100 题精度冒烟测试。

GSM8K 结果如下：

| 模型 | BF16 | GPTQ | 相对 BF16 | AWQ | 相对 BF16 | AutoRound | 相对 BF16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-27B | 91% | 89% | -2 pp | 85% | -6 pp | 84% | -7 pp |
| Qwen3.6-35B-A3B | 89% | 81% | -8 pp | 86% | -3 pp | 79% | -10 pp |

以上结果使用较小校准集和很低的搜索/训练迭代数，目标是验证功能、格式和端到端推理，不应作为生产参数下的最终精度排名。

## 2. 验证环境

| 项目 | 值 |
| --- | --- |
| 验证日期 | 2026-08-19 |
| 验证节点 | 远端 8 × NVIDIA H20 节点 |
| 容器 | `flagos_quant_infer` |
| Conda 环境 | `/opt/conda/envs/flagos` |
| Python | 3.12.7 |
| PyTorch | 2.11.0+cu130 |
| Transformers | 5.12.1 |
| vLLM | 0.24.0 |
| GPU | 8 × NVIDIA H20，每张约 143771 MiB |
| 远端工作目录 | `/data/flagos_quant_validation` |

模型源目录：

- Qwen3.6-27B：`/share-evpfs/flagos/models/Qwen3.6-27B`，约 52 GB，64 层；
- Qwen3.6-35B-A3B：`/share-evpfs/flagos/models/Qwen3.6-35B-A3B`，约 67 GB，40 层、256 experts、top-8。

## 3. 并行方式

并行粒度为一个“模型 × 量化算法”对应一个独立任务，而不是把单个量化任务拆到 8 张卡：

| 任务 | 模型 | 算法 |
| --- | --- | --- |
| 1 | Qwen3.6-27B | GPTQ |
| 2 | Qwen3.6-27B | AWQ |
| 3 | Qwen3.6-27B | AutoRound |
| 4 | Qwen3.6-35B-A3B | GPTQ |
| 5 | Qwen3.6-35B-A3B | AWQ |
| 6 | Qwen3.6-35B-A3B | AutoRound |

因此量化阶段最多同时使用六张卡，剩余卡可用于 BF16 基线或已完成 checkpoint 的推理验证。任务完成后会释放对应 GPU；验证末期只剩耗时最长的 35B GPTQ，因此当时只有卡 3 仍有显存占用。最终检查时八张卡均为 `0 MiB / 0%`，没有残留量化或 vLLM 进程。

## 4. 量化范围与格式

六组任务统一采用：

- 权重/激活位宽：W4A16；
- group size：128；
- 选择器：`--select linear`；
- 输出：可直接供推理框架加载的原生 packed checkpoint。

`linear` 选择器覆盖分类器识别出的 attention 和 MLP/MoE 线性投影：

- attention：`q_proj`、`k_proj`、`v_proj`、`o_proj` 及代码支持的同类命名；
- dense MLP：`gate_proj`、`up_proj`、`down_proj` 及代码支持的同类命名；
- MoE：routed experts 和 shared experts 的线性投影，包括 fused expert bank；
- 不包含 embedding、RMSNorm 和 `lm_head`，它们有独立标签或不是线性权重。

最终导出的 INT4 模块数量：

- Qwen3.6-27B：496；
- Qwen3.6-35B-A3B：31,030，主要来自 MoE experts。

### 4.1 GPTQ

- 兼容 AutoGPTQ 风格控制项；
- 本次使用 `desc_act=false`；
- 量化配置中 `quant_method=gptq`、`bits=4`、`group_size=128`；
- 关闭 act-order 的原因是 vLLM 0.24 Marlin 路径无法加载 Qwen3.6 中输出维度为 96 的投影与 `desc_act=true` 的组合。

### 4.2 AWQ

- AutoAWQ GEMM 兼容格式；
- `zero_point=true`；
- 本次快速搜索使用 `n_grid=2`；
- 量化配置中 `quant_method=awq`、`bits=4`、`group_size=128`。

### 4.3 AutoRound

- 使用 FlagOS-Compressor 内置原生实现，运行时不依赖安装官方 AutoRound 包；
- 算法参数和导出元数据兼容 AutoRound/GPTQ packed 推理路径；
- 量化配置中 `quant_method=gptq`、`algorithm=autoround`、`bits=4`、`group_size=128`、`desc_act=false`；
- 这一设计允许后续在国产芯片环境替换设备后端，而不把算法入口绑定到官方 CUDA 包。

## 5. 校准参数

校准文本：`/data/flagos_quant_validation/calibration.txt`，共 16 条。各任务实际参数：

| 模型 | 算法 | Samples | Seq length | 其他参数 |
| --- | --- | ---: | ---: | --- |
| 27B | GPTQ | 8 | 128 | `desc_act=false` |
| 27B | AWQ | 8 | 128 | `n_grid=2` |
| 27B | AutoRound | 8 | 128 | 10 iterations，batch size 2 |
| 35B-A3B | GPTQ | 4 | 64 | `desc_act=false` |
| 35B-A3B | AWQ | 4 | 64 | `n_grid=2` |
| 35B-A3B | AutoRound | 4 | 64 | 5 iterations，batch size 1 |

生产精度评估建议至少提高校准样本数、sequence length、AWQ `n_grid` 和 AutoRound iterations 后重新测试。

## 6. 有效 checkpoint

以下六个目录是最终有效产物：

| 模型 | 算法 | 输出目录 |
| --- | --- | --- |
| 27B | GPTQ | `/data/flagos_quant_validation/outputs/Qwen3.6-27B-gptq-noactorder-w4g128` |
| 27B | AWQ | `/data/flagos_quant_validation/outputs/Qwen3.6-27B-awq-zcnorm-w4g128` |
| 27B | AutoRound | `/data/flagos_quant_validation/outputs/Qwen3.6-27B-autoround-w4g128` |
| 35B-A3B | GPTQ | `/data/flagos_quant_validation/outputs/Qwen3.6-35B-A3B-gptq-noactorder-w4g128` |
| 35B-A3B | AWQ | `/data/flagos_quant_validation/outputs/Qwen3.6-35B-A3B-awq-zcnorm-w4g128` |
| 35B-A3B | AutoRound | `/data/flagos_quant_validation/outputs/Qwen3.6-35B-A3B-autoround-w4g128` |

配置结构检查：

| 模型 | 算法 | Architecture | Method/algorithm | Tensors | Shards | INT4 模块 | 文件总字节数 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 27B | GPTQ | `Qwen3_5ForCausalLM` | `gptq` | 2,339 | 5 | 496 | 17,755,325,440 |
| 27B | AWQ | `Qwen3_5ForCausalLM` | `awq` | 1,843 | 5 | 496 | 17,741,759,488 |
| 27B | AutoRound | `Qwen3_5ForCausalLM` | `gptq/autoround` | 2,339 | 5 | 496 | 17,755,325,440 |
| 35B-A3B | GPTQ | `Qwen3_5MoeForCausalLM` | `gptq` | 124,423 | 4 | 31,030 | 19,737,334,016 |
| 35B-A3B | AWQ | `Qwen3_5MoeForCausalLM` | `awq` | 93,393 | 4 | 31,030 | 19,545,968,896 |
| 35B-A3B | AutoRound | `Qwen3_5MoeForCausalLM` | `gptq/autoround` | 124,423 | 4 | 31,030 | 19,737,334,016 |

量化耗时：

| 模型 | GPTQ | AWQ | AutoRound |
| --- | ---: | ---: | ---: |
| 27B | 8 分 37 秒 | 2 分 02 秒 | 约 4 分 18 秒 |
| 35B-A3B | 97 分 00 秒 | 4 分 31 秒 | 约 6 分 09 秒 |

35B GPTQ 的耗时明显高于其他任务，主要因为需要逐个处理 31,030 个专家线性模块；运行期间还曾受到一个残留旧任务争抢资源影响，但最终有效任务已完整结束并导出。

## 7. 推理和精度评估

### 7.1 评估方法

- 数据集：canonical GSM8K test；
- 范围：前 100 题；
- 模板：Qwen chat template；
- thinking：`enable_thinking=false`；
- decoding：greedy，`temperature=0`；
- 最大生成长度：384 tokens；
- 提示要求输出 `FINAL_ANSWER: <number>`；
- 评分：抽取最终数字后 exact match；
- 六个量化模型和两个 BF16 基线的答案解析有效率均为 100%。

评估脚本位于远端：

`/data/flagos_quant_validation/qwen36_gsm8k_eval.py`

结果目录：

`/data/flagos_quant_validation/results/`

### 7.2 Qwen3.6-27B

| 模式 | 0–49 | 50–99 | 总计 |
| --- | ---: | ---: | ---: |
| BF16 | 44/50 | 47/50 | 91/100 |
| GPTQ | 43/50 | 46/50 | 89/100 |
| AWQ | 42/50 | 43/50 | 85/100 |
| AutoRound | 41/50 | 43/50 | 84/100 |

有效结果文件：

- BF16：`27b_bf16.json`、`27b_bf16_50_99.json`；
- GPTQ：`27b_gptq_noactorder.json`、`27b_gptq_noactorder_50_99.json`；
- AWQ：`27b_awq_zcnorm.json`、`27b_awq_zcnorm_50_99.json`；
- AutoRound：`27b_autoround.json`、`27b_autoround_50_99.json`。

### 7.3 Qwen3.6-35B-A3B MoE

| 模式 | 分段结果 | 总计 |
| --- | --- | ---: |
| BF16 | 42/50 + 47/50 | 89/100 |
| GPTQ | 18/25 + 20/25 + 23/25 + 20/25 | 81/100 |
| AWQ | 20/25 + 21/25 + 45/50 | 86/100 |
| AutoRound | 16/25 + 21/25 + 42/50 | 79/100 |

有效结果文件：

- BF16：`35b_bf16.json`、`35b_bf16_50_99.json`；
- GPTQ：`35b_gptq_noactorder_0_24.json`、`35b_gptq_noactorder_25_49.json`、`35b_gptq_noactorder_50_74.json`、`35b_gptq_noactorder_75_99.json`；
- AWQ：`35b_awq_zcnorm_0_24.json`、`35b_awq_zcnorm_25_49.json`、`35b_awq_zcnorm_50_99.json`；
- AutoRound：`35b_autoround_0_24.json`、`35b_autoround_25_49.json`、`35b_autoround_50_99.json`。

### 7.4 Packed kernel 验证

- 35B GPTQ 已确认使用 vLLM Marlin Linear 和 MARLIN WNA16 MoE packed kernel 完成加载和生成；
- AWQ 和 AutoRound 产物也均通过相应 packed kernel 的实际推理，而非仅检查配置文件；
- 所有有效产物均执行过真实文本生成。

## 8. 验证中发现并修复的问题

### 8.1 AWQ GQA 映射不兼容

Qwen3.6 的 GQA 结构中，某些 `v_proj -> o_proj` AWQ scale balance 关系维度不兼容。实现增加通用维度兼容过滤，跳过无效映射。

对应提交：`7bb989e fix: skip invalid AWQ GQA scale mapping`

### 8.2 Text-only export 配置丢失

Transformers 选择 text-only causal model 时，需要保留正确的嵌套配置和 architecture，否则导出 checkpoint 不能被后续推理框架正确识别。

对应提交：

- `c5132a7 fix: export the calibrated text model config`
- `ee7d014 fix: preserve nested config for text-only exports`

### 8.3 Qwen3.6 zero-centered RMSNorm

Transformers 的 `Qwen3_5RMSNorm` 使用有效权重 `1 + weight`。旧 AWQ equalization 直接对参数 `weight` 除以 scale，会破坏等价变换，导致导出模型生成乱码和精度为 0。

现已将 Qwen3.5/Qwen3Next 等 zero-centered RMSNorm 处理修正为：

`weight = (1 + weight) / scale - 1`

并增加回归测试。

对应提交：`443f253 fix: preserve zero-centered RMSNorm during AWQ scaling`

### 8.4 vLLM 0.24 的 Qwen3.5 text-only 注册缺失

当前 vLLM 环境已包含部分 Qwen3.5 text classes，但缺少 causal registry/marker interface。本次推理使用以下运行时兼容文件完成注册：

- `/data/flagos_quant_validation/sitecustomize.py`
- `/data/flagos_quant_validation/vllm_qwen35_text.py`

这两个文件只补充 vLLM 0.24 的运行时模型注册，不修改 checkpoint 权重，也没有加入 FlagOS-Compressor 仓库。

## 9. 无效或已被替代的产物

以下目录或结果仅用于排障，不能作为最终验证结果：

- `/data/flagos_quant_validation/outputs/Qwen3.6-27B-awq-w4g128`；
- `/data/flagos_quant_validation/outputs/Qwen3.6-35B-A3B-awq-w4g128`。

以上两个 AWQ checkpoint 是 zero-centered RMSNorm 修复前产物，会生成异常文本。必须使用目录名带 `-awq-zcnorm-` 的版本。

此外：

- `/data/flagos_quant_validation/outputs/Qwen3.6-27B-gptq-w4g128` 是 `desc_act=true` 的旧产物，结构有效，但当前 vLLM 不能加载输出维度为 96 的相关 kernel；
- `/data/flagos_quant_validation/outputs/Qwen3.6-35B-A3B-gptq-attention-w4g128` 是排障过程中生成的 attention-only 实验，不是完整 `linear` 量化产物；
- `27b_awq.json`、`27b_awq_smoke.json`、`35b_awq_*.json` 等不带 `zcnorm` 的结果文件属于修复前或冒烟阶段，不应计入最终精度。

为避免误删可能有排障价值的数据，这些旧产物仍保留在远端。

## 10. 代码状态

- 仓库：`flagos-ai/FlagOS-Compressor`；
- 分支：`feature/native-autoround-support`；
- Pull Request：https://github.com/flagos-ai/FlagOS-Compressor/pull/6；
- 远端最新提交：`443f253 fix: preserve zero-centered RMSNorm during AWQ scaling`；
- 本地分支已与 `origin/feature/native-autoround-support` 对齐；
- Ruff：通过；
- Pytest：101 passed。

本次功能相关提交包括：

- `59b89e8 feat: add calibrated GPTQ and AWQ export`
- `12b3c09 feat: add native AutoRound quantization`
- `7bb989e fix: skip invalid AWQ GQA scale mapping`
- `c5132a7 fix: export the calibrated text model config`
- `ee7d014 fix: preserve nested config for text-only exports`
- `443f253 fix: preserve zero-centered RMSNorm during AWQ scaling`

## 11. 后续建议

1. 使用更接近生产的校准设置重新测精度，例如 128–512 samples、512–2048 sequence length、AWQ 默认或更高 `n_grid`、AutoRound 200+ iterations；
2. 扩大评测集，不只使用 GSM8K 前 100 题，增加完整 GSM8K、MMLU、CEval/CMMLU 和长文本任务；
3. 在目标国产芯片上实现/接入 packed Linear 与 MoE kernel，并复用当前设备无关的 AutoRound 校准和导出逻辑；
4. 对 vLLM 新版本重新验证 Qwen3.5/Qwen3.6 原生注册，环境已原生支持后移除运行时兼容文件；
5. 为 Qwen3.6 MoE 增加长期 CI，覆盖 GPTQ/AWQ/AutoRound 导出配置、zero-centered RMSNorm 和 fused expert bank。
