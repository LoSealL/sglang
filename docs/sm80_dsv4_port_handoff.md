# DeepSeek-V4-Flash on A100 (sm_80): vLLM 移植总结 + sglang 继承方案

> 交接文档。vLLM 侧工作已完成并验证（分支 `feat/deepseek-v4-sm80`，/home/yuanqi_admin/works/vllm）；
> 本文供新会话在 /home/yuanqi_admin/works/sglang 开启 sm_80 分支实现时使用。
> 生成日期：2026-08-17

---

## Part 1 — vLLM 侧已验证的改动（要继承的内容）

### 1.1 最终成果（全量 DeepSeek-V4-Flash-0731，8×A100-80G，TP=8）

| 指标 | 数值 | 备注 |
|---|---|---|
| Decode conc=1（8k ctx） | 9 → **95 tok/s**（10.5ms/step） | FULL decode CUDA graph |
| Decode conc=128 | 1,105 → **4,001 tok/s**（util 96.4%） | |
| Prefill（32×1536） | **12,864 tok/s**（util 99%） | |
| 长上下文 prefill | 16k:8.4k / 32k:7.3k / 49k:6.3k / 100k:4.4k tok/s | TTFT@100k=22.5s |
| 长上下文 decode | 16k:35 / 32k:22 / 49k:16 tok/s | 见 1.5 已知 bug |
| 正确性 | 逐层 attention rel err ~3e-3（vs 暴力参考）；128-token 贪心与 eager 逐位一致 | 真实权重 |
| 权重显存 | ~21GB/卡（Marlin w8a16 + fp4→marlin-experts） | 156GB checkpoint |
| parser | reasoning/tool parser 全通过（含并行 tool call） | serve 已验证 |

### 1.2 核心 kernel / 代码（按依赖顺序）

**A. Triton sparse MLA attention kernel**
`vllm/v1/attention/ops/triton_sparse_mla.py`（~250 行）
- gather-free 两段式（SWA 窗口 + indexer top-k 压缩行）online-softmax bf16 kernel，head_dim=512
- Gemma 式 sink：sink 作为 softmax 的一个零值额外 logit（`m=sink, denom=1 if finite`），−inf = 无 sink
- **元素级 stride 寻址**（关键修复 `fb16844101`）：`offset = (slot//spb)*stride(0) + (slot%spb)*stride(1)`
  共享 KV 池的块 stride 可以不是行数整数倍（实测 800000/512=1562.5），整数除法行映射必错
- 索引 int64、−1 pad、`HAS_COMP` constexpr 特化 SWA-only 层

**B. Triton fp8 MQA logits（indexer 用，替换 DeepGEMM）**
`vllm/v1/attention/ops/triton_fp8_mqa_logits_cuda.py`（~270 行）
- `fp8_mqa_logits_cuda`（连续/varlen prefill）+ `paged_fp8_mqa_logits_cuda`（paged decode）
- 数学：`logits[m,n] = Σ_h relu(q[m,h]·k[n] * kv_scale[n]) * w[m,h]`；Q 的 per-token scale 已折入 weights
- **软件 e4m3fn 编解码**（Triton 在 sm_80 拒绝 fp8e4nv 指针）：
  - 解码 `_fp8e4m3_to_f32`：`(e+120)<<23 | m<<20` 重偏置，次正规数正确，NaN 字节(0x7F/0xFF)→±480（量化数据中不可达，有 `ponytail:` 标注）
  - 编码 `_fp32_to_e4m3_bits`（在 fused_indexer_q.py）：RNE、satfinite ±448、子正规数 magic-add；对 c10 规则 `>464→NaN 0x7F` 全对齐，穷举 bf16 + 10M 随机 fp32 与 torch 位级一致
  - **能力门控**：`USE_NATIVE_FP8 = is_rocm() or (is_cuda() and has_device_capability(89))`，sm89+ 保持原生 `.to(tl.float8e4nv)`，仅 ≤sm88 走软件路径（commit `90b056626d`）
- paged 版寻址同样元素级 stride（commit `f102d422cb`）；输出 workspace 用 caller 的 `max_model_len` 定宽，**免 `tensor.max()` host-sync**；只写 [0,ctx)，top-k 消费端按 seq_lens 界定扫描（`top_k_per_row_decode`/`persistent_topk` 均验证按 len 界定）

**C. Backend + attention 类**
`vllm/models/deepseek_v4/nvidia/triton_sparse.py`
- `TritonMLASparseBackend`：capability gate `major == 8`；bf16 plain-row KV（`use_fp8_ds_mla_layout=False`）
- `TritonSparseMetadataBuilder`：`_cudagraph_support = UNIFORM_BATCH if envs.VLLM_DSV4_ALLOW_CUDAGRAPH else NEVER`
- `DeepseekV4TritonMLAAttention`：`forward_mqa` decode/prefill 拆分，C4A 用 `compute_global_topk_indices_and_lens`（**传完整 block_table**，kernel 按绝对 req id 索引），C128A 用预算好的 c128a_* 字段
- `_o_proj`：`rocm_inv_rope_einsum`（纯 Triton 逆 RoPE）+ bf16 `wo_a` 缓存 + `wo_b`——注意 **wo_a 反量化前置**（见 D）

**D. 集成修复（真实权重才暴露）**
1. **wo_a 保 bf16 原始布局**（`325b739729`）：sm80 上 fp8 权重走 Marlin，其 finalize 会重排 wo_a，破坏 o_proj einsum。修法：模型 `load_weights` 末尾（finalize 之前）把 wo_a 块量化反量化为 bf16、换 `UnquantizedLinearMethod`，让 Marlin 跳过它
2. **DeepGEMM/FlashMLA 可用性门控**（`c70c34b744`）：metadata 构建等按 `is_deep_gemm_supported()`/flashmla 可导入性门控，sm80 不触发
3. **mHC 首层 prenorm GEMM 回退**（`f3bad711ec`）：`_torch_hc_prenorm_gemm` 替代 DeepGEMM broadcast 变体
4. **loader 兼容**（`795bbf5af6`）：`process_weights_after_loading` 钩子里做 dummy-loader 的 finalize
5. 模型 dispatch（`059284b4ae`）：`_select_dsv4_attn_cls` 在 `major==8` 返回 Triton 类

**E. 性能关键：conc=1 decode 的 FULL CUDA graph**（`0ef7f24d64`）
- 根因：CPU-bound。每步 ~7200 aten ops（~100ms Python），GPU 真实计算仅 6-11ms；8 rank 的 allreduce 核中位自旋 2ms×86 次/步等最慢 rank 的 CPU 提交
- 解法：模型自带 eager-break 机制（indexer/attention 段留在图外），其余段 FULL 捕获 → 10.5ms/step
- **已验证组合**（bit-identical 128-token 贪心 vs eager）：`VLLM_DSV4_ALLOW_CUDAGRAPH=1` + `mode=vllm_compile` + `cudagraph_mode=full_decode_only` + `pass_config(fuse_attn_quant=False)` + 非 eager
- **piecewise 两条路（breakable/compiled）在该模型上都损坏输出**——与上游 `MRV1_UNSUPPORTED_PIECEWISE_CUDAGRAPH_ARCHITECTURES` 封禁 DSv4 一致，不要碰
- FP8→Marlin 已有自动路由（`80<=sm<89` 自动 marlin w8a16），无需新代码

### 1.3 测试资产（全部在分支上）
- `tests/kernels/test_triton_sparse_mla.py`（含 all-masked 行、sink 边角）
- `tests/kernels/test_triton_fp8_mqa_logits_cuda.py`
- `tests/v1/attention/test_triton_sparse_backend_sm80.py`（含模拟 metadata 的 forward 测试）
- `tests/models/test_deepseek_v4_sm80_{dispatch,fp8_config,e2e}.py`（e2e 用 tiny 合成权重 <18GB）
- `scripts/`：bench（tiny TP=8 + 全量）、逐层数值探针（capture/check）、decode profiler

### 1.4 环境/上游坑（新会话必读）
- `.venv` 里的 `vllm_gguf_plugin` 插件 `override_quantization_method` 签名过期会崩 fp8 探测循环（我们打了 site-packages 补丁）
- `paged logits` 的 workspace 若用 `tensor.max()` 会引入每 C4A 层一次 host-sync（已改为传 max_model_len）
- 测试用 dummy 权重 ±1e-3 会经 DSv4 叠层 norm 下溢成 NaN——需重设初始化范围

### 1.5 遗留 bug（上游，非 sm80 引入）：ctx ≥ ~57k decode 乱码
- 现象：57344/65536/100k GARBLED；49152 OK；eager/graph 均坏；单 chunk（65536 budget）也坏 → 非图、非 chunk 边界竞态
- 证据链：topk_indices 全 0 → decode logits 含 300 NaN → 被 block_table 引用的 indexer cache slot 从未写入（scale=3e35 垃圾）→ 写入 kernel 重跑同输入能写上（magic 覆盖实验）→ 首跑早退
- 观测：8192-token chunk 的 compressed slot mapping `max_slot` 出现两种形态（2111 逻辑 vs 436671 物理池），写入端与读取端地址在长序列下不一致；用户判断为 vLLM 已知问题类
- **sglang 移植价值**：sglang 有独立的 slot-mapping 管线（见 Part 2 §6），此 bug 大概率不复现——移植本身就是一次交叉验证

---

## Part 2 — sglang 继承方案（调研结论）

### 2.0 总判断
sglang **不是**继承 vLLM 代码的形态：它 vendored 一切、零 `import vllm`，且已有一套**独立完整**的 DSv4 原生实现。因此"继承"= 把我们验证过的 **sm80 kernel 变体**插进 sglang 现有 dispatch 点，而不是搬模型代码。

### 2.1 sglang 现状（DSv4）
- 模型：`python/sglang/srt/models/deepseek_v4.py`（3735 行，`DeepseekV4ForCausalLM`）+ nextn/dspark 变体；MoE 复用 `deepseek_v2.DeepseekV2MoE(is_deepseek_v4=True)`；权重加载自带 HF 名重映射（`remap_weight_name_to_dpsk_hf_format`）
- Backend：`layers/attention/deepseek_v4_backend.py:500` `DeepseekV4AttnBackend`（注册名 `"dsv4"`，`attention_registry.py:151`）
  - decode/prefill 核心 = **sgl-kernel 的 FlashMLA**（`flash_mla_with_kvcache` / `flash_mla_sparse_fwd`）
  - indexer = `layers/attention/dsv4/indexer.py:881` `C4Indexer`，MQA-logits dispatch 在 :697-731：FP4→DeepGEMM fp8_fp4；CUDA 默认→`deep_gemm.fp8_paged_mqa_logits`；**XPU→`sgl_kernel.fp8_paged_mqa_logits_triton`（已有 Triton MQA kernel，仅 XPU 接线！）**；torch 回退
  - compressor = `dsv4/compressor.py`，Triton kernel 在 `python/sglang/kernels/ops/attention/dsv4/`
- **sm80 无任何门控**：DeepGEMM/FlashMLA-sparse 在 A100 上直接失败，这就是要修的入口

### 2.2 移植点映射表（vLLM → sglang）

| vLLM 资产 | sglang 插入点 | 动作 |
|---|---|---|
| `paged_fp8_mqa_logits_cuda` | `dsv4/indexer.py:697-731` dispatch 表 | 加 `is_sm80_supported()` 分支（参考旁边 XPU triton 分支的写法）。cache 契约一致：`[page,64,1,132]` u8 = fp8 值 + per-token fp32 scale。保留 max_len 免 sync 语义 |
| `_fp32_to_e4m3_bits` / `_fp8e4m3_to_f32` | `kernels/ops/attention/dsv4/quant_k_cache.py` 旁 | sglang 已有 `quant_to_nope_fp8_rope_bf16_pack_triton`（stride 参数化、软件 e4m3）——**先验证它 sm80 可跑**，大概率直接可用；不可跑再搬我们的位级版本 |
| `triton_sparse_mla.py` | `deepseek_v4_backend.py` 的 `forward_decode`/`forward_extend` | sm80 分支替换 FlashMLA 调用。**注意 DSv4 特有**：`attn_sink` 参数 + `extra_k_cache`（c4/c128 合并索引）语义要对齐 sglang 的 metadata（`swa_page_indices`/`extra_indices_in_kvcache`）。参考现成类比：`kernels/ops/attention/dsa/triton_sparse_mla.py`（V3.2 的 Triton sparse MLA，目前 gfx950-gated）——它不含 sink/extra_k_cache，需要扩 |
| 元素级 stride 寻址 | 所有读 KV 的 kernel | sglang 是**分层池**（SWA page128 / c4 page64 / c128 page2 / indexer 132B 行），无 vLLM 的 150528 packed pool——我们的元素级 stride 寻址天然兼容，但**不要**假设任何 packed 偏移 |
| wo_a bf16 保护 | `models/deepseek_v4.py` 权重加载（:3291 已有 fp8-wo_a streaming dequant） | **sglang 已解决**（加载时流式反量化），无需移植 |
| capability gate | `srt/utils/common.py:301` `is_sm80_supported()` 已存在（现无使用者） | 直接用 |
| FULL decode CUDA graph | sglang 的 cudagraph/overlap 体系 | 第二阶段再做；先把 eager 正确性打通 |

### 2.3 量化路由（sm80）
- **dense fp8 → marlin w8a16：sglang 已自动**（`fp8_utils.py:2011` `80<=sm<89 → can_auto_enable_marlin_fp8`）
- **fp4 experts：需要 `SGLANG_DSV4_FP4_DEQUANT=1`**（`fp8.py:383-386`，dequant 后走 Fp8MoEGEMM triton fused-MoE）。注意 `Mxfp4MarlinMoEMethod` 在 sm80 硬拒（`mxfp4_marlin_moe.py:143`，"requires SM90 or SM120"）——dequant 是唯一 A100 路径；vLLM 侧我们用的是 marlin-experts，sglang 用 dequant→triton MoE，功能等价、性能略差（可后续评估把 marlin moe gate 放宽到 sm80）
- KV cache：DSv4 默认 `fp8_e4m3` KV——sm80 上 indexer KV 写入走软件 e4m3（见上），**主 KV（SWA/c4/c128 bf16 池）建议 bf16**（对齐 vLLM sm80 决策：`use_fp8_ds_mla_layout=False`），即 `--kv-cache-dtype bf16` 或 override `_deepseek_v4_kv_cache_dtype`

### 2.4 建议实现顺序（新会话任务分解）
1. **sm80 gate + indexer Triton logits**：`dsv4/indexer.py` dispatch 加 sm80 分支（搬 `paged_fp8_mqa_logits_cuda` + contiguous 版），单测对拍（torch 参考 + 垃圾尾部 + packed 几何）
2. **quant/dequant kernel sm80 验证**：跑通 `quant_to_nope_fp8_rope_bf16_pack_triton` / `dequantize_k_cache_paged` 于 A100；不行则替换 encode 为我们的位级实现
3. **sparse MLA Triton kernel 接入**：扩 `dsa/triton_sparse_mla.py` 或新写 `dsv4/triton_sparse_mla.py`（带 sink + extra_k_cache），从我们 `vllm/v1/attention/ops/triton_sparse_mla.py` 起步改寻址/接口；`forward_decode/extend` 加 sm80 分支
4. **数值验证**：搬 `scripts/probe_dsv4_{capture,check}.py` 思路（逐层 hook + 暴力 softmax 参考），先 tiny 再全量；"The capital of France is → Paris." 作为 smoke
5. **MoE/长上下文**：`SGLANG_DSV4_FP4_DEQUANT=1` 起；跑 16k/32k/49k/57k/100k 曲线——**顺带交叉验证 vLLM 的 57k bug 是否 sglang 也存在**（不同 slot-mapping 管线，见 `dsv4/metadata_kernel.py` `_init_compressed_attn_metadata_kernel`：loc//4、loc//128 一次性算，比 vLLM 的双 builder 简单）
6. **性能**：conc=1 profile（同款 torch profiler worker 内剖析法）→ 视 CPU-bound 程度决定是否做 graph（sglang 的 overlap scheduler + cuda graph 域与 vLLM 不同，需单独评估）

### 2.5 关键差异警示（vLLM vs sglang）
- block_table 语义：vLLM 的 c4 topk→global slot 转换传**完整表**（绝对 req id）；sglang 的 `swa_page_indices` 是 metadata 预计算——对齐时以 sglang 现有 FlashMLA 调用参数为准
- sink 处理：sglang FlashMLA 路径如何传 sink 需查（vLLM 是 attn_sink Parameter）——Triton kernel 的 sink-as-logit 语义直接搬
- page_size==256 断言（backend :540）：保持，我们 vLLM 也是 256
- scheduler：sglang 的 chunked prefill 在 `managers/scheduler.py:3155` + `schedule_policy.py:1386`，SWA chunk cap 逻辑（:725）与 vLLM 不同——57k bug 若复现，从这里查起

### 2.6 vLLM 分支参考文件清单（复制源）
```
vllm/v1/attention/ops/triton_sparse_mla.py          # A: sparse MLA kernel
vllm/v1/attention/ops/triton_fp8_mqa_logits_cuda.py # B: paged/连续 fp8 logits + e4m3 位级码
vllm/models/deepseek_v4/nvidia/triton_sparse.py     # C: backend/attention 类（接口参考）
vllm/models/deepseek_v4/common/ops/fused_indexer_q.py     # e4m3 encode（能力门控示例）
vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py  # USE_NATIVE_FP8 门控示例
tests/kernels/test_triton_{sparse_mla,fp8_mqa_logits_cuda}.py     # 单测模板
scripts/probe_dsv4_{capture,check}.py               # 数值验证方法
```
对应 sglang 侧落点：
```
python/sglang/srt/layers/attention/dsv4/indexer.py:697-731   # logits dispatch
python/sglang/srt/layers/attention/deepseek_v4_backend.py    # forward_decode/extend
python/sglang/kernels/ops/attention/dsv4/                    # triton kernel 落点
python/sglang/srt/layers/attention/attention_registry.py     # dsv4 注册（无需改，backend 内分支）
python/sglang/srt/utils/common.py:301                        # is_sm80_supported
python/sglang/srt/configs/deepseek_v4.py                     # 配置
```

### 2.7 启动命令草案（sglang sm80，实现完成后）
```bash
SGLANG_DSV4_FP4_DEQUANT=1 python -m sglang.launch_server \
  --model-path /nvme2data/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --tp 8 --attention-backend dsv4 \
  --kv-cache-dtype bf16 \
  --context-length 49152 \
  --reasoning-parser deepseek_v4 --tool-call-parser deepseek_v4 \
  --port 8000
```
（kv dtype、context 上限等在实现中校正；57k bug 若 sglang 不复现则放开 context-length）

---

## Part 3 — sglang 实施决策记录（2026-08-18，设计已批准）

- 分支：`feat/deepseek-v4-sm80`（对齐 vLLM 侧分支名）
- 目标：8×A100-80G，TP=8，bs=1，FP8 ckpt 载入，prefill ≥1000 tok/s，decode ≥50 tok/s
- 阶段一（本分支主体）：eager 正确性打通 —— 移植 A（fp8 MQA logits triton）+ B（sparse MLA triton，sink + extra_k_cache，元素级 stride）+ sm80 dispatch 门控 + 数值验证（kernel 单测 + "capital of France → Paris" smoke）
- 阶段二（条件触发）：仅当 eager decode <50 tok/s 时做 CUDA graph 优化；piecewise cudagraph 明确不做（vLLM 侧验证损坏）
- 跳过：marlin-experts MoE（用 sglang 现成 `SGLANG_DSV4_FP4_DEQUANT=1` → triton MoE）；wo_a bf16 保护（sglang 已解决）
- 启动基线：§2.7 命令，`--context-length 49152` 起，57k+ 交叉验证后放开
