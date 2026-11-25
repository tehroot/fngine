# What fngine and ingine optimize

fngine and ingine are inference engines that are made for one model, one GPU and one user at a time. fngine runs
Qwen3.8-Flash-Next. ingine runs Inkling-2-Mini and took its infrastructure from fngine. This document explains the
techniques that give their speed, and how they differ from vLLM and llama.cpp. Paths with the prefix `ingine:` are
in the ingine repository at commit `556acc7` (see [README.md](README.md)).

## Target workload

- **One request at a time** (batch 1). The scheduler keeps one active request.
- **Small calls.** A decode call has 1 token. A speculative verify call has a few tokens (ingine: at most 8).
- **Bandwidth limit.** At batch 1, a decode step reads all active weights once and does little arithmetic per byte.
  DRAM bandwidth sets the speed, not compute.

The design goal follows from these points. Each decode step must read each weight byte once, at more than 80% of
DRAM bandwidth, with almost no idle time between kernels. `ingine:native/KERNELS.md` states this target per
kernel, for example "≥ 80% of DRAM bandwidth on the 8 experts' bytes per token" for `moe_down`.

## Techniques

| Technique | What it does | fngine (this repository) | ingine |
|---|---|---|---|
| Kernels for exact shapes | GEMV and MoE kernels for small row counts. The model dimensions are compile-time constants, so the compiler fully unrolls the loops. | One file per weight shape: `src/kernels/linear/fp8/shapes/n*_k*.cu` | `constexpr int H = 2304`, `I = 1152` in `ingine:native/src/kernels/moe/moe.cu` |
| Fused operations | Norms, convolutions, residual updates and gating run in the kernel that does the matrix product. Intermediates do not go to DRAM. | `gr_norm_down` (4-stream RMS norm + down projection), `src/kernels/gr/gr.cu` | `mix`, `proj_conv`, `moe_up`, `moe_down` (`ingine:native/KERNELS.md`) |
| One CUDA graph per step | The full decode step is one graph per call width. The host launches one graph, not hundreds of kernels. | `src/model/step_graph.cpp`, `src/core/decode_graph.*` | `ingine:native/src/model/decoder.cpp`, `runtime.cpp` |
| Programmatic dependent launch (PDL) | A kernel loads its weights before its predecessor finishes, then waits (`griddepcontrol.wait`). This hides the gap between kernels. Needs `sm_90` or later. | `src/core/pdl.cuh`, `src/kernels/common/pdl.cuh`; GDN, QSA, MoE decode and BF16 GEMV kernels | `ingine:native/src/core/pdl.cuh`; `pdl_wait_fenced` in `kernels/dense/cluster.cuh` |
| Thread-block clusters and DSMEM | Small reductions exchange data between SMs through distributed shared memory. ingine measured about 0.25 µs with `st.async` and an mbarrier, against about 0.55 µs for a cluster barrier. Needs `sm_90` or later. | `src/kernels/gr/gr.cu` (clusters of 4 CTAs) | `ingine:native/src/kernels/dense/dsmem.cuh`, router (cluster size 16), `gates.cu`, `mix.cu`, `gemv.cu`, `attn/attn.cu` |
| Auxiliary stream | Independent work (GDN gating, QSA indexer, the shared expert) runs on a second stream at a lower priority. | `src/model/forward.cpp` (see engineering notes, second pass) | not used |
| Native quantized formats | Expert weights stay in the checkpoint format and are decoded in registers. No conversion pass. | GGUF IQ formats (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL): `src/kernels/moe/iq_device.cuh` | NVFP4 (layers 0-19) and MXFP8 (layers 20-23) |
| FP8 dense weights | E4M3 rows with one FP32 scale per row. Halves the bytes of dense weights per step. | `src/kernels/linear/fp8/`: W8A16 paths, and a W8A8 path with FP8 tensor-core MMA (`fp8_a8_mma.cuh`) | `--dense-fp8` |
| FP8 KV cache | E4M3 keys and values. Long context fits in 32 GB. | K8V8: `src/kernels/qsa/fp8_e4m3_row_codec.cuh`, `kv_append.cu` | `--kv-fp8` on full-attention layers |
| L2 control | Weights read once per step are loaded evict-first. fngine also prefetches the next layer's weights into L2. | `src/model/l2_prefetch.cu` (`cp.async.bulk.prefetch`, measured 9.47 → 8.92 ms per step) | `ingine:native/src/kernels/common/l2.cuh` (+1-2% end to end) |
| The model's own drafter | Speculative decoding with the model's drafter. Drafts are verified exactly, so greedy output matches plain decoding. | MTP head (`src/kernels/mtp`), prompt lookup (`src/text/prompt_lookup.*`), controller (`src/engine/speculation.*`) | DSpark drafter; the GPU chooses the verify width and the host synchronizes once per round |
| Prefix reuse | Reuse of the state of an earlier prompt. An agent loop that sends the previous answer again prefills only the new tokens. | Session snapshots: `src/engine/session.*` | Checkpoints and parked conversations: `ingine:native/src/model/generator.cpp` |
| Model larger than the GPU | Adaptive placement of experts between VRAM and host RAM, an AVX-512 CPU tier for cold experts, PCIe streaming of experts for long prefills, an n-gram table on NVMe. | `src/model/placement.*`, `adaptive_experts.*`, `prefill_stream.*`; `src/cpu/expert_pool.*`, `iq_cpu.*`; `src/io/ngram_store.*` | not used (the model fits) |
| Correctness contracts | CPU oracles per kernel, model goldens from the reference, state equivalence for speculation. | `tests/`, `apps/validate`; engineering notes, "Correctness contracts" | `ingine:native/tests` (FP64 oracles, bit-lock hashes), `ingine-validate` |

## Comparison with vLLM and llama.cpp

| | vLLM | llama.cpp / llama-server | fngine / ingine |
|---|---|---|---|
| Design goal | Throughput for many users | Portability, many quantization formats, CPU+GPU | Latency for one user, on one model and one GPU |
| Batching | Continuous batching, PagedAttention | A small number of parallel slots | None (one active request) |
| Kernels | Libraries for general shapes (FlashAttention, FlashInfer, Marlin, CUTLASS) | General GGML operations (for example `mmvq` for quantized GEMV) | Hand-fused kernels, one set per model, shapes fixed at compile time |
| Launch cost | `torch.compile` and CUDA graphs; Python on the host | One graph node per operation; CUDA graphs available | One graph per step, plus PDL |
| New model | A Python model file; often available on the day of release | A C++ graph builder and a GGUF converter | Weeks of kernel work |
| Multi-GPU | Tensor and pipeline parallelism | Split by layer or by row | None |
| Offload to host RAM | Limited | Per layer or per tensor (`-ot`) | fngine: adaptive, per expert |

### Where the batch-1 gain comes from

At batch 1, vLLM and llama.cpp lose time in three places:

1. **General kernels.** A kernel made for all shapes does not reach the bandwidth limit at one row.
2. **Launch gaps.** Many small kernels run one after the other. Each boundary leaves the GPU partly idle.
3. **Unfused intermediates.** Each operation writes its output to DRAM, and the next operation reads it again.

fngine and ingine remove these three costs. At high batch, vLLM is faster, because fngine and ingine do not batch
requests.

**Caution:** neither repository records a vLLM or llama.cpp baseline on the same hardware and model. The size of
the gain is not known from data. Measure it before you use it to make a decision (see
[generalization.md](generalization.md), "Next steps").
