# Port to sm_80 (GA100, A100-80GB class)

fngine and ingine build only for `sm_120a` (RTX 5090 class). This document lists what prevents a build for `sm_80`,
what the port must change, and the order of the work. **Status: assessment only. No code is changed.** Paths with
the prefix `ingine:` are in the ingine repository at commit `556acc7` (see [README.md](README.md)).

Target: GA100 cards that operate as an A100 80 GB: about 80 GB of HBM2e, 108 SMs, NVLink, PCIe Gen3 or Gen4.
Verify these values on the real cards before you tune (step 1 below).

## The build gate

`CMakeLists.txt` accepts only `CMAKE_CUDA_ARCHITECTURES=120a` and stops on any other value. ingine has the same
gate in `ingine:native/CMakeLists.txt` and `ingine:native/CMakePresets.json`.

## Instructions that `sm_80` does not have

### fngine (this repository)

| Feature | Minimum arch | Where fngine uses it | Replacement on `sm_80` |
|---|---|---|---|
| FP8 tensor-core MMA (`mma.sync.aligned.kind::f8f6f4.m16n8k32...e4m3`) | `sm_120a` | `mma_fp8_e4m3` in `src/kernels/common/mma.cuh`, called by the W8A8 path `src/kernels/linear/fp8/fp8_a8_mma.cuh` | Use the W8A16 paths (`fp8_a16_gemm_mma.cuh`, `fp8_a16_ksplit_mma.cuh`): they decode FP8 and use BF16 `mma.sync`, which `sm_80` has. |
| Block-scaled FP4 MMA (`kind::mxf4nvf4.block_scale`) | `sm_120a` | `mma_nvfp4_e4m3` in `src/kernels/common/mma.cuh`; no caller found | Put it behind a guard or remove it. |
| `cvt.rn.satfinite.e2m1x2.f32` (FP4 encode) | `sm_100a` / `sm_120a` | `src/kernels/qsa/nvfp4_group16_codec.cuh` | Encode with integer operations. |
| FP8 E4M3 conversions (`__nv_fp8_*`, `cvt` with `e4m3`) | `sm_89` | `src/kernels/linear/fp8/*`, `src/kernels/qsa/*` (K8V8 cache, attention), `src/kernels/gr/gr.cu` | On `sm_80`, `cuda_fp8.h` compiles to a slow software conversion. Use a shift-based E4M3 → FP16 decode. |
| PDL (`griddepcontrol`, programmatic stream serialization) | `sm_90` | `src/core/pdl.cuh`, `src/kernels/common/pdl.cuh`; `qsa/*`, `gdn/*`, `moe/moe_decode.cu`, `linear/bf16/bf16_gemv.cuh` | Put it behind `__CUDA_ARCH__` guards. Keep the CUDA graphs. The overlap between kernels is lost. |
| Thread-block clusters and DSMEM | `sm_90` | `gr_norm_down` in `src/kernels/gr/gr.cu` | Reduce through global memory, or use one CTA per token. |
| `cp.async.bulk.prefetch.L2` | `sm_90` | `src/model/l2_prefetch.cu` | Remove it, or prefetch with ordinary loads with an L2 hint. |
| `mbarrier` with `expect_tx` | `sm_90` | `src/kernels/common/mbarrier.cuh`; no caller found | Put it behind a guard or remove it. |

The CPU expert tier (`src/cpu`) is host code, not GPU code. It needs AVX-512 with VNNI, BF16 and VBMI (Zen 4/5 or
Sapphire Rapids and later). Check the CPU of the target host.

### ingine

| Feature | Minimum arch | Where ingine uses it | Replacement on `sm_80` |
|---|---|---|---|
| `cvt.rn.f16x2.e2m1x2` (hardware FP4 decode) | `sm_100a` / `sm_120a` | `ingine:native/src/kernels/moe/moe.cu`, `moe/dequant.cu` (NVFP4 experts, layers 0-19) | Decode with bit operations (LOP3/PRMT, as in Marlin) or a lookup table. `ingine:native/src/kernels/prefill/wgemm.cu` already uses a table (`kE2M1`). |
| FP8 E4M3 conversions (`cvt.rn.bf16x2.e4m3x2`, `cvt.rn.f16x2.e4m3x2`, `__nv_fp8_*`) | `sm_89` | MXFP8 experts (layers 20-23), NVFP4 block scales, `--dense-fp8`, `--kv-fp8`, the drafter's E4M3 ring | Use a shift-based E4M3 → FP16 decode. Or use BF16 where memory permits (see "Memory"). |
| PDL | `sm_90` | `ingine:native/src/core/pdl.cuh`, `kernels/common/pdl.cuh`, `pdl_wait_fenced` in `kernels/dense/cluster.cuh`, most decode kernels | Put it behind `__CUDA_ARCH__` guards. Keep the CUDA graphs. |
| Thread-block clusters, DSMEM, `st.async`, `mbarrier.arrive.expect_tx`, `mapa`, `fence.mbarrier_init` | `sm_90` | `ingine:native/src/kernels/dense/dsmem.cuh`, `cluster.cuh`, `router.cu` (cluster size 16), `gates.cu`, `mix.cu`, `gemv.cu`, `attn/attn.cu` | **The largest task.** Exchange the data through global memory, or split each kernel into two launches. This adds latency on each of the 48 layer visits per token. |

## Features that work on `sm_80` with no change

- `mma.sync.aligned.m16n8k16` with BF16 and F16 inputs, `m16n8k8` TF32, `m16n8k32` S8, and `ldmatrix`.
- `cp.async.ca` and `cp.async.cg`.
- L2 eviction hints (`createpolicy`, `L2::evict_first`) and `redux.sync`.
- cuBLASLt BF16 GEMMs (ingine prefill, drafter, vision tower).
- Dynamic shared memory: ingine uses at most 101,376 bytes per block; `sm_80` permits up to 163 KB per block.
- CUDA 13.4: CUDA 13 supports `sm_80`.

## Hardware differences

| Item | RTX 5090 | A100 80 GB class | Effect on the port |
|---|---|---|---|
| DRAM bandwidth | about 1.79 TB/s | about 1.9-2.0 TB/s | The decode ceiling is about equal, or up to about 10% higher. |
| SMs | 170 | 108 | Tune the grid sizes and the persistent grids again. |
| L2 | 96 MB | 40 MB | Small weights possibly do not stay in L2. Measure the L2 policy and fngine's L2 prefetch again. |
| Shared memory per SM | about 100 KB | 164 KB | The current kernels fit. |
| FP32 / BF16 tensor | high / high | 19.5 / 312 TFLOPS | A software FP4/FP8 decode is affordable for small row counts. |
| Memory | 32 GB | about 80 GB | See "Memory". |

### Speed expectation

With equal kernel efficiency, the decode ceiling scales with bandwidth. Without PDL and clusters, each layer has
more launch and synchronization gaps, so the first `sm_80` build will be slower than that ceiling until it is
tuned. There is no reliable tokens/s estimate before the kernels run on the card.

### Memory

- **ingine:** the model (20.2 GB), a BF16 KV cache at 262,144 tokens (8 GiB for the 8 full-attention visits) and
  the park buffers fit easily in about 80 GB. The FP8 options were for capacity on 32 GB. But FP8 dense weights
  also halve the bytes read per step, so they still help decode speed if the decode cost on `sm_80` is low. Measure
  both.
- **fngine:** on an RTX 5090, about half of the 24,576 routed experts stay in host RAM. With two cards of about
  80 GB, the full expert set can possibly stay in VRAM, and the CPU tier and the PCIe streaming become unnecessary.
  The pack size is not checked yet. Expert placement over two cards is a redesign of `src/model/placement.*`, not
  only a port.

### Multi-GPU

Neither engine has tensor parallelism. At batch 1, tensor parallelism adds about two all-reduces per layer and
makes decode slower. For a model that fits on one card, run one independent server per card. For fngine's full
expert set over two cards, prefer a split of experts between the cards (expert parallelism) over a split of each
matrix.

## Validation

- The CPU oracles of the kernels apply with no change.
- Bit-lock hashes (`ingine:native/tests/kernels/test_bitlock.cpp`) will change. A new FP4/FP8 decode or a split
  kernel changes the reduction order. Record the hashes per architecture.
- Model-level gates: `apps/validate` here (engineering notes, "Correctness contracts") and `ingine-validate`.
- GA100 and other GPUs give different BF16 rounding results. Do not expect bit parity between architectures.

## Order of work

1. **Verify the hardware.** On one card: SM count, clocks, DRAM bandwidth over the full HBM, NVLink link state,
   PCIe generation and width. Also measure FP32, HFMA2 and BF16 `mma.sync` throughput, and the throughput of a
   LOP3-based FP4/FP8 decode.
2. **Add the architecture layer.** Let the CMake gate accept `80`. Put `__CUDA_ARCH__` guards with `sm_80` code
   paths on PDL, clusters, the FP8/FP4 MMA and the FP4/FP8 conversions. This step can be done without the GPU;
   compile and test on the GPU. In this fork, the architecture layer is part of the shared kernel library
   ([generalization.md](generalization.md)).
3. **Port the expert and linear kernels.** fngine: select the W8A16 paths, port the IQ expert decode if needed.
   ingine: `moe_up`, `moe_down`, prefill `wgemm`, `dequant`. Check them against the oracles.
4. **Rewrite the cluster kernels.** fngine: `gr_norm_down`. ingine: router, `message_gates`, `mix`, `gemv`, the
   attention exchange.
5. **Validate and tune.** Run the validators. Tune the grid sizes for 108 SMs.

**Shorter first build for ingine:** serve with BF16 dense weights and a BF16 KV cache (no `--dense-fp8`, no
`--kv-fp8`). Then only the expert decode needs new FP4/FP8 code before the first end-to-end run. Steps 2 and 4 are
still necessary.
