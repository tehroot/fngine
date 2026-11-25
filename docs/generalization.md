# Generalization to other models

fngine and ingine are each made for one model. This document divides their code into three layers by how easily
another model can use it, and gives the plan for the shared engine in this fork and for the model repositories.
Paths with the prefix `ingine:` are in the ingine repository at commit `556acc7` (see [README.md](README.md)).

## Summary

About one third of each engine is general infrastructure. The rest is kernels and model code with the model's
dimensions compiled in. A general version is possible, but most of the speed comes from that specialization (see
[optimizations.md](optimizations.md)). The general version must keep the specialization, with the shapes as
template parameters, not as values read at run time.

## Code size by area (lines)

Counts are from 2026-11-24 (`wc -l` on `.c*` and `.h` files under `src`). Layer numbers are defined in the next
section. "Mixed" areas hold code of more than one layer.

### fngine (this repository, `src`)

| Area | Lines | Layer |
|---|---:|---|
| `core` (device, arena, graphs, PDL, host memory, exchange) | 2,398 | 1 |
| `text` (tokenizer, Jinja, chat template, tool-call parser, prompt lookup) | 4,126 | 1 |
| `serve` (HTTP, OpenAI API) | 3,717 | 1 |
| `engine` (engine, session, speculation controller) | 1,645 | mixed 1 and 3 |
| `io` (pack format, direct I/O, n-gram store) | 1,933 | mixed 1 and 3 |
| `cpu` (CPU expert tier, IQ decode, AVX-512) | 1,813 | 2 |
| `kernels/common`, `linear`, `moe`, `head` | 8,315 | 2 |
| `kernels/gdn`, `qsa`, `gr`, `ple`, `mtp` | 5,264 | 3 |
| `model` (weights, forward, placement, step graph, runtime) | 5,510 | mixed 2 and 3 |
| **Total** | **34,721** | Layer 1: about 30% (`core` + `text` + `serve`) |

### ingine (`ingine:native/src`)

| Area | Lines | Layer |
|---|---:|---|
| `core` (device, arena, graphs, PDL, safetensors) | 2,639 | 1 |
| `text` (tokenizer, Jinja, chat format, stream parser) | 2,540 | 1 |
| `serve` (HTTP, OpenAI API, SSE) | 1,213 | 1 |
| `engine` (scheduler, cancellation) | 612 | 1 |
| `image` (image decode, CPU) | 419 | 1 |
| `kernels/*` (dense, moe, attn, head, prefill, draft, vision, common) | 9,142 | 2 and 3 |
| `model` (weights, state, decode step, prefill, drafter, speculative loop) | 4,539 | 3, some 2 |
| **Total** | **21,104** | Layer 1: about 35% |

## The three layers

| Layer | Content | Effort to share |
|---|---|---|
| **1. General already** | Device and memory management, CUDA graphs, PDL helpers, tokenizer, Jinja chat templates, tool-call parsing, OpenAI HTTP server, scheduler, image decode | Low. ingine already took this layer from fngine with a namespace change. Extract it once as a shared library in this fork. |
| **2. Can be parameterized** | Multi-row GEMV and projection kernels (fngine `src/kernels/linear`, ingine `kernels/dense`); MoE up/down per quantization format (GGUF IQ, NVFP4, MXFP8); flash-decode attention (GQA, sliding window, sink logits); RMS norm; router top-k; sampling and the head; the structure of the speculative loop; prefix checkpoints; the CPU expert tier and the expert placement (for models larger than the GPU); the architecture layer (`sm_80` and `sm_120a` code paths) | Medium. Make the dimensions and the weight formats template parameters, and instantiate them per model. fngine's per-shape files (`src/kernels/linear/fp8/shapes/`) already show this pattern. The compiler still sees fixed shapes, so the speed stays. |
| **3. Specific to one architecture** | fngine (Qwen3.8-Flash-Next): GDN (gated delta net), QSA, PLE and GR kernels, the MTP head, the n-gram store. ingine (Inkling-2-Mini): the loop visit plan (layers 6-17 three times), message gates, the residual mix over block states, relative-position logits, the dynamic short convolution, the DSpark drafter. | High. Each new architecture needs its own kernels here. |

## Target structure

This fork holds layers 1 and 2. Each model repository holds layer 3 for one model or model family and depends on
this fork.

```
fngine fork (this repository)
  runtime/             layer 1: shared runtime (core, text, serve, engine, image)
  kernels/             layer 2: templated kernel library
    arch/                sm_80 and sm_120a paths (PDL, clusters, FP4/FP8 decode); see sm80-port.md
    gemv/ moe/ attn/ norm/ router/ head/ prefill/ cpu/
  models/qwen3.8-flash-next/   fngine's current model, moved to layer 3 form as the first example

<model> repository (one per model or model family)
  weights.cpp          checkpoint names -> tensors
  step.cpp             the decode step graph (kernel sequence per layer)
  kernels/             kernels that only this architecture needs
  tests/               goldens from the reference implementation
```

The directory names are a proposal. The first refactor decides the final names.

For a standard transformer or MoE (Llama-type, dense Qwen-type, Mixtral-type), most of the work is in layers 1
and 2. The work for a new model of that type then drops from weeks to days. For an unusual architecture such as
Inkling-2 or Qwen3.8-Flash-Next, layer 3 stays large.

## Scope

This structure is in principle what llama.cpp is, but with shape-specialized fused kernels in place of general
operations. Keep the scope narrow:

- one user, batch 1 (a verify width of a few tokens);
- the lab's two GPU types (`sm_120a` RTX 5090 and `sm_80` A100-80GB class);
- shapes as template parameters.

That scope is where the gain over vLLM and llama.cpp is. For batched work (for example embedding a document
corpus), use a batching engine such as vLLM.

## Next steps

1. **Measure baselines.** For each target model, measure llama.cpp and vLLM at batch 1 on the target card. Compare
   the decode speed with the bandwidth limit: active weight bytes per token ÷ DRAM bandwidth. The gap is the gain
   available for that model.
2. **Extract layer 1** in this fork as a shared library, with no change to model code or kernels. fngine must
   still pass its tests after the change.
3. **Build layer 2** from fngine's and ingine's kernels, with the `sm_80` architecture layer
   ([sm80-port.md](sm80-port.md)).
4. **Create the first model repository** and test layers 1 and 2 on it.

## Open questions

- Which models come next? Their architectures decide which kernel families to build first.
- Does a model need multi-GPU? The current engines have no tensor parallelism. At batch 1 it adds about two
  all-reduces per layer, so use it only for a model that does not fit on one card.
- Does ingine move onto the shared library too, or does it stay as it is?
