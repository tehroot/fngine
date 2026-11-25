# Engineering documentation

Start here. This folder is the starting point for maintainers and coding agents.

This fork of fngine is the base of a shared inference engine. The work goes up the chain: fngine is the origin of
ingine's infrastructure, so the shared runtime and the shared kernel library are built here. Work for each model
goes in separate repositories that use this one.

The documents use ASD-STE100 Simplified Technical English. Keep that style when you add to them.

## Documents

| Document | Contents |
|---|---|
| [engineering-notes.md](engineering-notes.md) | fngine's full build, validation and performance record (Qwen3.8-Flash-Next on one RTX 5090) |
| [optimizations.md](optimizations.md) | What fngine and ingine optimize, and how they differ from vLLM and llama.cpp |
| [generalization.md](generalization.md) | Which code is general, which code can be parameterized, which code is specific to one model; the plan for the shared engine and the model repositories |
| [sm80-port.md](sm80-port.md) | Port from `sm_120a` (RTX 5090) to `sm_80` (GA100, A100-80GB class): instruction gaps in fngine and ingine, hardware differences, order of work |

## Repositories

| Repository | Role |
|---|---|
| [tehroot/fngine](https://github.com/tehroot/fngine) (this fork, `origin`) | Shared engine work. Push here. |
| [satellitedown/fngine](https://github.com/satellitedown/fngine) (`upstream`) | Original fngine. Fetch only; push is disabled in the local clone. |
| [satellitedown/inkling-2-mini-abliterated-ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/tree/556acc7cf42749536684854a6df820e41258f929) | ingine, the engine for Inkling-2-Mini. It took its infrastructure from fngine. Its decode kernels are the main source for the shared kernel library. |
| Model repositories | Not created yet. One per model or model family (see [generalization.md](generalization.md)). |

**Path convention.** A path with the prefix `ingine:` is in the ingine repository at commit `556acc7`, for example
`ingine:native/KERNELS.md`. A path with no prefix is in this repository.

## Status (2026-11-25)

| Item | State |
|---|---|
| fngine on RTX 5090 (`sm_120a`) | Works. 180-210 tokens/s coding-chat decode (see `../README.md`). |
| ingine on RTX 5090 (`sm_120a`) | Released. 490-560 tokens/s decode. |
| Analysis of the optimizations and of generalization | Written (this folder). |
| `sm_80` port | Not started. Assessment only: [sm80-port.md](sm80-port.md). |
| Baselines of vLLM and llama.cpp on the same hardware | Not measured. Neither repository records one. |
| Shared runtime library (layer 1) | Not started. |
| Shared kernel library (layer 2) | Not started. |
| Model repositories | Not created. The list of target models is open. |

## Rules for changes

- Do not push to `upstream`. Work on branches in this fork.
- fngine's correctness contracts are in [engineering-notes.md](engineering-notes.md), "Correctness contracts".
  ingine's kernel contracts are in `ingine:native/KERNELS.md`.
- A kernel change keeps its rounding points and its bit-lock hashes, unless a change is measured against the
  goldens. A port to a new architecture is the exception: record the hashes per architecture and explain them.
- Do not put host names, IP addresses or other private lab details in this repository. It is public.
- Update the status table above when the state of an item changes.
