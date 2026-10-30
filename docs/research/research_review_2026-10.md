# GRADIENT: Research Review & Roadmap (October 2026)

*Scope: the research ideas in `docs/whitepaper.md` (features 1–14) and the "Future Research" notes in `docs/dev-notes/plan_phases_v3.md`, checked against the code at commit `afc072c` (before the v0.2.0 upgrade), the diffusers 0.42.1 source, and the 2025–2026 literature. The status table after the summary lists what the v0.2.0 upgrade has already addressed.*

**Legend.** Priority: P0 = next milestone, P1 = within two milestones, P2 = later or opportunistic. Effort for one developer: S ≤ 1 week, M = 2–4 weeks, L > 1 month.

---

## 1. Executive summary

GRADIENT's thesis still holds, and few tools serve it: a local **lab for diffusion models**, not another generation UI. Three facts shape the roadmap:

1. **The research prototypes don't measure what they claim.** The sensitivity map comes out as a mathematical constant. The "latent space" map is a UMAP of i.i.d. Gaussian noise (details in §2).
2. **The ecosystem has changed architecture.** Current open models are MMDiT or single-stream DiTs trained with flow matching and using LLM/VLM text encoders: SD3.5, FLUX.1/FLUX.2, Qwen-Image, Z-Image. This makes some ideas obsolete (ELLA, FreeU-style UNet weighting, arithmetic on VAE latents) and others much easier.
3. **diffusers now ships much of the plumbing** GRADIENT planned to write: 11 guiders, step callbacks that can swap embeddings, model hooks, layer-skip, six caching methods, an attention-backend dispatcher, group offloading and ten quantization backends. The project's differentiation therefore has to come from **measurement, attribution and experiment-as-data**, not from exposing knobs. SD.Next already exposes the guider and caching stacks.

**Recommendation:** rebuild the core as an *instrumented, architecture-agnostic sampler* on diffusers ≥ 0.42.1. Fix measurement next (sensitivity and an output-space atlas). Then add steering and interpretability (modulation timelines, ConceptAttention, attribute directions), and add the agent last.

---

## Status after the v0.2.0 upgrade (Oct 2026)

| Finding | Status in v0.2.0 |
|---|---|
| Stack (diffusers ^0.33, torch ~2.5, `pkg_resources` import broke startup) | **Fixed**: diffusers ≥ 0.42.1, torch ≥ 2.14 (2.15 default), transformers 5, `dtype` API, Python ≥ 3.11 |
| SD1.5-only hand-written loop | **Fixed**: any diffusers family via the pipeline's own `__call__` + `callback_on_step_end` hooks (SDXL, SD3.5, FLUX.1/2, Qwen-Image, Z-Image, Chroma, Sana, Krea 2 ...) |
| Logged batch seeds not reproducible | **Fixed**: one seeded generator per image; image *i* regenerates bit-identically from its logged seed (tested) |
| Scheduled CFG silently fell back to 7.5 | **Fixed**: `linear()`, `cosine()`, keyframes and `[step, value]` lists, applied per step on classic-CFG families; metadata records whether it was applied |
| `custom_sigmas` values ignored | **Fixed**: passed to `set_timesteps(sigmas=...)`; clear error for schedulers that don't accept sigmas |
| ResourceManager eviction crash / no unloading | **Fixed**: LRU pipeline cache with real unload callbacks; `memory_mode` (gpu / balanced / low / minimal / auto) |
| Sensitivity map constant (§2.1) | **Replaced** with central finite differences on shared noise (step 1 of §5 item 2); JVP and step-localised curves remain open |
| Latent-space map of i.i.d. noise (§2.2) | **Open**: endpoint kept as scaffolding and labelled as a prototype; redesign per §5 item 11 |
| Blocking GPU work on the event loop | **Fixed**: GPU work runs in a worker thread behind a single GPU lock (first step toward the task queue) |
| Reproducibility metadata (hashes, versions, approximation flag) | **Partly**: pipeline class, family, dtype, memory mode, scheduler and applied flags are logged; hashes and versions remain open |

### Implemented in v0.3.0–v0.4.0 (Lab tab, `/api/v1/lab/*`)

| Roadmap item | Status |
|---|---|
| Exact JVP sensitivity (§5 item 2, step 2) | **Done for SD 1.x/SDXL**: forward-mode AD (`torch.autograd.forward_ad` dual numbers) through the full CFG sampler, UNet and VAE under math-mode SDPA; matches finite differences (cosine > 0.999 in tests). Drives the **Taylor preview** (instant CFG slider, checked against a real render). Other families use finite differences; JVP for flow-matching DiTs is open |
| σ-timeline structural modulation (M3) | **Done as lanes** on every request: prompt replace/blend, guidance, attention-layer output scale, LoRA weight, each over a step-fraction window, for any family exposing step callbacks. Open: lanes on arbitrary module inputs, per-token/region masks, continuous curves instead of windows |
| Cross-attention attribution | **Done for UNet models** (v0.3.0): recording processor computes explicit softmax probabilities only on selected layers and steps |
| ConceptAttention for joint-attention DiTs (§5 item 4) | **Done for FLUX.1, FLUX.2, SD3/3.5** (v0.4.0): concept tokens run through the double-stream blocks via read-only forward hooks (images bit-identical); output-space maps, softmax over concepts, averaged over blocks/steps; segmentation view. A test verifies that the concept stream reproduces the model's own text-stream attention outputs when fed the prompt tokens. Open: single-stream blocks, Qwen-Image/Z-Image/Chroma, real-image mode (encode + noise to σ), segmentation benchmarks |
| Token ablation scored in feature space | **Done, model-agnostic**: leave-one-word-out (word removal, or embedding masking on SD/SDXL) with DINOv3 ViT-S/16 patch/CLS distances, DINOv2-small fallback. Open: batched ablation, region-restricted scores, comparison against attention maps on a prompt set |

## 2. Code reality check (at `afc072c`)

### 2.1 `ai/viz/sensitivity.py`: confirmed flawed (worse than suspected)

- The UNet call is wrapped in `torch.no_grad()`, and on every step `latents_grad = latents_prev.detach().requires_grad_(True)` (line 110). Both cut the graph between `guidance_scale_tensor` and the result.
- `final_latent` is therefore a **leaf** tensor, and `loss = final_latent.abs().mean()`. The gradient of a mean of absolute values with respect to its own leaf input is `sign(x)/N`. So the "sensitivity map" is a **uniform constant of 1/N everywhere** (about 6.1×10⁻⁵ for a 4×64×64 latent). Its min, max and mean are equal and its std is 0. `guidance_scale_tensor.grad` is never populated.
- Even without the `detach`, `no_grad` would remove the dependence that runs through ε(x_t). Smaller issues: it works only in latent space (no decode), uses the deprecated `_encode_prompt`, computes fp16 gradients, leaves the heatmap as a TODO, and isn't wired into the UI.

### 2.2 `ai/viz/latent_space.py`: confirmed flawed

- It draws `np.random.randn(N, 4, 64, 64)`, flattens each sample to 16,384 dimensions and runs UMAP with a cosine metric. The draw is unseeded (`request.seed` only seeds UMAP), so runs aren't reproducible.
- I.i.d. isotropic Gaussian samples have no manifold. In high dimensions their pairwise distances concentrate, so the kNN graph is effectively random and the 2-D layout is an artifact of UMAP initialisation.
- The "previews" are VAE decodes of raw noise divided by 0.18215 (std ≈ 5.5). That produces saturated texture, not images. The code also ignores `projection_method` and never produces `z`, because `n_components` is fixed at 2.
- The deeper problem: initial noise carries meaning **only through the sampler**. A map has to be built from outputs or from internal features (§5, item 11).

### 2.3 Other bugs that affect research validity

- **Logged seeds are wrong for batches.** One generator draws all `num_outputs` latents, but image *i* is logged as `seed+i`, and re-running `seed+i` doesn't reproduce it. diffusers accepts a list of per-image generators; use that.
- **Scheduled CFG.** The parser silently falls back to 7.5 (so the logs misstate what ran). If a scheduled value were ≤ 1, the doubled CFG batch would reach `scheduler.step` without being split.
- **`custom_sigmas`** sets only the step count. The values themselves are ignored but still written to metadata as if they had been applied.
- **`ResourceManager.ensure_available_vram`** calls `release(lru_id, vram_freed)`, but `release` takes one argument. The first eviction raises `TypeError`, and nothing is ever actually unloaded.
- **Stack.** `pyproject.toml` pins diffusers ^0.33.1 and torch ~2.5. diffusers 0.39 raised the minimum to torch 2.6, 0.40 renamed `torch_dtype` to `dtype`, 0.42 requires Python ≥ 3.11, and **0.42.1 patches a `trust_remote_code` bypass**. Pin ≥ 0.42.1.
- **Reproducibility metadata** lacks model/LoRA file hashes, library versions, dtype, attention backend and an "approximation" flag (caches change outputs).

---

## 3. The architectural shift and what it means for each idea

| GRADIENT idea | SD1.5/SDXL assumption | MMDiT / flow-matching reality (2026) |
|---|---|---|
| UNet layer weighting | down/mid/up blocks, FreeU skip scaling | Stacks of double-stream and single-stream blocks. FreeU doesn't apply. The levers are block-index ranges, layer skip (SLG/STG), and Stable Flow's "vital layers". |
| Cross-attention maps | separate text cross-attention | **Joint attention**: text and image tokens share one softmax, and in single-stream DiTs (Z-Image) everything is a single sequence. Fused kernels never materialise the probabilities. ConceptAttention-style maps built in attention-output space are the current best practice. |
| ELLA | SD1.5 adapter on top of CLIP | Effectively obsolete: only an SD1.5 checkpoint was released, with the last update in 2024. LLM/VLM encoders are now native: Mistral-3 (FLUX.2 [dev]), Qwen3 (FLUX.2 [klein], Z-Image), Qwen2.5-VL (Qwen-Image). |
| Noise schedules / custom sigmas | Karras σ ∈ [0.03, 14.6], ε-prediction | Flow matching uses x_t = (1−σ)x₀ + σε with σ ∈ (0, 1], velocity prediction, a time `shift` and a resolution-dependent `mu`. FlowMatchEuler accepts explicit `sigmas`. Optimised schedules: AYS, OSS. |
| Guidance vectors | CFG with a fixed scale | Many checkpoints are **distilled**. FLUX.1/2 [dev] embed a guidance value. [klein] and Z-Image-Turbo are step-distilled and ignore CFG, so negative prompts need NAG. CFG variants: APG, CFG++, Rectified-CFG++, CFG-Zero*, TCFG, FDG, MAMBO-G, plus guidance intervals. |
| Concept sliders / attribute directions | text-embedding or LoRA sliders | FluxSpace (training-free, on attention outputs), SliderSpace (unsupervised LoRA directions), and SAE features on the residual stream. |
| Latent arithmetic | 4-channel SD latents | Arithmetic on VAE latents (16 channels for FLUX.1; TAEF2 uses 32 for FLUX.2) isn't semantic. Do arithmetic in embedding, attention-output or SAE space. |
| Prompt switching | A1111 `[from:to:when]` | Native: FLUX, FLUX.2, Qwen-Image and Z-Image all read `prompt_embeds` back from `callback_on_step_end` outputs (verified in source). Qwen-Image also needs the mask to match. Disable `TextKVCacheConfig` when switching. |

---

## 4. What diffusers 0.42 provides natively

| Need | Native facility (verified in the 0.42.1 source) |
|---|---|
| Guidance research | `diffusers.guiders`: CFG, CFG-Zero*, APG (+ APG-mix), AutoGuidance, PAG, SEG, SkipLayer/STG, Tangential CFG, FDG, MAMBO-G. Every guider has `start`/`stop`, which is a guidance interval. |
| Using guiders in an own loop | `guider.set_state(step, n, t)` then `guider.forward(pred_cond, pred_uncond)`. They work outside modular pipelines, so they also work on SD1.5. |
| Pipeline composition | Modular Diffusers left "experimental" in 0.40. Modular pipelines exist for SDXL, SD3, FLUX, FLUX.2, FLUX.3, Qwen-Image (2.1), Z-Image and Wan, but **not SD1.5**. Also `ComponentsManager` (auto CPU offload) and a Mellon node-UI bridge. |
| Per-step intervention | `callback_on_step_end` with `latents`/`prompt_embeds`; official CFG-cutoff callbacks |
| Model surgery | `ModelHook`/`HookRegistry`, `LayerSkipConfig` (skip attention, FF or attention scores per block) |
| Caching (approximate) | FirstBlockCache, FasterCache, PAB, TaylorSeer, MagCache (now covers FLUX.2), SeaCache; exact `TextKVCache` |
| Attention | `set_attention_backend()` and an `attention_backend` context: native SDPA, FA2/3/4 (Hub kernels), Sage, xFormers, Flex, `_native_math` |
| Memory | `enable_group_offload` (block/leaf level, CUDA streams, disk), `enable_layerwise_casting` (fp8 storage), model/sequential offload |
| Quantization | bitsandbytes, torchao, GGUF, Quanto, ModelOpt, Nunchaku, SDNQ, AutoRound, BFL backends; `PipelineQuantizationConfig` |
| Schedulers | FlowMatchEuler (sigmas, mu, shift, Karras/beta/exponential sigmas, stochastic), FlowMatchHeun, FlowMatchLCM |
| Gaps (GRADIENT's opportunity) | Per-block LoRA scales exist only for UNet models. There's no attention-map extraction API and no sensitivity or attribution tooling. |

---

## 5. Feature-by-feature assessment

**(1) Advanced XYZ grids & batching (P0, M).** *Status:* works, synchronous, logs to SQLite. *State of the art:* evaluation now relies on multi-seed statistics with VLM and preference metrics (VQAScore, HPSv3), not eyeballed grids. *Recommendation:* make grids **experiment specs** (declarative YAML/JSON). Add guider type, guidance interval, shift and LoRA block scales as axes. Use common random numbers (the same noise per cell) and N seeds per cell, compute a metric per cell, and run everything through the real task queue.

**(2) Parameter sensitivity (P0, M).** *Status:* broken (§2.1); v0.2.0 ships step 1 below. *Method:* the quantity of interest is ∂D(x₀(w))/∂w, where D is the VAE decode or a perceptual feature extractor (DINOv3 patches). Build it in three steps:
1. **Central finite differences** with common random numbers: w ± δ, run as one batched forward pass. This works for any parameter, including LoRA weight and shift, and is the whitepaper's "empirical sampling".
2. **Forward-mode JVP** (`torch.func.jvp`): w is a scalar, so a single tangent pass gives the exact ∂x₀/∂w. Flash SDPA lacks forward-mode AD, so run the JVP inside `attention_backend("_native_math")`.
3. **Step-localised sensitivity**: perturb w only on step k and plot sensitivity against σ. This shows *when* guidance matters, which links directly to the guidance-interval findings. Reverse-mode gradients with respect to many inputs (e.g. per-token embeddings) need per-step gradient checkpointing or adjoint methods (the Adjoint Matching lineage).

*New UX this enables:* a **Taylor preview**. Compute the tangent once, then let the user drag the slider and show x₀ + Δw·∂x₀/∂w decoded with TAESD/TAEF2. That delivers the whitepaper's "estimate before regeneration" goal literally.

**(3) Post-hoc semantic labeling (P2, M).** *Status:* absent. *Recommendation:* caption atlas clusters with a local VLM: llama.cpp's `llama-server` handles Qwen2.5-VL, Gemma 3/4 and Mistral Small 3.1 through libmtmd and an OpenAI-compatible API, while llama-cpp-python lags behind. Use SAM 3 (text-prompted concept segmentation) for region labels. Store labels in LanceDB alongside the embeddings.

**(4) Direct token influence (P1, M).** *Status:* absent. *State of the art:* ConceptAttention (ICML 2025) reuses DiT attention parameters, with no training, to build concept saliency maps. It outperforms CLIP-, DINO- and UNet-based interpretability methods on zero-shot segmentation. DAAM-style aggregation still works for SD/SDXL. *Recommendation:* implement a custom attention processor that computes probabilities only for selected layers and steps. Add **token ablation** (leave-one-out embedding, then DINOv3 distance) as a model-agnostic causal measure. Caveat: with LLM encoders, chat templates and system prompts add tokens, so map word spans to tokens explicitly.

**(5) Prompt-based highlighting (P1, S once item 11 exists).** In the atlas, colour points by their SigLIP 2 text–image similarity to a query. In the image view, overlay the ConceptAttention map for the query concept.

**(6) Dynamic prompt weaving / ELLA (P1, S–M).** *Status:* absent. *Recommendation:* **drop ELLA.** Implement weaving as σ-scheduled operations on `prompt_embeds` via `callback_on_step_end` or the own loop:
- switch, linear blend, or per-token-span blend;
- A1111 `[a:b:when]` syntax, with `when` expressed in σ rather than step index so it transfers across schedulers.

Pad embeddings to equal length (Qwen-Image masks) and disable text KV caching while weaving. For distilled models, record that guidance is embedded.

**(7) Structural diffusion modulation (P0 for guidance/sigmas, S; P1 for block-level, M).** *Status:* the CFG schedule parser is missing and custom sigmas are ignored. Map the whitepaper's levers to current methods:
- **Noise schedules:** pass real `sigmas`, `shift` and `mu`, and offer presets (AYS for SD/SDXL; OSS-style schedules for flow models).
- **Guidance vectors:** plug the guider zoo into the own loop. Add Rectified-CFG++ and CFG++ as custom guiders, and NAG for distilled models.
- **"UNet layer influence":** for UNets, FreeU plus per-block LoRA scales (native). For DiTs, `LayerSkipConfig`, SLG/PAG/SEG layer choices, attention-output scaling and **per-block LoRA scales for DiTs**. That last one is missing in diffusers and is a good GRADIENT contribution.

Present all of these as **modulation lanes on a shared σ-timeline**. That doubles as the reactive-art interface: audio or MIDI can drive lanes later.

**(8) Cross-model / cross-seed latent grafting (P2, M–L).** *Status:* absent.
- **Same model:** masked blends of two trajectories at a chosen σ, mid-trajectory swaps, and MultiDiffusion-style regional composition.
- **Cross-model:** possible directly only when two models share a VAE latent space (SDXL family; FLUX.2 [dev]/[klein]). Otherwise decode, re-encode and re-noise to σ (an img2img hand-off).
- **Editing-style grafting:** FlowEdit (inversion-free, ICCV 2025) and Stable Flow (attention injection in vital layers, CVPR 2025) are the modern primitives.

**(9) Agentic art director (P2, M; after milestone 2).** *State of the art:* T2I-Copilot (ICCV 2025) splits work across interpreter, generator and evaluator agents. ComfyMind (NeurIPS 2025) plans over ComfyUI workflows as a search tree with execution feedback. *GRADIENT's edge:* the experiment DB plus experiment specs give an agent a typed action space and memory. Use the agent to *design sweeps and test hypotheses* ("does APG reduce oversaturation at w = 9 across 32 seeds?"), with a VLM or HPSv3 judge. Expose GRADIENT as an MCP server so external agents can drive it.

**(10) Live diffusion visualization (P0, S).** Stream predicted x₀ (not noisy x_t) per step via TAESD/TAEF1/TAEF2. TAEF2 decodes FLUX.2 latents but isn't integrated into diffusers yet, so it needs a small wrapper. Add per-step overlays for the guidance delta (cond − uncond) and the update magnitude. This is cheap and immediately differentiating.

**(11) Interactive latent space navigator (P0 redesign, M–L).** Replace the noise UMAP with three layers:
1. **Output atlas.** Embed every generated image with SigLIP 2 (text-aligned) and DINOv3 (visual structure). Store vectors and full parameters in LanceDB, project with UMAP, and render with Apple's Embedding Atlas (MIT licence, ships a Svelte component, WebGPU, clustering and auto-labels). Clicking a point restores its exact parameters.
2. **Internal-feature views.** h-space for UNets; joint-attention outputs (FluxSpace) and SAE features (Shabalin et al. 2025 on FLUX.1) for DiTs.
3. **Generative navigation.** Seed slerp, embedding interpolation, and attribute directions: FluxSpace edits, SliderSpace or Concept-Slider LoRAs, and LOCO-Edit, where the singular vectors of the x₀-predictor Jacobian give unsupervised local edit directions computable with JVP/VJP power iteration. "Latent painting" = applying a direction under a mask.

**(12, 13) Advanced prompt management (P2, S).** Presets exist. Add prompt versioning with diffs linked to runs, a wildcard grammar, LLM rewriting (FLUX.2 has native caption upsampling) and image-to-prompt via the VLM.

**(14) Research data plotting (P1, S–M).** `experiments/plotting.py` is empty. Add a metrics table keyed by output, Vega-Lite/Plotly charts in the UI (sweep curves, seed-variance bands), and **notebook export** that regenerates every figure from DB queries.

**Model/adapter support (P0, M).** *v0.2.0:* all diffusers families load; a verified one-click catalog covers SD1.5/SDXL/SD3.5/FLUX.1/FLUX.2/Qwen-Image/Z-Image/Chroma/Sana/Krea 2, LoRAs, VAEs, ControlNets and GGUF LLMs. Remaining: treat SDXL, plus FLUX.2 [klein] 4B and Z-Image-Turbo (6B, 8 steps, under 16 GB) as the flow-matching references for the instrumented sampler, and add quantization presets for Qwen-Image and FLUX.2 [dev].

**Experiment logging / API (P0, S).** Fix seeds, add hashes and versions, and add an `approximate` flag. Make the API asynchronous (submit → job id → WebSocket progress).

---

## 6. "Future research" notes (plan_phases_v3)

| Note | Assessment and recommended form | Priority/Effort |
|---|---|---|
| Gradient flow visualization | Reframe as **per-block attribution**: JVP tangent norms per block, activation patching; DifFRACT (Jun 2026) traces transcoder circuits in FLUX.1-schnell, which makes a good stretch goal. | P2 / L |
| Cross-attention maps & control | ConceptAttention for reading maps, attention-output steering (FluxSpace) for control | P1 / M |
| Concept algebra | Do it in SAE, attention-output or LoRA-direction space, not VAE latents | P1 / M |
| Embedding/TI training UI | TI doesn't carry over to LLM encoders. LoRA is standard; delegate to external trainers and import the results | P2 / S |
| Style/artist DB | Falls out of the atlas: nearest-neighbour search on SigLIP/DINO embeddings | P2 / S |
| Node editor | **Don't build one.** ComfyUI (subgraphs, App Mode) and Mellon cover it. Experiment specs are GRADIENT's alternative | not recommended |
| Collaboration/federated | Out of scope for a local lab | drop |
| Multimodal inputs | Reference images are native in FLUX.2 and Qwen-Image-Edit. Audio fits as a driver for modulation lanes | P2 / M |

---

## 7. Corrections to `docs/research/VRAM_optimalization.md`

- The xFormers advice is outdated. PyTorch SDPA is the default, and the attention dispatcher adds FA3/Sage Hub kernels.
- `enable_model_cpu_offload` is *coarser* than sequential offload, not more granular. The diffusers docs call sequential offload extremely slow, and group offloading with streams supersedes both.
- Passing `load_in_8bit=True` to pipeline `from_pretrained` is not the current API. Use `PipelineQuantizationConfig` or per-component configs. For FLUX.2-class models, GGUF, torchao, Nunchaku and SDNQ matter more.
- FP8 is now a single call: `enable_layerwise_casting`.
- ToMe matters little for DiTs. Caches cover that role, but they **change outputs**: tag such runs as approximate and keep exact mode as the default in a research tool.
- Gradient checkpointing *is* relevant: reverse-mode sensitivity through the sampler needs it.

---

## 8. Positioning: what is genuinely differentiating

- **ComfyUI** is the execution and ecosystem leader. It added subgraphs (Aug 2025) and App Mode / ComfyHub (Mar 2026), and agent research such as ComfyMind builds on it. Experiments, statistics and introspection are not first-class there.
- **InvokeAI** centres on canvas-based creative workflows rather than measurement.
- **A1111-lineage UIs** (Forge, SD.Next) target generation; SD.Next already exposes the guider and caching stacks.

So "more knobs" isn't a moat. GRADIENT's defensible niche is **a lab notebook for diffusion**:

1. An **instrumented sampler** in which every step's x₀, guidance delta, attention outputs and tangent are inspectable.
2. **Correct measurement**: finite-difference/JVP sensitivity, step-localised effects, token ablation, ConceptAttention.
3. **Experiment as data**: specs → runs → metrics → atlas → notebooks, with exact reproducibility and approximation flags.
4. **Steering as a timeline**: σ-scheduled guidance, weaving and block-level modulation, shared by researchers and reactive artists.
5. **Hypothesis-driven agents** that use the DB as memory.

Ship the instrumentation as Modular Diffusers custom blocks too, so it works inside Mellon and diffusers rather than competing with them.

---

## 9. Roadmap: next three milestones

**M1: Trustworthy core (about 5 weeks; partly done in v0.2.0).**
- ~~Upgrade to diffusers ≥ 0.42.1, torch ≥ 2.6 and the `dtype` API.~~ Done in v0.2.0.
- Build the instrumented sampler with family adapters: SD1.5, SDXL, FLUX.2 [klein] 4B, Z-Image-Turbo.
- Run guiders standalone; ~~implement the CFG schedule parser, real sigmas~~ (done); add shift and σ-based scheduling.
- ~~Fix per-image generators~~ (done) and expand metadata (hashes, versions, approximation flag).
- Build the task queue (one GPU worker, async API, WebSocket progress).
- Add x₀ previews (TAESD/TAEF2) and memory presets (group offload, quantization).
- *Exit test:* bit-identical re-runs from PNG metadata on every supported family.

**M2: Measurement (about 6 weeks).**
- Sensitivity v2: finite differences plus JVP, step-localised curves, Taylor preview.
- Output atlas: SigLIP 2/DINOv3 embeddings in LanceDB, rendered with the Embedding Atlas Svelte component.
- Metrics plugins: CLIPScore/VQAScore, HPSv3, seed diversity.
- Experiment specs, plotting and notebook export.
- *Exit test:* reproduce a known result end to end, e.g. guidance-interval gains or APG reducing oversaturation.

**M3: Steering and interpretability (about 8 weeks).**
- σ-timeline modulation lanes: guidance, prompt weaving, layer skip, per-block LoRA scales for DiTs.
- ConceptAttention maps and token ablation.
- Attribute directions (FluxSpace, LOCO-Edit Jacobian sliders).
- FlowEdit-based grafting.
- Agent v1 that designs sweeps and judges them with a VLM via `llama-server`.

---

## Sources

- diffusers on PyPI (0.42.1, 27 Oct 2026): https://pypi.org/project/diffusers/
- diffusers release summaries (v0.39–v0.42.1): https://releases.sh/hugging-face/diffusers-releases.md
- diffusers guiders: https://huggingface.co/docs/diffusers/en/api/modular_diffusers/guiders
- diffusers caching: https://huggingface.co/docs/diffusers/main/optimization/cache
- diffusers memory / group offloading: https://huggingface.co/docs/diffusers/optimization/memory
- diffusers attention backends: https://huggingface.co/docs/diffusers/optimization/attention_backends
- diffusers callbacks: https://huggingface.co/docs/diffusers/using-diffusers/callback
- diffusers FLUX.2 pipeline: https://huggingface.co/docs/diffusers/api/pipelines/flux2
- Mellon + Modular Diffusers: https://huggingface.co/docs/diffusers/v0.39.0/modular_diffusers/mellon
- SD.Next changelog (guider/caching stacks): https://raw.githubusercontent.com/vladmandic/sdnext/master/CHANGELOG.md
- ConceptAttention (ICML 2025): https://arxiv.org/abs/2502.04320
- ConceptAttention reference code: https://github.com/helblazer811/ConceptAttention
- FluxSpace (CVPR 2025): https://arxiv.org/abs/2412.09611
- SliderSpace (ICCV 2025): https://arxiv.org/abs/2502.01639
- FlowEdit (ICCV 2025): https://arxiv.org/abs/2412.08629
- Stable Flow (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Avrahami_Stable_Flow_Vital_Layers_for_Training-Free_Image_Editing_CVPR_2025_paper.html
- LOCO-Edit, low-rank Jacobian subspaces (NeurIPS 2024): https://arxiv.org/abs/2409.02374
- SAEs/ITDA on FLUX.1: https://arxiv.org/abs/2505.24360
- DifFRACT circuit tracing (Jun 2026): https://arxiv.org/html/2606.15796
- Rectified-CFG++ (NeurIPS 2025): https://arxiv.org/abs/2510.07631
- CFG-Zero*: https://arxiv.org/abs/2503.18886
- APG: https://arxiv.org/abs/2410.02416
- CFG++: https://arxiv.org/abs/2406.08070
- Frequency-Decoupled Guidance: https://arxiv.org/abs/2506.19713
- MAMBO-G: https://arxiv.org/abs/2508.03442
- Guidance interval (Kynkäänniemi et al.): https://arxiv.org/abs/2404.07724
- Normalized Attention Guidance (NeurIPS 2025): https://arxiv.org/abs/2505.21179
- Align Your Steps: https://arxiv.org/abs/2404.14507
- Optimal Stepsize for Diffusion Sampling: https://arxiv.org/abs/2503.21774
- Adjoint Matching (ICLR 2025): https://arxiv.org/abs/2409.08861
- PyTorch issue, no forward-AD for flash SDPA: https://github.com/pytorch/pytorch/issues/165530
- ELLA repository: https://github.com/TencentQQGYLab/ELLA
- Z-Image: https://arxiv.org/abs/2511.22699
- Qwen-Image technical report: https://arxiv.org/abs/2508.02324
- TAEF2 (tiny autoencoder for FLUX.2): https://huggingface.co/madebyollin/taef2
- T2I-Copilot (ICCV 2025): https://arxiv.org/abs/2507.20536
- ComfyMind (NeurIPS 2025): https://arxiv.org/abs/2505.17908
- HPSv3 (ICCV 2025): https://arxiv.org/abs/2508.03789
- DINOv3: https://arxiv.org/abs/2508.10104
- SigLIP 2: https://arxiv.org/abs/2502.14786
- SAM 3: https://github.com/facebookresearch/sam3
- llama.cpp multimodal (libmtmd): https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
- Embedding Atlas (Apple): https://github.com/apple/embedding-atlas
- ComfyUI subgraphs: https://blog.comfy.org/p/subgraph-official-release
- ComfyUI App Mode (Mar 2026): https://gigazine.net/gsc_news/en/20260312-comfyui-app-mode
