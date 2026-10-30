# src/gradient/ai/diffusion/pipelines.py
"""Text-to-image and image-to-image generation for any diffusers model family.

Generation calls the pipeline's own `__call__` (so SDXL / SD3.5 / FLUX / Qwen-Image
/ Z-Image ... behave exactly as their authors intended) and attaches research
hooks through `callback_on_step_end`:
  * scheduled CFG (per-step guidance values, see guidance.py)
  * step progress reporting (/api/v1/system/status)
The hook point is also where future modulation (prompt switching, latent
grafting, layer skipping) plugs in.

Every image is generated with its own seeded generator, so the logged seed of
image i reproduces that image alone.
"""

import base64
import datetime
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from gradient.core.config import get_outputs_dir, get_setting
from gradient.core.device import get_device
from gradient.core.scheduling import progress_tracker, run_gpu_task
from gradient.database.models import ExperimentRun, GeneratedOutput
from gradient.database.session import get_db_session, update_run_status

from .guidance import is_constant, parse_guidance_schedule
from .loading import DYNAMIC_CFG_FAMILIES, LoadedPipeline, get_pipeline
from .models import (
    DiffusionTaskBase,
    ImageResult,
    ImageToImageRequest,
    ImageToImageResponse,
    Text2ImageRequest,
    Text2ImageResponse,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- helpers
def get_output_base_path():
    return get_outputs_dir()


def prepare_init_image(image_base64: str, width: int, height: int) -> Image.Image:
    """Decode and resize the initial image (raises ValueError on bad input)."""
    try:
        if "," in image_base64[:100] and image_base64.lstrip().startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]
        image = Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not decode the input image: {e}") from e
    return image.resize((width, height), Image.Resampling.LANCZOS)


def resolve_seed(seed: int | None) -> int:
    if seed is None:
        return int(torch.randint(0, 2**32 - 1, (1,)).item())
    return int(seed) % (2**32)


def _set_scheduler(pipe, entry: LoadedPipeline, scheduler_id: str | None) -> str:
    """Switch scheduler (validated against the model's compatible list) or restore the default."""
    import diffusers

    if entry.default_scheduler_config is None or getattr(pipe, "scheduler", None) is None:
        return "n/a"
    default_name = entry.default_scheduler_config.get("_class_name", type(pipe.scheduler).__name__)
    target = scheduler_id if scheduler_id and scheduler_id != "Default" else default_name
    if type(pipe.scheduler).__name__ == target:
        return target  # schedulers are always rebuilt from the model's default config
    if target not in compatible_schedulers(entry):
        raise ValueError(
            f"Scheduler '{target}' is not compatible with {entry.pipeline_class}. "
            f"Compatible: {', '.join(compatible_schedulers(entry))}"
        )
    cls = getattr(diffusers, target)
    pipe.scheduler = cls.from_config(entry.default_scheduler_config)
    return target


def compatible_schedulers(entry: LoadedPipeline) -> list[str]:
    import diffusers

    name = (entry.default_scheduler_config or {}).get("_class_name")
    if not name:
        return []
    cls = getattr(diffusers, name, None)
    names = {name} | set(getattr(cls, "_compatibles", []) or [])
    return sorted(n for n in names if isinstance(n, str) and hasattr(diffusers, n))


def _encode_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class GenerationResult:
    images: list[Image.Image]
    seeds: list[int]
    params: dict[str, Any]


# --------------------------------------------------------------------------- core (blocking)
def generate_images(
    request: DiffusionTaskBase,
    task: str = "text2image",
    init_image_b64: str | None = None,
    strength: float | None = None,
    num_outputs: int = 1,
    output_type: str = "pil",
    probes: list | None = None,
    call_overrides: dict | None = None,
) -> GenerationResult:
    """Blocking generation. Run via `run_gpu_task`.

    `probes` (see gradient.ai.lab.probes) are attached for the duration of the call and see
    every step boundary; `call_overrides` replaces pipeline kwargs (e.g. precomputed
    prompt embeddings for token ablation).
    """
    pipe, entry = get_pipeline(
        request.model_id,
        task=task,
        vae_id=request.vae_id,
        loras=request.loras,
        embeddings=request.embeddings,
    )
    call_signature = inspect.signature(pipe.__call__).parameters
    call_params = set(call_signature)
    params: dict[str, Any] = request.model_dump(mode="json", exclude={"image_base64"})
    params.update(
        pipeline=type(pipe).__name__,
        family=entry.family,
        memory_mode=entry.memory_mode,
        dtype=str(entry.dtype).removeprefix("torch."),
    )

    params["scheduler_id"] = _set_scheduler(pipe, entry, request.scheduler_id)

    kwargs: dict[str, Any] = {
        "prompt": request.prompt,
        "height": request.height,
        "width": request.width,
        "num_inference_steps": request.num_inference_steps,
        "num_images_per_prompt": num_outputs,
        "output_type": output_type,
    }
    if request.negative_prompt and "negative_prompt" in call_params:
        kwargs["negative_prompt"] = request.negative_prompt
    elif request.negative_prompt:
        params["negative_prompt_applied"] = False

    # Noise schedule
    if request.custom_sigmas:
        if "sigmas" not in call_params:
            raise ValueError(f"{entry.pipeline_class} does not accept custom sigmas")
        if "sigmas" not in inspect.signature(pipe.scheduler.set_timesteps).parameters:
            raise ValueError(
                f"Scheduler {type(pipe.scheduler).__name__} does not support custom sigmas; "
                "choose EulerDiscreteScheduler (UNet models) or the default flow-matching scheduler."
            )
        kwargs["sigmas"] = list(request.custom_sigmas)
        params["custom_sigmas_applied"] = True

    # Guidance (constant or scheduled)
    est_steps = max(
        1,
        (len(request.custom_sigmas) - 1) if request.custom_sigmas else request.num_inference_steps,
    )
    schedule = parse_guidance_schedule(request.guidance_scale, est_steps)
    # Qwen-Image style pipelines run real CFG through `true_cfg_scale` (needs a negative prompt);
    # their `guidance_scale` only exists for guidance-distilled variants.
    gs_param = call_signature.get("guidance_scale")
    uses_true_cfg = "true_cfg_scale" in call_params and (
        gs_param is None or gs_param.default is None
    )
    if uses_true_cfg:
        kwargs["true_cfg_scale"] = schedule[0]
        if schedule[0] > 1 and "negative_prompt" not in kwargs:
            kwargs["negative_prompt"] = " "
        params["guidance_parameter"] = "true_cfg_scale"
    else:
        kwargs["guidance_scale"] = schedule[0]
    callbacks_enabled = "callback_on_step_end" in call_params
    scheduled = not is_constant(schedule)
    if scheduled and (entry.family not in DYNAMIC_CFG_FAMILIES or not callbacks_enabled):
        logger.warning(
            "Scheduled guidance is not supported for %s; using %.2f",
            entry.pipeline_class,
            schedule[0],
        )
        params["guidance_schedule_applied"] = False
        scheduled = False
    elif scheduled:
        params["guidance_schedule_applied"] = True
        params["guidance_values"] = [round(v, 4) for v in schedule]

    cfg_active = schedule[0] > 1.0

    from gradient.ai.lab.probes import ModulationController

    all_probes = list(probes or [])
    if request.modulation:
        if not callbacks_enabled:
            raise ValueError(
                f"{entry.pipeline_class} does not support step callbacks, so modulation is unavailable"
            )
        all_probes.insert(0, ModulationController(request.modulation, request))

    def on_step_end(pipeline, step: int, timestep, callback_kwargs: dict) -> dict:
        progress_tracker.update(step + 1)
        if scheduled and step + 1 < len(schedule):
            value = schedule[step + 1]
            # The CFG batch layout is fixed at the first step: keep CFG on (>1) or off (<=1).
            pipeline._guidance_scale = max(value, 1.0001) if cfg_active else min(value, 1.0)
        for probe in all_probes:
            callback_kwargs = probe.on_step_end(pipeline, step, timestep, callback_kwargs)
        return callback_kwargs

    if callbacks_enabled:
        kwargs["callback_on_step_end"] = on_step_end

    # Per-image seeds => each image is reproducible from its own logged seed.
    base_seed = resolve_seed(request.seed)
    seeds = [(base_seed + i) % (2**32) for i in range(num_outputs)]
    exec_device = getattr(pipe, "_execution_device", get_device())
    gen_device = "cpu" if torch.device(exec_device).type == "mps" else exec_device
    kwargs["generator"] = [torch.Generator(device=gen_device).manual_seed(s) for s in seeds]

    if task == "image2image":
        if init_image_b64 is None:
            raise ValueError("image_base64 is required for image-to-image")
        kwargs["image"] = prepare_init_image(init_image_b64, request.width, request.height)
        if strength is not None and "strength" in call_params:
            kwargs["strength"] = strength

    if call_overrides:
        kwargs.update(call_overrides)
        if "prompt_embeds" in call_overrides:
            kwargs.pop("prompt", None)
            kwargs.pop("negative_prompt", None)

    kwargs = {k: v for k, v in kwargs.items() if k in call_params or k == "callback_on_step_end"}
    probe_steps = est_steps
    if task == "image2image" and strength is not None:
        probe_steps = max(1, int(est_steps * strength))

    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        if request.width * request.height >= int(
            get_setting("performance.vae_tiling_min_pixels", 1572864)
        ):
            vae.enable_tiling()
        elif hasattr(vae, "disable_tiling"):
            vae.disable_tiling()

    progress_tracker.start(task, request.model_id, est_steps)
    attached: list = []
    try:
        with torch.inference_mode():
            for probe in all_probes:
                probe.attach(pipe, entry, probe_steps, kwargs)
                attached.append(probe)
            if all_probes:
                allowed = set(getattr(pipe, "_callback_tensor_inputs", ["latents"]))
                wanted = {"latents"} | {k for probe in all_probes for k in probe.tensor_inputs}
                kwargs["callback_on_step_end_tensor_inputs"] = sorted(wanted & allowed)
            output = pipe(**kwargs)
    finally:
        for probe in reversed(attached):
            probe.detach()
        progress_tracker.finish()
    for probe in all_probes:
        params.update(probe.report())
    images = list(output.images)
    if not images:
        raise RuntimeError("Pipeline returned no images")
    return GenerationResult(images=images, seeds=seeds, params=params)


# --------------------------------------------------------------------------- async wrappers
async def _create_run(run_uuid: str, run_type: str, params: dict, start_time: float) -> int | None:
    try:
        async with get_db_session() as db:
            run = ExperimentRun(
                run_uuid=run_uuid,
                run_type=run_type,
                start_time=datetime.datetime.fromtimestamp(start_time, tz=datetime.UTC),
                status="running",
                base_parameters=params,
            )
            db.add(run)
            await db.flush()
            return run.id
    except Exception as e:
        logger.error("Run %s: could not create DB record: %s", run_uuid, e)
        return None


def _save_outputs(
    result: GenerationResult, run_uuid: str, prefix: str
) -> list[tuple[str, dict, Image.Image]]:
    out_dir = get_output_base_path() / run_uuid
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for img, seed in zip(result.images, result.seeds, strict=False):
        params = {**result.params, "seed": seed}
        info = PngInfo()
        info.add_text("grad_parameters", json.dumps(params, default=str))
        path = out_dir / f"{prefix}_{uuid.uuid4().hex[:8]}_seed{seed}.png"
        img.save(path, "PNG", pnginfo=info)
        saved.append((str(path.resolve()), params, img))
    return saved


async def _run(
    run_type: str,
    request: Text2ImageRequest | ImageToImageRequest,
    parent_run_uuid: str | None,
    parent_run_id: int | None,
) -> tuple[list[ImageResult], float]:
    start = time.time()
    run_uuid = parent_run_uuid or str(uuid.uuid4())
    db_run_id = parent_run_id
    if parent_run_uuid is None:
        logged = request.model_dump(mode="json", exclude={"image_base64"})
        if isinstance(request, ImageToImageRequest):
            logged["image_provided"] = True
        db_run_id = await _create_run(run_uuid, run_type, logged, start)

    task = "image2image" if isinstance(request, ImageToImageRequest) else "text2image"
    try:
        result = await run_gpu_task(
            generate_images,
            request,
            task,
            getattr(request, "image_base64", None),
            getattr(request, "strength", None),
            request.num_outputs,
        )
    except Exception as e:
        if db_run_id and parent_run_uuid is None:
            await update_run_status(
                db_run_id, "failed", error=f"{type(e).__name__}: {e}", end_time=time.time()
            )
        raise

    prefix = "t2i" if task == "text2image" else "i2i"
    saved = _save_outputs(result, run_uuid, prefix)
    images = [
        ImageResult(image_base64=_encode_png(img), seed=p["seed"], parameters=p)
        for _, p, img in saved
    ]

    if db_run_id:
        try:
            async with get_db_session() as db:
                db.add_all(
                    GeneratedOutput(
                        run_id=db_run_id,
                        output_type="image",
                        file_path=path,
                        parameters=params,
                        creation_time=datetime.datetime.now(tz=datetime.UTC),
                    )
                    for path, params, _ in saved
                )
        except Exception as e:
            logger.error("Run %s: could not log outputs: %s", run_uuid, e)
        if parent_run_uuid is None:
            await update_run_status(db_run_id, "completed", end_time=time.time())

    elapsed_ms = (time.time() - start) * 1000
    logger.info("Run %s (%s): %d image(s) in %.0f ms", run_uuid, run_type, len(images), elapsed_ms)
    return images, elapsed_ms


async def run_text_to_image(
    request: Text2ImageRequest, parent_run_uuid: str | None = None, parent_run_id: int | None = None
) -> Text2ImageResponse:
    images, ms = await _run("text_to_image", request, parent_run_uuid, parent_run_id)
    return Text2ImageResponse(images=images, request_params=request, performance_ms=ms)


async def run_image_to_image(
    request: ImageToImageRequest,
    parent_run_uuid: str | None = None,
    parent_run_id: int | None = None,
) -> ImageToImageResponse:
    images, ms = await _run("image_to_image", request, parent_run_uuid, parent_run_id)
    return ImageToImageResponse(images=images, request_params=request, performance_ms=ms)
