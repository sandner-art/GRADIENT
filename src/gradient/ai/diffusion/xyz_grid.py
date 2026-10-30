# src/gradient/ai/diffusion/xyz_grid.py

import datetime  # Import datetime
import itertools
import logging
import time
import uuid  # Import uuid

from gradient.database.models import ExperimentRun  # Import model

# DB Imports
from gradient.database.session import get_db_session, update_run_status  # Import updater

from .models import Text2ImageRequest, XYZGridRequest, XYZGridResult
from .pipelines import resolve_seed, run_text_to_image

logger = logging.getLogger(__name__)


async def run_xyz_grid(request: XYZGridRequest) -> list[XYZGridResult]:
    """Generates images based on XYZ grid parameter variations, logging as one ExperimentRun."""
    start_time = time.time()
    run_uuid = str(uuid.uuid4())  # Unique ID for the entire grid run
    results: list[XYZGridResult] = []
    base_params_dict = request.base_parameters.model_dump(exclude_unset=True)
    grid_status = "completed"  # Assume success initially
    grid_error_details = []

    # --- DB Logging Start for Grid ---
    db_run: ExperimentRun | None = None
    try:
        async with get_db_session() as db:
            db_run = ExperimentRun(
                run_uuid=run_uuid,
                run_type="xyz_grid",
                start_time=datetime.datetime.fromtimestamp(start_time, tz=datetime.UTC),
                status="running",
                # Store base parameters and axis definitions
                base_parameters={
                    "base": base_params_dict,
                    "x_axis": request.x_axis.model_dump(mode="json") if request.x_axis else None,
                    "y_axis": request.y_axis.model_dump(mode="json") if request.y_axis else None,
                    "z_axis": request.z_axis.model_dump(mode="json") if request.z_axis else None,
                },
            )
            db.add(db_run)
            await db.commit()
            await db.refresh(db_run)
            logger.info(f"Created ExperimentRun record ID {db_run.id} for XYZ grid {run_uuid}")
    except Exception as e:
        logger.error(
            f"Failed to create initial DB record for XYZ grid run {run_uuid}: {e}", exc_info=True
        )
        db_run = None  # Ensure db_run is None
    # --- End DB Logging Start ---

    # --- Axis Preparation (Unchanged) ---
    axes = [request.x_axis, request.y_axis, request.z_axis]
    active_axes = [(i, axis) for i, axis in enumerate(axes) if axis is not None]

    if not active_axes:
        logger.warning(f"XYZ Grid run {run_uuid} has no active axes. Running base parameters once.")
        # Handle single run case (simplified)
        params_for_run = base_params_dict.copy()
        t2i_request = Text2ImageRequest(**params_for_run, num_outputs=1)
        grid_cell_result = XYZGridResult(
            x_value=None, y_value=None, z_value=None, parameters=params_for_run.copy()
        )
        try:
            # Pass run_uuid so t2i logs output under this run
            response = await run_text_to_image(
                t2i_request, parent_run_uuid=run_uuid, parent_run_id=db_run.id if db_run else None
            )
            if response and response.images:
                grid_cell_result.image_result = response.images[0]
                grid_cell_result.parameters = response.images[
                    0
                ].parameters  # Get specific params from result
            else:
                grid_cell_result.error = "Generation failed or produced no image."
                grid_status = "failed"  # Mark grid as failed if any cell fails
                grid_error_details.append(f"Cell (base): {grid_cell_result.error}")
        except Exception as e:
            logger.error(
                f"Error generating single cell for XYZ grid {run_uuid}: {e}", exc_info=True
            )
            grid_cell_result.error = f"Generation error: {type(e).__name__} - {e}"
            grid_status = "failed"
            grid_error_details.append(f"Cell (base): {grid_cell_result.error}")
        results.append(grid_cell_result)
        # Update final status
        if db_run:
            await update_run_status(
                db_run.id,
                grid_status,
                error="; ".join(grid_error_details) or None,
                end_time=time.time(),
            )
        return results

    # --- Grid Iteration ---
    axis_params = [axis.parameter for _, axis in active_axes]
    axis_values_lists = [axis.values for _, axis in active_axes]

    if len(axis_params) != len(set(axis_params)):
        logger.error(f"Duplicate parameter names found across XYZ axes: {axis_params}")
        # Continue but log error, grid generation might be weird
        if db_run:
            await update_run_status(db_run.id, "failed", error="Duplicate axis parameters")
        raise ValueError(f"Each axis must vary a different parameter, got {axis_params}")

    total_images = 1
    for values in axis_values_lists:
        total_images *= len(values)
    logger.info(f"Starting XYZ grid {run_uuid}. Axes: {axis_params}. Total images: {total_images}")

    value_combinations = list(itertools.product(*axis_values_lists))

    # One shared seed for the whole grid (resolved now if random), unless seed is an axis.
    if "seed" not in axis_params and base_params_dict.get("seed") is None:
        base_params_dict["seed"] = resolve_seed(None)

    # Generate all cells of one model before switching models (avoids reloading a model per
    # cell); results are put back into the original grid order afterwards.
    order = list(range(len(value_combinations)))
    if "model_id" in axis_params:
        model_pos = axis_params.index("model_id")
        order.sort(key=lambda idx: str(value_combinations[idx][model_pos]))
    cell_results: dict[int, XYZGridResult] = {}

    for i in order:
        combo = value_combinations[i]
        params_for_run = base_params_dict.copy()
        current_axis_values = {}

        for axis_index, axis_value in enumerate(combo):
            param_name = axis_params[axis_index]
            params_for_run[param_name] = axis_value
            current_axis_values[param_name] = axis_value

        x_val = combo[active_axes.index((0, request.x_axis))] if request.x_axis else None
        y_val = combo[active_axes.index((1, request.y_axis))] if request.y_axis else None
        z_val = combo[active_axes.index((2, request.z_axis))] if request.z_axis else None

        # Keep the base seed fixed across cells (unless seed is an axis) so cells differ only
        # by the axis parameters -- that is the point of a comparison grid.
        if "seed" not in params_for_run:
            params_for_run["seed"] = None

        grid_cell_result = XYZGridResult(
            x_value=x_val, y_value=y_val, z_value=z_val, parameters=params_for_run.copy()
        )  # Store initial params

        logger.info(
            f"Generating image {i + 1}/{total_images} for grid {run_uuid}: X={x_val}, Y={y_val}, Z={z_val}"
        )
        t2i_request = Text2ImageRequest(**params_for_run, num_outputs=1)

        try:
            # Pass run_uuid/run id so outputs are saved and logged under this grid run
            response = await run_text_to_image(
                t2i_request, parent_run_uuid=run_uuid, parent_run_id=db_run.id if db_run else None
            )

            if response and response.images:
                grid_cell_result.image_result = response.images[0]
                # Get the *actual* parameters used from the image result (includes resolved seed)
                grid_cell_result.parameters = response.images[0].parameters
                logger.info(
                    f"Image {i + 1}/{total_images} generated successfully for grid {run_uuid}."
                )
            else:
                error_msg = "Generation failed or produced no image."
                grid_cell_result.error = error_msg
                logger.error(
                    f"Image {i + 1}/{total_images} failed for grid {run_uuid}: {error_msg}"
                )
                grid_status = "failed"  # Mark overall grid as failed
                grid_error_details.append(f"Cell {i + 1}: {error_msg}")

        except Exception as e:
            error_msg = f"Generation error: {type(e).__name__} - {e}"
            logger.error(
                f"Error generating image {i + 1}/{total_images} for grid {run_uuid}: {e}",
                exc_info=True,
            )
            grid_cell_result.error = error_msg
            grid_status = "failed"
            grid_error_details.append(f"Cell {i + 1}: {error_msg}")

        cell_results[i] = grid_cell_result

    results.extend(cell_results[i] for i in range(len(value_combinations)))

    # --- Update Final Grid Status ---
    end_time = time.time()
    final_error_str = "; ".join(grid_error_details) if grid_error_details else None
    if db_run:
        await update_run_status(db_run.id, grid_status, error=final_error_str, end_time=end_time)

    logger.info(
        f"XYZ grid {run_uuid} generation completed. Processed {len(results)} cells. Status: {grid_status}"
    )
    return results
