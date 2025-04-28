# gradient-project-root/src/gradient/ai/diffusion/xyz_grid.py (NEW FILE)

import logging
import itertools
import copy
from typing import List, Dict, Any, Optional

from .models import XYZGridRequest, XYZGridResult, AxisParameter, Text2ImageRequest, ImageResult
from .pipelines import run_text_to_image # Use the existing text2image function

logger = logging.getLogger(__name__)

async def run_xyz_grid(request: XYZGridRequest) -> List[XYZGridResult]:
    """Generates images based on XYZ grid parameter variations."""
    results: List[XYZGridResult] = []
    base_params_dict = request.base_parameters.model_dump(exclude_unset=True)

    # Prepare axes, handling None cases
    axes = [request.x_axis, request.y_axis, request.z_axis]
    active_axes = [(i, axis) for i, axis in enumerate(axes) if axis is not None] # Store index (0=x, 1=y, 2=z) and AxisParameter

    if not active_axes:
        logger.warning("XYZ Grid request received with no active axes. Running base parameters once.")
        # Treat as a single run with base parameters
        x_val, y_val, z_val = None, None, None
        params_for_run = base_params_dict.copy()
        t2i_request = Text2ImageRequest(**params_for_run, num_outputs=1) # Force 1 output per cell
        grid_cell_result = XYZGridResult(x_value=x_val, y_value=y_val, z_value=z_val, parameters=params_for_run.copy())
        try:
            response = await run_text_to_image(t2i_request)
            if response and response.images:
                grid_cell_result.image_result = response.images[0]
                # Update parameters in result with the actual ones used (e.g., resolved seed)
                grid_cell_result.parameters = response.request_params.model_dump(exclude_unset=True)
            else:
                grid_cell_result.error = "Generation failed or produced no image."
        except Exception as e:
            logger.error(f"Error generating single cell for XYZ grid: {e}", exc_info=True)
            grid_cell_result.error = f"Generation error: {e}"
        results.append(grid_cell_result)
        return results


    axis_params = [axis.parameter for _, axis in active_axes]
    axis_values_lists = [axis.values for _, axis in active_axes]

    # Check for duplicate parameter names across active axes
    if len(axis_params) != len(set(axis_params)):
        logger.error(f"Duplicate parameter names found across XYZ axes: {axis_params}")
        # Return an error or handle gracefully? For now, log and continue, behavior might be odd.
        # raise ValueError("Duplicate parameter names used for different XYZ axes.")

    total_images = 1
    for values in axis_values_lists:
        total_images *= len(values)

    logger.info(f"Starting XYZ grid generation. Axes: {axis_params}. Total images: {total_images}")

    # Iterate through the Cartesian product of all axis values
    value_combinations = list(itertools.product(*axis_values_lists))

    for i, combo in enumerate(value_combinations):
        params_for_run = base_params_dict.copy()
        current_axis_values = {} # Track values for this specific cell

        # Apply axis values to parameters
        for axis_index, axis_value in enumerate(combo):
            param_name = axis_params[axis_index]
            logger.debug(f"Cell {i+1}/{total_images}: Setting {param_name} = {axis_value}")
            params_for_run[param_name] = axis_value
            current_axis_values[param_name] = axis_value

        # Prepare grid cell result structure
        x_val = combo[active_axes.index((0, request.x_axis))] if request.x_axis else None
        y_val = combo[active_axes.index((1, request.y_axis))] if request.y_axis else None
        z_val = combo[active_axes.index((2, request.z_axis))] if request.z_axis else None

        # Ensure seed is handled correctly - if seed is an axis, use that value, otherwise use base or increment
        if 'seed' not in axis_params and params_for_run.get('seed') is not None:
             # Increment seed for each cell if not controlled by an axis
             params_for_run['seed'] += i
             logger.debug(f"Cell {i+1}: Incrementing seed to {params_for_run['seed']}")
        elif 'seed' not in params_for_run:
             params_for_run['seed'] = None # Explicitly None if not set by base or axis

        grid_cell_result = XYZGridResult(x_value=x_val, y_value=y_val, z_value=z_val, parameters=params_for_run.copy())

        # Run text-to-image for this specific combination
        # Ensure num_outputs is 1 for grid generation
        logger.info(f"Generating image {i+1}/{total_images} for grid cell: X={x_val}, Y={y_val}, Z={z_val}")
        t2i_request = Text2ImageRequest(**params_for_run, num_outputs=1)

        try:
            response = await run_text_to_image(t2i_request)
            if response and response.images:
                grid_cell_result.image_result = response.images[0]
                # Update parameters with the actual ones used (e.g., resolved seed)
                # Important: response.request_params has the *final* params including resolved seed
                grid_cell_result.parameters = response.request_params.model_dump(exclude_unset=True)
                logger.info(f"Image {i+1}/{total_images} generated successfully.")
            else:
                error_msg = "Generation failed or produced no image."
                grid_cell_result.error = error_msg
                logger.error(f"Image {i+1}/{total_images} failed: {error_msg}")
        except Exception as e:
            error_msg = f"Generation error: {type(e).__name__} - {e}"
            logger.error(f"Error generating image {i+1}/{total_images} for grid cell: {e}", exc_info=True)
            grid_cell_result.error = error_msg

        results.append(grid_cell_result)
        # Optional: Add a small delay or check for cancellation requests here for long grids

    logger.info(f"XYZ grid generation completed. Processed {len(results)} cells.")
    return results