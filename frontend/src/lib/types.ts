// frontend/src/lib/types.ts (NEW or Append)

// --- From diffusion/models.py ---
export interface ImageResult {
	image_base64: string;
	seed: number;
	parameters: Record<string, any>; // Dictionary for parameters used
}

// --- From viz/models.py ---
// Not strictly needed for grid display, but good to have centrally

// --- From diffusion/models.py (XYZ Grid) ---
export interface AxisParameter {
    parameter: string;
    values: any[]; // Can be string, number etc.
}

export interface DiffusionTaskBase {
    model_id: string;
    prompt: string;
    negative_prompt?: string;
    height?: number;
    width?: number;
    num_inference_steps?: number;
    guidance_scale?: number;
    seed?: number | null;
    custom_sigmas?: number[] | null;
}

export interface XYZGridRequest {
    base_parameters: DiffusionTaskBase;
    x_axis?: AxisParameter | null;
    y_axis?: AxisParameter | null;
    z_axis?: AxisParameter | null;
}

export interface XYZGridResult {
	x_value?: any;
	y_value?: any;
	z_value?: any;
	parameters: Record<string, any>; // Full parameters for this cell
	image_result?: ImageResult | null; // The generated image
	error?: string | null; // Error message if generation failed
}

export interface XYZGridResponse {
	grid_results: XYZGridResult[];
	request_params: XYZGridRequest;
}

// Add other types (LLM, Sensitivity etc.) here as needed