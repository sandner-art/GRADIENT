// frontend/src/lib/types.ts

// --- From diffusion/models.py ---
export interface ImageResult {
	image_base64: string;
	seed: number;
	parameters: Record<string, any>; // Dictionary for parameters used
}

export interface DiffusionTaskBase {
	model_id: string;
	prompt: string;
	negative_prompt?: string | null; // Allow null
	height?: number;
	width?: number;
	num_inference_steps?: number;
	guidance_scale?: number; // Allow float, string or list for scheduled CFG later? For now, number.
	seed?: number | null;
	custom_sigmas?: number[] | null;
	scheduler_id?: string | null; // Added scheduler
	// vae_id?: string | null; // Add when ready
}

// --- Text-to-Image Specific --- vvv
export interface Text2ImageRequest extends DiffusionTaskBase {
	num_outputs?: number;
}

export interface Text2ImageResponse {
	images: ImageResult[];
	request_params: Text2ImageRequest; // Use the specific request type
	performance_ms?: number | null;
}
// --- Text-to-Image Specific --- ^^^


// --- From presets API models --- vvv

/** Subset of parameters typically saved in a preset. */
export interface PresetSetting {
	negative_prompt?: string | null;
	height?: number | null;
	width?: number | null;
	num_inference_steps?: number | null;
	guidance_scale?: number | null; // Keep as number for now, handle scheduled CFG later if needed
	scheduler_id?: string | null;
	strength?: number | null; // For Img2Img presets
    // vae_id?: string | null; // Add when ready
    // Allow any other extra fields the backend might store
    [key: string]: any;
}

/** Base structure for creating/reading presets (excluding DB ID etc.) */
export interface PresetBase {
	model_identifier: string;
	preset_name: string;
	settings: PresetSetting;
	description?: string | null;
}

/** Structure for creating a new preset via API */
export interface PresetCreate extends PresetBase {
	// No extra fields needed for creation currently
}

/** Structure for reading a preset from the API (includes DB fields) */
export interface PresetRead extends PresetBase {
	id: number;
	is_user_defined: boolean;
	creation_time: string; // Represent DateTime as string from JSON
	last_used_time?: string | null; // Represent DateTime as string from JSON
}

// --- From presets API models --- ^^^

// --- From viz/models.py ---
// Add LatentSpaceVizRequest/Response etc. here later if needed


// --- From diffusion/models.py (XYZ Grid) ---
export interface AxisParameter {
	parameter: string;
	values: any[]; // Can be string, number etc.
}

export interface XYZGridRequest {
	base_parameters: DiffusionTaskBase; // Uses the base type
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

// --- From diffusion/models.py (Img2Img) --- vvv
export interface ImageToImageRequest extends DiffusionTaskBase {
    image_base64: string;
    strength?: number;
    num_outputs?: number;
}

export interface ImageToImageResponse {
    images: ImageResult[];
    request_params: ImageToImageRequest;
    performance_ms?: number | null;
}
// --- From diffusion/models.py (Img2Img) --- ^^^


// Add other types (LLM, Sensitivity etc.) here as needed