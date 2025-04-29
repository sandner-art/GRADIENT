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