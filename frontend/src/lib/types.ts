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
	loras?: LoRAParameter[] | null;
	modulation?: ModulationLane[] | null;
	// vae_id?: string | null; // Add when ready
}

export interface LoRAParameter {
	name: string;
	weight?: number;
}

/** One lane on the denoising timeline, active for steps in [start, end) (fractions of the run). */
export type ModulationLaneType = 'guidance' | 'prompt' | 'attention_scale' | 'lora_weight';
export interface ModulationLane {
	type: ModulationLaneType;
	start: number;
	end: number;
	value?: number | null;
	prompt?: string | null;
	mode?: 'replace' | 'blend';
	targets?: string | null;
	lora_index?: number | null;
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
// --- Model manager (src/gradient/model_manager/schemas.py) --- vvv
export type ModelType =
	'diffusers' | 'checkpoints' | 'vae' | 'lora' | 'controlnet' | 'embeddings' | 'llm' | 'vision';
export type JobStatus =
	'queued' | 'resolving' | 'downloading' | 'completed' | 'failed' | 'cancelled';

export interface GenerationDefaults {
	width?: number | null;
	height?: number | null;
	steps?: number | null;
	guidance_scale?: number | null;
	scheduler?: string | null;
}

export interface CatalogEntry {
	id: string;
	name: string;
	type: ModelType;
	family?: string | null;
	repo_id?: string | null;
	files?: string[] | null;
	gated: boolean;
	license?: string | null;
	size_gb?: number | null;
	min_vram_gb?: number | null;
	pipeline?: string | null;
	base_model_family?: string | null;
	defaults?: GenerationDefaults | null;
	tags: string[];
	description?: string | null;
	homepage?: string | null;
	notes?: string | null;
	installed: boolean;
	installed_path?: string | null;
}

export interface LocalModel {
	name: string;
	type: ModelType;
	format: 'diffusers' | 'single_file' | 'peft' | 'gguf' | 'directory';
	path: string;
	size_bytes: number;
	family?: string | null;
	pipeline?: string | null;
	source?: Record<string, string> | null;
	defaults?: GenerationDefaults | null;
}

export interface DownloadRequest {
	catalog_id?: string;
	repo_id?: string;
	url?: string;
	model_type?: ModelType | null;
	files?: string[] | null;
	target_name?: string | null;
	token?: string | null;
}

export interface DownloadJob {
	id: string;
	status: JobStatus;
	name: string;
	model_type?: ModelType | null;
	source: string;
	target_path?: string | null;
	total_bytes?: number | null;
	downloaded_bytes: number;
	speed_bps: number;
	progress?: number | null;
	error?: string | null;
	error_code?: string | null;
	catalog_id?: string | null;
	created_at: string;
	finished_at?: string | null;
}

export interface StorageInfo {
	search_dirs: { path: string; exists: boolean; free_gb?: number; total_gb?: number }[];
	download_dir: string;
	hf_token_configured: boolean;
	civitai_token_configured: boolean;
}
// --- Model manager --- ^^^

// --- Lab (src/gradient/ai/lab/models.py) --- vvv
export interface TaylorRequest {
	base_request: Text2ImageRequest;
	method?: 'auto' | 'jvp' | 'finite_difference';
	delta?: number | null;
}

export interface TaylorResponse {
	method: 'jvp' | 'finite_difference';
	guidance_scale: number;
	seed: number;
	base_image_base64: string;
	/** 8-bit RGB: value = 128 + 127 * tangent / tangent_scale (tangent in image units [0,1] per unit CFG) */
	tangent_png_base64: string;
	tangent_scale: number;
	magnitude_heatmap_base64: string;
	mean_abs_tangent: number;
	performance_ms: number;
}

export interface AttentionMapRequest {
	base_request: Text2ImageRequest;
	words?: string[] | null;
	layers?: string;
	start?: number;
	end?: number;
}

export interface WordMap {
	word: string;
	token_positions: number[];
	attention_share: number;
	heatmap_base64: string;
	overlay_base64: string;
}

export interface AttentionMapResponse {
	image_base64: string;
	seed: number;
	maps: WordMap[];
	layers_recorded: string[];
	performance_ms: number;
}

export interface AblationRequest {
	base_request: Text2ImageRequest;
	mode?: 'remove_word' | 'mask_embedding';
	words?: string[] | null;
	max_words?: number;
	feature_model?: string | null;
}

export interface AblationResult {
	word: string;
	ablated_prompt?: string | null;
	global_distance: number;
	patch_distance: number;
	heatmap_base64: string;
	overlay_base64: string;
	ablated_image_base64: string;
}

export interface AblationResponse {
	base_image_base64: string;
	seed: number;
	mode: string;
	feature_model: string;
	feature_note?: string | null;
	results: AblationResult[];
	performance_ms: number;
}

export interface LayerInfo {
	name: string;
	kind: 'self' | 'cross' | 'joint' | string;
}

export interface LayersResponse {
	model_id: string;
	pipeline: string;
	layers: LayerInfo[];
	suggestions: string[];
}

export interface RuntimeStatus {
	queue_depth: number;
	current: { task: string; model_id: string; step: number; total_steps: number } | null;
}
// --- Lab --- ^^^
