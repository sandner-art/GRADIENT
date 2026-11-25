// frontend/src/lib/types.ts

// --- From diffusion/models.py ---
export interface ImageResult {
	image_base64: string;
	seed: number;
	parameters: Record<string, any>; // Dictionary for parameters used
}

export type PromptSyntax = 'a1111' | 'comfy' | 'plain';

export interface DiffusionTaskBase {
	model_id: string;
	prompt: string;
	negative_prompt?: string | null; // Allow null
	prompt_syntax?: PromptSyntax;
	styles?: string[] | null;
	style_position?: 'template' | 'before' | 'after' | 'chunk' | 'blocks';
	style_start?: number;
	tiling?: 'none' | 'x' | 'y' | 'xy';
	height?: number;
	width?: number;
	num_inference_steps?: number;
	guidance_scale?: number; // Allow float, string or list for scheduled CFG later? For now, number.
	seed?: number | null;
	custom_sigmas?: number[] | null;
	scheduler_id?: string | null; // Added scheduler
	loras?: LoRAParameter[] | null;
	embeddings?: string[] | null;
	vae_id?: string | null;
	/** none, bnb-8bit, bnb-4bit, torchao-int8, torchao-int8-dynamic, torchao-fp8, gguf:<file>; '+text' also quantizes text encoders */
	quantization?: string | null;
	modulation?: ModulationLane[] | null;
	// vae_id?: string | null; // Add when ready
}

export interface LoRAParameter {
	name: string;
	weight?: number;
}

/** One lane on the denoising timeline, active for steps in [start, end) (fractions of the run). */
export type ModulationLaneType =
	'guidance' | 'prompt' | 'attention_scale' | 'lora_weight' | 'prompt_injection';
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
	/** e.g. why 'auto' fell back from JVP to finite differences */
	note?: string | null;
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
	/** Double-stream blocks usable by ConceptAttention (0 = not a supported DiT). */
	concept_blocks: number;
}

export interface ConceptAttentionRequest {
	base_request: Text2ImageRequest;
	concepts?: string[] | null;
	blocks?: string | null;
	start?: number;
	end?: number;
}

export interface ConceptMap {
	concept: string;
	color: string;
	/** Fraction of image patches where this concept wins. */
	area: number;
	mean_probability: number;
	heatmap_base64: string;
	overlay_base64: string;
}

export interface ConceptAttentionResponse {
	image_base64: string;
	seed: number;
	segmentation_base64: string;
	concepts: ConceptMap[];
	blocks_recorded: number[];
	num_blocks: number;
	steps_recorded: number;
	performance_ms: number;
}

export interface RuntimeStatus {
	queue_depth: number;
	current: { task: string; model_id: string; step: number; total_steps: number } | null;
}
// --- Lab --- ^^^

// --- Queue (src/gradient/core/jobs.py) --- vvv
export type QueueJobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
export type JobKind =
	| 'text_to_image'
	| 'image_to_image'
	| 'xyz_grid'
	| 'lab_taylor'
	| 'lab_attention'
	| 'lab_concept_attention'
	| 'lab_token_ablation';

export interface JobOutputRef {
	id: number | null;
	seed?: number;
}

export interface QueueJob {
	id: string;
	kind: JobKind | string;
	label: string;
	status: QueueJobStatus;
	priority: number;
	created: number;
	started: number | null;
	finished: number | null;
	/** step/total for the current render; item/items for grids */
	progress: {
		step?: number;
		total?: number;
		item?: number;
		items?: number;
		phase?: string;
		/** counts live previews (GET /queue/{id}/preview); preview_step is the step shown */
		preview?: number;
		preview_step?: number;
	};
	run_uuid: string | null;
	error: string | null;
	result: {
		run_uuid?: string;
		outputs?: JobOutputRef[];
		performance_ms?: number;
		loras_skipped?: string[];
		embeddings_skipped?: string[];
	} | null;
	position: number | null;
	retryable: boolean;
	model_id: string | null;
	/** the job's settings (input images left out) */
	payload?: Record<string, unknown>;
}

export type QueueEvent =
	| { type: 'snapshot'; jobs: QueueJob[] }
	| { type: 'job'; job: QueueJob }
	| { type: 'removed'; ids: string[] }
	| { type: 'order'; order: string[] }
	| { type: 'shutdown' };
// --- Queue --- ^^^

// --- Gallery (src/gradient/gallery/store.py) --- vvv
export interface GalleryTag {
	tag: string;
	source: 'manual' | 'kurator' | 'artagents' | 'ai' | string;
	confidence: number | null;
}

export interface GalleryItem {
	id: number;
	run_uuid: string | null;
	run_type: string | null;
	output_type: string;
	created: string | null;
	model_id: string | null;
	prompt: string | null;
	seed: number | null;
	width: number | null;
	height: number | null;
	rating: number | null;
	favorite: boolean;
	notes: string | null;
	tags: GalleryTag[];
	file_name: string;
	image_url: string;
	thumb_url: string;
}

export interface GalleryDetail extends GalleryItem {
	parameters: Record<string, unknown>;
	negative_prompt: string | null;
	file_path: string;
	exists: boolean;
	run_status: string | null;
	/** captions from other tools, one per source label */
	captions: { source: string; text: string; updated: string | null }[];
	/** project folders this image was exported to */
	projects: { uuid: string; name: string; file_name: string }[];
}

export interface GalleryPage {
	items: GalleryItem[];
	total: number;
	offset: number;
	next_offset: number | null;
}

export interface RunItem {
	run_uuid: string;
	run_type: string;
	status: string;
	started: string | null;
	finished: string | null;
	model_id: string | null;
	prompt: string | null;
	/** Group runs (jobs queued with the same `group`): the group's name */
	title?: string | null;
	count: number;
	cover_ids: number[];
	error: string | null;
}

export interface GalleryFacets {
	models: { model_id: string; count: number }[];
	run_types: { run_type: string; count: number }[];
	tags: { tag: string; count: number }[];
	projects: { uuid: string; name: string; count: number }[];
	total: number;
	favorites: number;
}

/** A project folder shared with other tools (Kurator, ArtAgents, digiKam ...). */
export interface ProjectInfo {
	uuid: string;
	name: string;
	path: string;
	created: string | null;
	updated: string | null;
	last_sync: string | null;
	count: number;
	exists: boolean;
	options: Partial<ExportOptions>;
}

export interface ExportOptions {
	sidecars: boolean;
	model_keywords: boolean;
	captions: 'none' | 'prompt';
	link: boolean;
}

export interface SyncReport {
	files: number;
	renamed: { from: string; to: string }[];
	missing: string[];
	not_in_gallery: number;
	tags_added: number;
	tags_removed: number;
	ratings_changed: number;
	captions: number;
	unreadable_sidecars: string[];
	unknown_files: string[];
	unknown_count: number;
	project: ProjectInfo;
}

export interface ExportResult {
	added: number;
	updated: number;
	missing_source: number[];
	kept_foreign_sidecars: string[];
	kept_captions: number;
	errors: string[];
	sync?: SyncReport;
	project: ProjectInfo;
}

export interface GalleryFilter {
	run?: string | null;
	model?: string | null;
	run_type?: string | null;
	q?: string;
	min_rating?: number | null;
	favorite?: boolean | null;
	tag?: string | null;
	project?: string | null;
	sort?: 'newest' | 'oldest' | 'rating';
}
// --- Gallery --- ^^^

/** A way to load a model (GET /system/quantization). */
export interface QuantOption {
	value: string;
	label: string;
	about: string;
	backend: string | null;
	available: boolean;
	reason: string | null;
	/** a few words for a disabled option ("other model", "needs GPU", ...) */
	short?: string | null;
}

export interface QuantVariantReport {
	quantization: string;
	label: string;
	error: string | null;
	weights_mb: Record<string, number>;
	denoiser_mb: number | null;
	load_seconds: number | null;
	seconds_per_image: number | null;
	peak_memory_mb: number | null;
	psnr: number | null;
	ssim: number | null;
	mean_abs_diff: number | null;
	feature_distance: number | null;
	relative_change: number | null;
	taylor_cosine: number | null;
	taylor_method: string | null;
	verdict: string;
	strip_base64: string | null;
}

export interface QuantReportResponse {
	model_id: string;
	seeds: number[];
	reference: string;
	seed_spread: number | null;
	feature_model: string | null;
	feature_note: string | null;
	variants: QuantVariantReport[];
	image_base64: string;
	summary: string;
	performance_ms: number;
}

// ---- Explorer (/lab/explore/*)
export type ExploreKind = 'seed' | 'prompt' | 'mix' | 'lora';

export interface ExploreOp {
	step: number;
	mask_base64: string | null;
	kind: ExploreKind;
	seed: number;
	prompt: string | null;
	mix_prompt: string | null;
	mix_weight: number;
	lora_weight: number | null;
	label: string | null;
}

export interface ExploreBaseBody {
	base_request: Text2ImageRequest;
	init_image_base64: string | null;
	strength: number | null;
	path: ExploreOp[];
}

export interface ExploreFrame {
	step: number;
	sigma: number;
	image_base64: string;
	still_changing_base64: string;
	still_changing: number;
}

export interface ExploreStartResponse {
	image_base64: string;
	seed: number;
	steps: number;
	scheduler: string;
	prompt: string;
	frames: ExploreFrame[];
	note: string | null;
	performance_ms: number;
}

export interface ExploreMapPoint {
	step: number;
	sigma: number;
	change: number;
	distance: number;
	spread: number;
	estimate: number | null;
	images_base64: string[];
	heatmap_base64: string;
}

export interface ExploreMapResponse {
	image_base64: string;
	reference_image_base64: string;
	seed: number;
	points: ExploreMapPoint[];
	window: number[] | null;
	suggested: Record<'big' | 'variations' | 'details', number>;
	estimate_error: number | null;
	estimate_correlation: number | null;
	metric: 'dino' | 'pixels';
	feature_model: string | null;
	feature_note: string | null;
	summary: string;
	performance_ms: number;
}

export interface TokenSuggestion {
	word: string;
	score: number;
	strength: number;
	lift: number;
	alternatives: string[];
	overlay_base64: string | null;
}

export interface ExploreTokensResponse {
	image_base64: string;
	prompt: string;
	method: string;
	words: TokenSuggestion[];
	note: string | null;
	performance_ms: number;
}

export interface ExploreBranchSpec {
	kind: ExploreKind;
	seed?: number | null;
	prompt?: string | null;
	mix_prompt?: string | null;
	mix_weight?: number;
	lora_weight?: number | null;
	label?: string | null;
}

export interface ExploreBranch {
	label: string;
	op: ExploreOp;
	image_base64: string;
	parameters: Record<string, unknown>;
	change: number | null;
	outside_change: number | null;
}

export interface ExploreBranchesResponse {
	reference_image_base64: string;
	seed: number;
	step: number;
	sigma: number;
	branches: ExploreBranch[];
	metric: 'dino' | 'pixels';
	feature_note: string | null;
	performance_ms: number;
}

/** A LoRA of an imported test sweep and where GRADIENT found it (src/gradient/gallery/sweeps.py). */
export interface SweepLora {
	key: string;
	name: string;
	sha256: string | null;
	resolved: string | null;
	status: 'ok' | 'found_by_hash' | 'unverified' | 'missing' | 'hash_mismatch';
	message: string;
}

/** POST /sweeps/preview: what importing an ArtAgents test sweep would do. */
export interface SweepPreview {
	id: string;
	sha256: string;
	name: string;
	run: string;
	model_id: string;
	model_found: boolean;
	model_defaults: { steps?: number; guidance_scale?: number };
	options: Record<string, string | number>;
	images: number;
	prompts: number;
	conditions: number;
	loras: SweepLora[];
	ready: boolean;
	problems: string[];
	results_dir: string;
}

export interface SweepQueued extends SweepPreview {
	group: string;
	run_uuid: string;
	queued: number;
}

/** GET /sweeps: an imported sweep with its jobs' states. */
export interface SweepStatus {
	id: string;
	name: string;
	run: string;
	model_id: string;
	run_uuid: string;
	results_dir: string;
	written: number | null;
	imported: number;
	images: number;
	counts: Record<string, number>;
	done: boolean;
}
