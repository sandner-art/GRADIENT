// frontend/src/lib/lab/explore.svelte.ts
// The explorer's session, shared by the Lab tool and the Explore mode beside the render
// forms: the render being explored (settings + path of branches taken), the region, and
// the latest map, word suggestions and branches.

import { post } from '#lib/apiClient.js';
import { lab, runLabJob } from '#lib/lab/labState.svelte.js';
import type {
	ExploreBaseBody,
	ExploreBranch,
	ExploreBranchesResponse,
	ExploreBranchSpec,
	ExploreMapResponse,
	ExploreOp,
	ExploreStartResponse,
	ExploreTokensResponse,
	Text2ImageRequest
} from '#lib/types.js';
import { apiErrorMessage } from '#lib/utils/format.js';

export interface ExploreSession {
	base_request: Text2ImageRequest;
	init_image_base64: string | null;
	strength: number | null;
	path: ExploreOp[];
	/** where it came from, for the heading ("Lab settings", "image #12") */
	source: string;
}

export const explore = $state<{
	session: ExploreSession | null;
	start: ExploreStartResponse | null;
	/** the region as a PNG data URL (white = region); null = the whole image */
	mask: string | null;
	map: ExploreMapResponse | null;
	tokens: ExploreTokensResponse | null;
	branches: ExploreBranchesResponse | null;
	/** the step new branches start from */
	step: number;
	error: string | null;
}>({
	session: null,
	start: null,
	mask: null,
	map: null,
	tokens: null,
	branches: null,
	step: 0,
	error: null
});

/** Keys of saved settings that make a render request (as in "Use settings"). */
const REQUEST_KEYS = [
	'model_id',
	'prompt',
	'negative_prompt',
	'width',
	'height',
	'num_inference_steps',
	'guidance_scale',
	'seed',
	'scheduler_id',
	'loras',
	'embeddings',
	'vae_id',
	'quantization',
	'prompt_syntax'
] as const;

/** A render request from an image's saved parameters (null-safe, constant guidance). */
export function requestFromParams(params: Record<string, unknown>): Text2ImageRequest {
	const out: Record<string, unknown> = { num_outputs: 1 };
	for (const key of REQUEST_KEYS) if (params[key] !== undefined) out[key] = params[key];
	out.prompt_syntax ??= 'plain'; // renders before 0.10 read prompts as plain text
	if (typeof out.guidance_scale !== 'number') delete out.guidance_scale;
	if (out.quantization === 'none') out.quantization = null;
	return out as unknown as Text2ImageRequest;
}

/** The saved path of an explorer branch image (its parameters.explore_path). */
export function pathFromParams(params: Record<string, unknown>): ExploreOp[] {
	const path = params.explore_path;
	return Array.isArray(path) ? (path as ExploreOp[]) : [];
}

function body(): ExploreBaseBody {
	const s = explore.session;
	if (!s) throw new Error('Nothing to explore yet');
	return {
		base_request: s.base_request,
		init_image_base64: s.init_image_base64,
		strength: s.strength,
		path: s.path
	};
}

function clearResults(keepRegion = false) {
	explore.start = null;
	explore.map = null;
	explore.tokens = null;
	explore.branches = null;
	if (!keepRegion) explore.mask = null;
	explore.error = null;
}

/** Begin exploring a render (replaces the current session). */
export function openSession(session: ExploreSession) {
	explore.session = session;
	clearResults();
	explore.step = 0;
}

async function run<T>(label: string, fn: () => Promise<T>): Promise<T | null> {
	explore.error = null;
	return runLabJob(label, async () => {
		try {
			return await fn();
		} catch (e) {
			explore.error = apiErrorMessage(e);
			return null;
		}
	});
}

/** Render (or replay) the current end of the path and show how it formed. */
export async function loadImage(): Promise<boolean> {
	const res = await run('Rendering the image to explore…', () =>
		post<ExploreStartResponse>('/lab/explore/start', { ...body(), frames: 8 })
	);
	if (!res) return false;
	explore.start = res;
	if (explore.session) explore.session.base_request.seed = res.seed;
	explore.step = Math.min(explore.step || Math.round(res.steps * 0.4), res.steps - 1);
	return true;
}

/** Where can the region still move? Branches at several steps, measured. */
export async function measure(numSteps = 6, branches = 3, featureModel: string | null = null) {
	const res = await run(`Branching at ${numSteps} steps × ${branches}…`, () =>
		post<ExploreMapResponse>('/lab/explore/map', {
			...body(),
			mask_base64: explore.mask,
			num_steps: numSteps,
			branches,
			feature_model: featureModel
		})
	);
	if (res) {
		explore.map = res;
		explore.step = res.suggested.variations;
	}
}

export async function suggestWords(
	method: 'auto' | 'attention' | 'ablation' = 'auto',
	alternatives = 6
) {
	const res = await run('Finding the words that act here…', () =>
		post<ExploreTokensResponse>('/lab/explore/tokens', {
			...body(),
			mask_base64: explore.mask,
			method,
			alternatives
		})
	);
	if (res) explore.tokens = res;
}

export async function renderBranches(
	specs: ExploreBranchSpec[],
	featureModel: string | null = null
) {
	if (!specs.length) {
		explore.error = 'Pick at least one branch (variations, a word, a mix or a LoRA weight).';
		return;
	}
	const res = await run(`Rendering ${specs.length} branches from step ${explore.step}…`, () =>
		post<ExploreBranchesResponse>('/lab/explore/branches', {
			...body(),
			mask_base64: explore.mask,
			step: explore.step,
			branches: specs,
			feature_model: featureModel
		})
	);
	if (res) explore.branches = res;
}

/** Step into a branch: it becomes the image, the path grows by one. */
export async function continueFrom(branch: ExploreBranch) {
	if (!explore.session) return;
	explore.session.path = [...explore.session.path, branch.op];
	clearResults(true);
	await loadImage();
}

/** Go back to an earlier point of the path (0 = the original render). */
export async function backTo(length: number) {
	if (!explore.session) return;
	explore.session.path = explore.session.path.slice(0, length);
	clearResults(true);
	await loadImage();
}

export const busy = () => lab.busy !== null;

/**
 * Open a session for a saved image (its parameters), continuing its explorer path if it
 * is a branch. Image-to-image renders need their input image. Returns a problem or null.
 */
export function openFromParams(
	params: Record<string, unknown>,
	source: string,
	initImage: string | null = null
): string | null {
	const request = requestFromParams(params);
	if (!request.model_id || !request.prompt) return 'This image has no saved settings to explore.';
	if (params.custom_sigmas) return "Images rendered with custom sigmas can't be explored yet.";
	const strength = typeof params.strength === 'number' ? params.strength : null;
	const i2i = params.explore_image_to_image as { strength?: number } | undefined;
	const needsInit = strength !== null || i2i !== undefined;
	if (needsInit && !initImage)
		return 'This is an image-to-image render: explore it from the Image-to-Image preview, which has its input image.';
	openSession({
		base_request: request,
		init_image_base64: needsInit ? initImage : null,
		strength: needsInit ? (strength ?? i2i?.strength ?? null) : null,
		path: pathFromParams(params),
		source
	});
	return null;
}

/** Word replaced in a prompt (whole word, case-insensitive, first occurrence). */
export function replaceWord(prompt: string, word: string, replacement: string): string {
	const re = new RegExp(
		`(^|[^\\w'-])${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\w'-])`,
		'i'
	);
	return prompt.replace(re, (_m, pre: string) => `${pre}${replacement}`);
}
