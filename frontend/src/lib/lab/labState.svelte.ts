// frontend/src/lib/lab/labState.svelte.ts
// Settings shared by every Lab tool. Module-level state, so it survives tab switches.

import type { ModulationLane, Text2ImageRequest } from '#lib/types.js';

export interface LabBase {
	model_id: string;
	prompt: string;
	negative_prompt: string;
	width: number;
	height: number;
	num_inference_steps: number;
	guidance_scale: number;
	seed: number | null;
	scheduler_id: string | null;
	lora: string | null;
	lora_weight: number;
}

export const lab = $state<{ base: LabBase; lanes: ModulationLane[]; busy: string | null }>({
	base: {
		model_id: '',
		prompt: 'a red lighthouse on a rocky coast at sunset, dramatic clouds',
		negative_prompt: 'blurry, low quality',
		width: 512,
		height: 512,
		num_inference_steps: 25,
		guidance_scale: 7,
		seed: 1234,
		scheduler_id: null,
		lora: null,
		lora_weight: 1
	},
	/** Structural modulation lanes being edited in the Modulation tool. */
	lanes: [{ type: 'prompt', start: 0, end: 0.3, prompt: 'a castle on a hill', mode: 'replace' }],
	/** Label of the running Lab job, or null when idle (one job at a time). */
	busy: null
});

/** Problems that would make every tool fail, so the UI can say so before calling the API. */
export function baseProblem(): string | null {
	const b = lab.base;
	if (!b.model_id) return 'Pick a model first.';
	if (!b.prompt.trim()) return 'The prompt is empty.';
	if (b.width % 8 || b.height % 8 || b.width <= 0 || b.height <= 0)
		return 'Width and height must be positive multiples of 8.';
	if (!(b.num_inference_steps > 0)) return 'Steps must be positive.';
	return null;
}

export function baseRequest(overrides: Partial<Text2ImageRequest> = {}): Text2ImageRequest {
	const b = lab.base;
	return {
		model_id: b.model_id,
		prompt: b.prompt.trim(),
		negative_prompt: b.negative_prompt.trim() || null,
		width: b.width,
		height: b.height,
		num_inference_steps: b.num_inference_steps,
		guidance_scale: b.guidance_scale,
		seed: b.seed,
		scheduler_id: b.scheduler_id,
		loras: b.lora ? [{ name: b.lora, weight: b.lora_weight }] : null,
		num_outputs: 1,
		...overrides
	};
}

/** Runs one Lab job at a time; returns null without running when another job is in progress. */
export async function runLabJob<T>(label: string, job: () => Promise<T>): Promise<T | null> {
	if (lab.busy) return null;
	lab.busy = label;
	try {
		return await job();
	} finally {
		lab.busy = null;
	}
}
