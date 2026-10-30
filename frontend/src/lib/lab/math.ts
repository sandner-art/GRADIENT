// frontend/src/lib/lab/math.ts
// Pure helpers for the Lab tab (unit-tested in math.test.ts).

import type { LayersResponse, ModulationLane } from '#lib/types.js';

/**
 * Taylor preview in the browser: image(w + Δ) ≈ image(w) + Δ · ∂image/∂w.
 *
 * `base` and `tangent` are RGBA pixel arrays of the same size. The tangent PNG stores
 * 128 + 127 · t / scale per channel, where t is in image units ([0, 1]) per unit of CFG.
 */
export function composeTaylor(
	base: Uint8ClampedArray,
	tangent: Uint8ClampedArray,
	scale: number,
	delta: number,
	out: Uint8ClampedArray<ArrayBuffer> = new Uint8ClampedArray(base.length)
): Uint8ClampedArray<ArrayBuffer> {
	const k = (delta * scale * 255) / 127;
	for (let i = 0; i < base.length; i += 4) {
		out[i] = base[i] + k * (tangent[i] - 128);
		out[i + 1] = base[i + 1] + k * (tangent[i + 1] - 128);
		out[i + 2] = base[i + 2] + k * (tangent[i + 2] - 128);
		out[i + 3] = 255;
	}
	return out;
}

/** Mean absolute RGB difference of two RGBA arrays, in 0–255 units. */
export function meanAbsDiff(a: Uint8ClampedArray, b: Uint8ClampedArray): number {
	if (a.length !== b.length || a.length === 0) return NaN;
	let sum = 0;
	for (let i = 0; i < a.length; i += 4) {
		sum += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]);
	}
	return sum / ((a.length / 4) * 3);
}

/** "cat, red  hat,,moon" -> ["cat", "red hat", "moon"]; empty -> null (= all words). */
export function parseWordList(text: string): string[] | null {
	const words = text
		.split(',')
		.map((w) => w.trim().replace(/\s+/g, ' '))
		.filter(Boolean);
	return words.length ? words : null;
}

/** Steps [first, last) covered by a lane; same rounding as the backend's step_range. */
export function laneSteps(start: number, end: number, total: number): [number, number] {
	const first = Math.floor(start * total + 1e-9);
	const last = Math.ceil(end * total - 1e-9);
	return [first, Math.max(first, last)];
}

/** Client-side check that mirrors the backend validator; returns an error message or null. */
export function laneProblem(lane: ModulationLane, loraCount: number): string | null {
	if (!(lane.end > lane.start)) return 'end must be greater than start';
	if (lane.start < 0 || lane.end > 1) return 'start/end must be between 0 and 1';
	if (lane.type !== 'prompt' && (lane.value === null || lane.value === undefined))
		return 'needs a value';
	if (lane.type === 'prompt' && !lane.prompt?.trim()) return 'needs a prompt';
	if (lane.type === 'attention_scale' && !lane.targets?.trim()) return 'needs target layers';
	if (lane.type === 'lora_weight') {
		if (loraCount === 0) return 'add a LoRA in the base settings first';
		if (lane.lora_index == null || lane.lora_index >= loraCount) return 'pick a LoRA';
	}
	return null;
}

/** Only the fields the backend expects for this lane type. */
export function cleanLane(lane: ModulationLane): ModulationLane {
	const out: ModulationLane = { type: lane.type, start: lane.start, end: lane.end };
	if (lane.type === 'prompt') {
		out.prompt = lane.prompt ?? '';
		out.mode = lane.mode ?? 'replace';
		if (out.mode === 'blend') out.value = lane.value ?? 0.5;
	} else {
		out.value = lane.value ?? null;
	}
	if (lane.type === 'attention_scale') out.targets = lane.targets ?? '';
	if (lane.type === 'lora_weight') out.lora_index = lane.lora_index ?? 0;
	return out;
}

/** Number of layers a comma-separated glob pattern selects (same semantics as fnmatch). */
export function countMatches(layers: LayersResponse | null, pattern: string): number | null {
	if (!layers) return null;
	const globs = pattern
		.split(',')
		.map((p) => p.trim())
		.filter(Boolean)
		.map(globToRegExp);
	if (!globs.length) return 0;
	return layers.layers.filter((l) => globs.some((g) => g.test(l.name))).length;
}

export function globToRegExp(glob: string): RegExp {
	let re = '';
	for (let i = 0; i < glob.length; i++) {
		const c = glob[i];
		if (c === '*') re += '.*';
		else if (c === '?') re += '.';
		else if (c === '[') {
			const close = glob.indexOf(']', i + 1);
			if (close === -1) re += '\\[';
			else {
				let body = glob.slice(i + 1, close);
				if (body.startsWith('!')) body = '^' + body.slice(1);
				re += `[${body.replace(/\\/g, '\\\\')}]`;
				i = close;
			}
		} else re += c.replace(/[.+^${}()|\\]/g, '\\$&');
	}
	return new RegExp(`^${re}$`);
}

/** Families with double-stream blocks that ConceptAttention supports. */
export const CONCEPT_ATTENTION_FAMILIES = ['flux1', 'flux2', 'sd3'];
/** UNet families with separate cross-attention layers. */
export const CROSS_ATTENTION_FAMILIES = ['sd15', 'sdxl'];

/** Which attention tool fits a model family; null when neither is known to work. */
export function attentionMethodFor(family: string | null | undefined): 'concept' | 'cross' | null {
	if (!family) return null;
	if (CONCEPT_ATTENTION_FAMILIES.includes(family)) return 'concept';
	if (CROSS_ATTENTION_FAMILIES.includes(family)) return 'cross';
	return null;
}
