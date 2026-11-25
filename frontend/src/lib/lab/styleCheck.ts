// frontend/src/lib/lab/styleCheck.ts
// Lab → Style check: request and result types, sorting and the scatter's scale.

import type { StylePosition } from '#lib/styles.js';
import type { Text2ImageRequest } from '#lib/types.js';

export interface StyleCheckRequest {
	base_request: Text2ImageRequest;
	styles: string[];
	positions: StylePosition[];
	style_starts: number[];
	num_seeds: number;
	feature_model: string | null;
	clip_model: string | null;
}

export interface StyleCheckCell {
	style: string | null;
	position: string | null;
	start: number | null;
	seed: number;
	image_base64: string;
	parameters: Record<string, unknown>;
}

export interface StyleCheckRow {
	style: string;
	position: StylePosition;
	start: number;
	strength: number;
	strength_rel: number | null;
	content: number | null;
	content_rel: number | null;
	consistency: number | null;
	tokens: number | null;
	chunks: number | null;
	notes: string[];
	cells: number[];
}

export interface StyleCheckResponse {
	model_id: string;
	prompt: string;
	seeds: number[];
	feature_model: string;
	clip_model: string | null;
	seed_distance: number | null;
	baseline_content: number | null;
	baseline_consistency: number | null;
	baseline: number[];
	rows: StyleCheckRow[];
	cells: StyleCheckCell[];
	image_base64: string;
	notes: string[];
	elapsed_s: number;
}

export type SortKey = 'strength' | 'content' | 'consistency' | 'order';

/** Strength on the scale where 1 = as different as a new seed (raw distance if unknown). */
export const strengthOf = (r: StyleCheckRow) => r.strength_rel ?? r.strength;

export function sortRows(rows: StyleCheckRow[], key: SortKey): StyleCheckRow[] {
	if (key === 'order') return rows;
	const value = (r: StyleCheckRow) =>
		key === 'strength'
			? strengthOf(r)
			: key === 'content'
				? (r.content_rel ?? -Infinity)
				: (r.consistency ?? -Infinity);
	return [...rows].sort((a, b) => value(b) - value(a));
}

/** "0, 0.3" -> [0, 0.3]; values outside [0, 0.95] or not numbers are dropped (max 4). */
export function parseStarts(text: string): number[] {
	const out: number[] = [];
	for (const part of text.split(',')) {
		if (!part.trim()) continue;
		const v = Number(part.trim());
		if (Number.isFinite(v) && v >= 0 && v <= 0.95 && !out.includes(v)) out.push(v);
	}
	return (out.length ? out : [0]).slice(0, 4);
}

/** A nice upper bound for an axis: covers the data and the reference value 1. */
export function axisMax(values: number[]): number {
	const top = Math.max(1, ...values.filter((v) => Number.isFinite(v)));
	const step = top <= 1.2 ? 0.2 : top <= 3 ? 0.5 : 1;
	return Number((Math.ceil((top * 1.05) / step) * step).toFixed(2));
}

export const labelOf = (r: StyleCheckRow) =>
	`${r.style} · ${r.position}${r.start ? ` · from ${r.start}` : ''}`;
