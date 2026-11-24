// frontend/src/lib/sweeps.ts
// Test sweeps from ArtAgents as files (src/gradient/api/endpoints/sweeps.py).

import { get, post } from '#lib/apiClient.js';
import type { SweepPreview, SweepQueued, SweepStatus } from '#lib/types.js';

export const previewSweep = (sweep_text: string, lora_map: Record<string, string> = {}) =>
	post<SweepPreview>('/sweeps/preview', { sweep_text, lora_map });

export const queueSweep = (sweep_text: string, lora_map: Record<string, string> = {}) =>
	post<SweepQueued>('/sweeps', { sweep_text, lora_map });

export const listSweeps = () => get<{ items: SweepStatus[] }>('/sweeps');

export const writeSweepResults = (id: string) =>
	post<{ results_dir: string; images: number; errors: number }>(`/sweeps/${id}/results`, {});

/** How a LoRA was found, as a short mark and whether the sweep can use it. */
export function loraState(status: string): { mark: string; ok: boolean } {
	if (status === 'ok' || status === 'found_by_hash') return { mark: '✓', ok: true };
	if (status === 'unverified') return { mark: '?', ok: true };
	return { mark: '✗', ok: false };
}

/** "12/40 done · 2 failed" for a sweep's job counts. */
export function sweepProgress(s: Pick<SweepStatus, 'counts' | 'images'>): string {
	const c = s.counts;
	const finished = (c.completed ?? 0) + (c.failed ?? 0) + (c.cancelled ?? 0) + (c.missing ?? 0);
	const parts = [`${finished}/${s.images} done`];
	if (c.failed) parts.push(`${c.failed} failed`);
	if (c.cancelled) parts.push(`${c.cancelled} cancelled`);
	return parts.join(' · ');
}
