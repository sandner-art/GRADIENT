// frontend/src/lib/gallery/api.ts
// Typed calls for the gallery/history API (src/gradient/api/endpoints/history.py).

import { del, get, patch, post } from '#lib/apiClient.js';
import type {
	GalleryDetail,
	GalleryFacets,
	GalleryFilter,
	GalleryPage,
	RunItem
} from '#lib/types.js';

export function filterQuery(filter: GalleryFilter, offset = 0, limit = 60): string {
	const params = new URLSearchParams();
	for (const [key, value] of Object.entries(filter)) {
		if (value === null || value === undefined || value === '' || value === false) continue;
		params.set(key, String(value));
	}
	params.set('offset', String(offset));
	params.set('limit', String(limit));
	return params.toString();
}

export const listOutputs = (filter: GalleryFilter, offset = 0, limit = 60) =>
	get<GalleryPage>(`/history/outputs?${filterQuery(filter, offset, limit)}`);

export const getOutput = (id: number) => get<GalleryDetail>(`/history/outputs/${id}`);

export const updateOutput = (
	id: number,
	changes: { rating?: number; favorite?: boolean; notes?: string; tags?: string[] }
) => patch<GalleryDetail>(`/history/outputs/${id}`, changes);

export const bulkAction = (
	ids: number[],
	action: 'rate' | 'favorite' | 'add_tags' | 'remove_tags' | 'delete',
	value?: unknown
) => post<{ changed: number }>('/history/outputs/bulk', { ids, action, value });

export const deleteOutput = (id: number) => del<{ removed: number }>(`/history/outputs/${id}`);

export const listRuns = (offset = 0, limit = 40, runType?: string | null) =>
	get<{ items: RunItem[]; total: number }>(
		`/history/runs?offset=${offset}&limit=${limit}${runType ? `&run_type=${encodeURIComponent(runType)}` : ''}`
	);

export const deleteRun = (runUuid: string) =>
	del<{ removed: number }>(`/history/runs/${encodeURIComponent(runUuid)}`);

export const getFacets = () => get<GalleryFacets>('/history/facets');

export const thumbUrl = (id: number, size = 256) =>
	`/api/v1/history/outputs/${id}/thumb?size=${size}`;
export const imageUrl = (id: number) => `/api/v1/history/outputs/${id}/image`;
export const downloadUrl = (id: number) => `/api/v1/history/outputs/${id}/image?download=true`;

/** Human labels for run types. */
export function runTypeLabel(runType: string | null | undefined): string {
	const labels: Record<string, string> = {
		text_to_image: 'Text-to-image',
		image_to_image: 'Image-to-image',
		xyz_grid: 'XYZ grid',
		lab_taylor: 'Lab · Taylor',
		lab_attention: 'Lab · attention',
		lab_concept_attention: 'Lab · ConceptAttention',
		lab_token_ablation: 'Lab · ablation'
	};
	return labels[runType ?? ''] ?? (runType || 'Unknown');
}
