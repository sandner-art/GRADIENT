// frontend/src/lib/gallery/api.ts
// Typed calls for the gallery/history API (src/gradient/api/endpoints/history.py).

import { del, get, patch, post } from '#lib/apiClient.js';
import type {
	ExportOptions,
	ExportResult,
	GalleryDetail,
	GalleryFacets,
	GalleryFilter,
	GalleryPage,
	ProjectInfo,
	RunItem,
	SyncReport
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

// ---- project folders (src/gradient/api/endpoints/projects.py)
export const listProjects = () => get<{ items: ProjectInfo[]; root: string }>('/projects');

/** The gallery filter as the export endpoint takes it (no sort, no empty values). */
export function exportFilter(filter: GalleryFilter): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(filter)) {
		if (key === 'sort' || value === null || value === undefined || value === '' || value === false)
			continue;
		out[key] = value;
	}
	return out;
}

export const exportProject = (
	body: ({ ids: number[] } | { filter: Record<string, unknown> }) &
		({ project: string } | { name: string; path?: string | null }) &
		Partial<ExportOptions>
) => post<ExportResult>('/projects/export', body);

export const syncProject = (uuid: string, captionSource: string, captionsAsTags: boolean) =>
	post<SyncReport>(`/projects/${uuid}/sync`, {
		caption_source: captionSource,
		captions_as_tags: captionsAsTags
	});

export const openProjectFolder = (path: string, captionSource = 'artagents') =>
	post<SyncReport>('/projects/open', { path, caption_source: captionSource });

export const forgetProject = (uuid: string) => del<{ forgotten: string }>(`/projects/${uuid}`);

/** Where a tag came from, for people. */
export function tagSourceLabel(source: string): string {
	if (source === 'manual') return 'yours';
	if (source === 'xmp') return 'from an XMP sidecar (Kurator, digiKam, Lightroom…)';
	return `from ${source}`;
}

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
