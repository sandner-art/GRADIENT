// frontend/src/lib/lab/layers.svelte.ts
// Attention layer names per model, fetched on demand (listing loads the model).

import { get } from '#lib/apiClient.js';
import type { LayersResponse } from '#lib/types.js';

const cache = $state<Record<string, LayersResponse>>({});

export function cachedLayers(modelId: string): LayersResponse | null {
	return cache[modelId] ?? null;
}

export async function fetchLayers(modelId: string): Promise<LayersResponse> {
	const hit = cache[modelId];
	if (hit) return hit;
	const r = await get<LayersResponse>(`/lab/layers?model_id=${encodeURIComponent(modelId)}`);
	cache[modelId] = r;
	return r;
}
