// frontend/src/lib/gallery/state.svelte.ts
// Loaded gallery page, shared by the grid and the viewer so edits show everywhere at once.

import type { GalleryFacets, GalleryItem } from '#lib/types.js';

export const gallery = $state<{
	items: GalleryItem[];
	total: number;
	nextOffset: number | null;
	loading: boolean;
	facets: GalleryFacets | null;
}>({ items: [], total: 0, nextOffset: 0, loading: false, facets: null });

export function patchItem(id: number, changes: Partial<GalleryItem>) {
	const i = gallery.items.findIndex((it) => it.id === id);
	if (i >= 0) gallery.items[i] = { ...gallery.items[i], ...changes };
}

export function removeItems(ids: number[]) {
	const before = gallery.items.length;
	gallery.items = gallery.items.filter((it) => !ids.includes(it.id));
	gallery.total = Math.max(0, gallery.total - (before - gallery.items.length));
}
