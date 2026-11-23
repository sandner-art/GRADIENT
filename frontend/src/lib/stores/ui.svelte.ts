// frontend/src/lib/stores/ui.svelte.ts
// App-wide UI state: active tab, the image viewer, notices, gallery filter, and
// "use these settings" from the gallery into the render forms.

import type { GalleryFilter, ImageToImageRequest, Text2ImageRequest } from '#lib/types.js';
import { i2iFormStore } from '#lib/stores/i2iFormStore.js';
import { t2iFormStore } from '#lib/stores/t2iFormStore.js';

export type Tab = 't2i' | 'i2i' | 'grid' | 'gallery' | 'lab' | 'chat' | 'models';

export interface Toast {
	id: number;
	text: string;
	kind: 'info' | 'success' | 'error';
	action?: { label: string; run: () => void };
}

export const ui = $state<{
	tab: Tab;
	queueOpen: boolean;
	/** image ids the viewer pages through, and the current position */
	viewer: { ids: number[]; index: number } | null;
	galleryFilter: GalleryFilter;
	/** bumped to make the gallery reload (e.g. after deletions elsewhere) */
	galleryVersion: number;
	toasts: Toast[];
}>({
	tab: 't2i',
	queueOpen: false,
	viewer: null,
	galleryFilter: { sort: 'newest' },
	galleryVersion: 0,
	toasts: []
});

export function goTo(tab: Tab) {
	ui.tab = tab;
	if (typeof window !== 'undefined') window.scrollTo({ top: 0, behavior: 'smooth' });
}

export function openViewer(ids: number[], index = 0) {
	if (ids.length)
		ui.viewer = { ids: [...ids], index: Math.max(0, Math.min(index, ids.length - 1)) };
}

export function closeViewer() {
	ui.viewer = null;
}

export function openGallery(filter: GalleryFilter = {}) {
	ui.galleryFilter = { sort: 'newest', ...filter };
	goTo('gallery');
}

let toastId = 0;
export function notify(
	text: string,
	kind: Toast['kind'] = 'info',
	action?: Toast['action'],
	ms = 6000
) {
	const toast: Toast = { id: ++toastId, text, kind, action };
	ui.toasts = [...ui.toasts.slice(-3), toast];
	if (ms > 0) setTimeout(() => dismiss(toast.id), ms);
}

export function dismiss(id: number) {
	ui.toasts = ui.toasts.filter((t) => t.id !== id);
}

// ---- "Use settings" from a saved image
const COPY_KEYS = [
	'model_id',
	'prompt',
	'negative_prompt',
	'width',
	'height',
	'num_inference_steps',
	'guidance_scale',
	'scheduler_id',
	'custom_sigmas',
	'loras',
	'embeddings',
	'vae_id'
] as const;

/** Settings of a generation, ready for a form store (seed optional). */
export function settingsFrom(
	params: Record<string, unknown>,
	keepSeed: boolean
): Partial<Text2ImageRequest> {
	const out: Record<string, unknown> = {};
	for (const key of COPY_KEYS) if (key in params) out[key] = params[key];
	if (typeof out.guidance_scale !== 'number') delete out.guidance_scale; // scheduled CFG strings
	out.seed = keepSeed ? ((params.seed as number | null) ?? null) : null;
	return out as Partial<Text2ImageRequest>;
}

export function useSettings(params: Record<string, unknown>, keepSeed: boolean) {
	const settings = settingsFrom(params, keepSeed);
	t2iFormStore.update((s) => ({ ...s, ...settings }));
	goTo('t2i');
	notify(keepSeed ? 'Settings and seed loaded' : 'Settings loaded with a random seed', 'success');
}

export async function useAsInitImage(imageUrl: string, params: Record<string, unknown>) {
	const blob = await (await fetch(imageUrl)).blob();
	const dataUrl: string = await new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => resolve(String(reader.result));
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(blob);
	});
	const settings = settingsFrom(params, false);
	i2iFormStore.update((s) => ({
		...s,
		...(settings as Partial<ImageToImageRequest>),
		image_base64: dataUrl
	}));
	goTo('i2i');
	notify('Image loaded as the image-to-image input', 'success');
}
