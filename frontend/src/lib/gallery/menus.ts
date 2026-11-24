// frontend/src/lib/gallery/menus.ts
// Right-click menus for images and queue jobs (shown by components/ContextMenu.svelte).

import { deleteOutput, downloadUrl, getOutput, imageUrl, updateOutput } from './api.js';
import { patchItem, removeItems } from './state.svelte.js';
import { cancelJob, moveJob, outputIds, retryJob } from '#lib/stores/queue.svelte.js';
import {
	closeViewer,
	notify,
	openGallery,
	openViewer,
	sendToLab,
	ui,
	useAsInitImage,
	useSettings,
	type MenuItem
} from '#lib/stores/ui.svelte.js';
import type { GalleryDetail, QueueJob } from '#lib/types.js';
import { apiErrorMessage } from '#lib/utils/format.js';

async function copy(text: string, what: string) {
	try {
		await navigator.clipboard.writeText(text);
		notify(`${what} copied`, 'success', undefined, 1500);
	} catch {
		notify('The browser blocked clipboard access', 'error');
	}
}

function download(url: string) {
	const a = document.createElement('a');
	a.href = url;
	a.download = '';
	document.body.appendChild(a);
	a.click();
	a.remove();
}

/** Run an action that needs the image's saved settings; reports errors as a notice. */
function withDetail(id: number, fn: (d: GalleryDetail) => unknown) {
	return async () => {
		try {
			await fn(await getOutput(id));
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		}
	};
}

export interface ImageMenuOptions {
	/** images to page through when opening the viewer (default: just this one) */
	ids?: number[];
	rating?: number | null;
	favorite?: boolean;
	/** extra entries shown after "Open" (e.g. Select, Compare) */
	extra?: MenuItem[];
	/** extra entries shown before Delete (e.g. Export to a project…) */
	more?: MenuItem[];
	inViewer?: boolean;
}

export function imageMenu(id: number, opts: ImageMenuOptions = {}): MenuItem[] {
	const ids = opts.ids?.length ? opts.ids : [id];
	const leaveViewer = () => {
		if (opts.inViewer) closeViewer();
	};
	const items: MenuItem[] = [];
	if (!opts.inViewer) items.push({ label: 'Open', run: () => openViewer(ids, ids.indexOf(id)) });
	items.push(...(opts.extra ?? []));
	if (items.length) items.push({ separator: true });
	items.push(
		{
			label: 'Use settings',
			run: withDetail(id, (d) => {
				leaveViewer();
				useSettings(d.parameters, true);
			})
		},
		{
			label: 'Use settings with a new seed',
			run: withDetail(id, (d) => {
				leaveViewer();
				useSettings(d.parameters, false);
			})
		},
		{
			label: 'Use as input (image-to-image)',
			run: withDetail(id, async (d) => {
				leaveViewer();
				await useAsInitImage(imageUrl(id), d.parameters);
			})
		},
		{
			label: 'Send to Lab',
			run: withDetail(id, (d) => {
				leaveViewer();
				sendToLab(d.parameters);
			})
		},
		{ separator: true },
		{
			rating: opts.rating ?? null,
			rate: async (stars) => {
				try {
					const d = await updateOutput(id, { rating: stars });
					patchItem(id, { rating: d.rating });
					ui.galleryVersion++;
				} catch (e) {
					notify(apiErrorMessage(e), 'error');
				}
			}
		},
		{
			label: opts.favorite ? 'Remove from favourites' : 'Add to favourites',
			run: async () => {
				try {
					const d = await updateOutput(id, { favorite: !opts.favorite });
					patchItem(id, { favorite: d.favorite });
					ui.galleryVersion++;
				} catch (e) {
					notify(apiErrorMessage(e), 'error');
				}
			}
		},
		{
			label: 'Copy prompt',
			run: withDetail(id, (d) => copy(String(d.parameters.prompt ?? ''), 'Prompt'))
		},
		{
			label: 'Copy seed',
			run: withDetail(id, (d) => copy(String(d.seed ?? d.parameters.seed ?? ''), 'Seed'))
		},
		{ label: 'Download', run: () => download(downloadUrl(id)) },
		{
			label: 'Show its run in the gallery',
			run: withDetail(id, (d) => {
				leaveViewer();
				if (d.run_uuid) openGallery({ run: d.run_uuid });
			})
		},
		...(opts.more ?? []),
		{ separator: true },
		{
			label: 'Delete…',
			danger: true,
			keepOpen: true,
			run: () => {
				// second step: replace the entry with a confirmation
				if (!ui.menu) return;
				ui.menu.items = ui.menu.items.map((it) =>
					'label' in it && it.label === 'Delete…'
						? {
								label: 'Delete: click to confirm (moves to the trash folder)',
								danger: true,
								run: async () => {
									try {
										await deleteOutput(id);
										removeItems([id]);
										ui.galleryVersion++;
										if (ui.viewer?.ids.includes(id)) {
											const rest = ui.viewer.ids.filter((x) => x !== id);
											if (rest.length) openViewer(rest, Math.min(ui.viewer.index, rest.length - 1));
											else closeViewer();
										}
										notify('Image moved to the trash folder');
									} catch (e) {
										notify(apiErrorMessage(e), 'error');
									}
								}
							}
						: it
				);
			}
		}
	);
	return items;
}

/** Menu for a queue job (waiting, running or finished). */
export function jobMenu(job: QueueJob): MenuItem[] {
	const items: MenuItem[] = [];
	const ids = outputIds(job);
	const settings = job.payload ?? null;
	const isRender = job.kind === 'text_to_image' || job.kind === 'image_to_image';
	if (ids.length) items.push({ label: 'Open images', run: () => openViewer(ids, 0) });
	if (job.status === 'queued') {
		items.push(
			{ label: 'Run next', run: () => moveJob(job.id, 'top') },
			{ label: 'Move to the end', run: () => moveJob(job.id, 'bottom') }
		);
	}
	if (job.retryable) items.push({ label: 'Retry', run: () => retryJob(job.id) });
	if (job.run_uuid && job.status === 'completed')
		items.push({ label: 'Show in the gallery', run: () => openGallery({ run: job.run_uuid }) });
	if (settings && isRender) {
		items.push(
			{ separator: true },
			{ label: 'Use settings', run: () => useSettings(settings, true) },
			{ label: 'Send to Lab', run: () => sendToLab(settings) }
		);
	}
	if (settings?.prompt)
		items.push({ label: 'Copy prompt', run: () => copy(String(settings.prompt), 'Prompt') });
	if (job.status === 'queued' || job.status === 'running') {
		items.push(
			{ separator: true },
			{
				label: job.status === 'running' ? 'Stop' : 'Cancel',
				danger: true,
				run: () => cancelJob(job.id)
			}
		);
	}
	return items;
}
