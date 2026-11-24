<!-- frontend/src/lib/components/gallery/ImageViewer.svelte -->
<!-- Full-screen viewer for saved images. Keys: ←/→ browse, 1–5 rate (0 clears), F favourite,
     I details, Delete twice deletes, Esc closes. -->
<script lang="ts">
	import RatingStars from './RatingStars.svelte';
	import TagInput from './TagInput.svelte';
	import {
		bulkAction,
		deleteOutput,
		downloadUrl,
		getOutput,
		imageUrl,
		runTypeLabel,
		tagSourceLabel,
		updateOutput
	} from '#lib/gallery/api.js';
	import { gallery, patchItem, removeItems } from '#lib/gallery/state.svelte.js';
	import { imageMenu } from '#lib/gallery/menus.js';
	import {
		closeViewer,
		notify,
		openGallery,
		openMenu,
		sendToLab,
		ui,
		useAsInitImage,
		useSettings
	} from '#lib/stores/ui.svelte.js';
	import type { GalleryDetail } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let detail = $state<GalleryDetail | null>(null);
	let error = $state<string | null>(null);
	let showInfo = $state(true);
	let confirmDelete = $state(false);
	let notes = $state('');
	// plain cache of loaded details (not UI state, so not reactive)
	// eslint-disable-next-line svelte/prefer-svelte-reactivity
	const cache = new Map<number, GalleryDetail>();

	const id = $derived(ui.viewer ? ui.viewer.ids[ui.viewer.index] : null);
	const count = $derived(ui.viewer?.ids.length ?? 0);
	// generation parameters are free-form JSON
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const params = $derived((detail?.parameters ?? {}) as Record<string, any>);
	const manualTags = $derived(
		(detail?.tags ?? []).filter((t) => t.source === 'manual').map((t) => t.tag)
	);
	const otherTags = $derived((detail?.tags ?? []).filter((t) => t.source !== 'manual'));

	$effect(() => {
		const current = id;
		confirmDelete = false;
		error = null;
		if (current === null) {
			detail = null;
			return;
		}
		const cached = cache.get(current);
		if (cached) {
			detail = cached;
			notes = cached.notes ?? '';
		}
		getOutput(current)
			.then((d) => {
				cache.set(current, d);
				if (id === current) {
					detail = d;
					notes = d.notes ?? '';
				}
			})
			.catch((e) => {
				if (id === current) error = apiErrorMessage(e);
			});
		// preload the neighbours
		for (const n of [1, -1]) {
			const next = ui.viewer?.ids[(ui.viewer.index + n + count) % count];
			if (next !== undefined && next !== current) new Image().src = imageUrl(next);
		}
	});

	function step(delta: number) {
		if (!ui.viewer || count < 2) return;
		ui.viewer.index = (ui.viewer.index + delta + count) % count;
	}

	async function save(changes: Parameters<typeof updateOutput>[1]) {
		if (id === null) return;
		try {
			const updated = await updateOutput(id, changes);
			cache.set(id, updated);
			detail = updated;
			patchItem(id, {
				rating: updated.rating,
				favorite: updated.favorite,
				tags: updated.tags,
				notes: updated.notes
			});
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		}
	}

	async function removeTag(tag: string) {
		if (id === null) return;
		try {
			await bulkAction([id], 'remove_tags', [tag]);
			cache.delete(id);
			const fresh = await getOutput(id);
			cache.set(id, fresh);
			detail = fresh;
			patchItem(id, { tags: fresh.tags });
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		}
	}

	async function copyText(text: string, done: string) {
		try {
			await navigator.clipboard.writeText(text);
			notify(done, 'success', undefined, 1500);
		} catch {
			notify('The browser blocked clipboard access', 'error');
		}
	}

	async function remove() {
		if (id === null || !ui.viewer) return;
		if (!confirmDelete) {
			confirmDelete = true;
			return;
		}
		try {
			await deleteOutput(id);
			removeItems([id]);
			cache.delete(id);
			const ids = ui.viewer.ids.filter((x) => x !== id);
			if (!ids.length) closeViewer();
			else ui.viewer = { ids, index: Math.min(ui.viewer.index, ids.length - 1) };
			notify('Image moved to the trash folder', 'info');
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		}
	}

	function openInLab() {
		closeViewer();
		sendToLab(params);
	}

	async function copyPrompt() {
		try {
			await navigator.clipboard.writeText(String(params.prompt ?? ''));
			notify('Prompt copied', 'success', undefined, 1500);
		} catch {
			notify('The browser blocked clipboard access', 'error');
		}
	}

	function onKey(e: KeyboardEvent) {
		if (!ui.viewer) return;
		const target = e.target as HTMLElement | null;
		if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
		if (e.key === 'Escape') {
			if (confirmDelete) confirmDelete = false;
			else closeViewer();
		} else if (e.key === 'ArrowRight') step(1);
		else if (e.key === 'ArrowLeft') step(-1);
		else if (/^[0-5]$/.test(e.key)) save({ rating: Number(e.key) });
		else if (e.key.toLowerCase() === 'f') save({ favorite: !detail?.favorite });
		else if (e.key.toLowerCase() === 'i') showInfo = !showInfo;
		else if (e.key === 'Delete') remove();
		else return;
		e.preventDefault();
	}

	const row = (label: string, value: unknown) =>
		value === null || value === undefined || value === '' ? null : { label, value: String(value) };
	const facts = $derived(
		[
			row('Model', params.model_id),
			row('Seed', params.seed),
			row('Size', params.width && params.height ? `${params.width}×${params.height}` : null),
			row('Steps', params.num_inference_steps),
			row('Guidance', params.guidance_scale),
			row('Sampler', params.scheduler_id),
			row(
				'LoRAs',
				(params.loras ?? [])
					.map((l: { name: string; weight: number }) => `${l.name} (${l.weight})`)
					.join(', ')
			),
			row('Strength', params.strength),
			row('Family', params.family),
			row('Lab tool', params.lab_tool),
			row('Created', detail?.created ? new Date(detail.created).toLocaleString() : null)
		].filter((r) => r !== null)
	);
</script>

<svelte:window onkeydown={onKey} />

{#if ui.viewer && id !== null}
	<div
		class="fixed inset-0 z-50 flex bg-neutral-950 text-gray-100"
		role="dialog"
		aria-modal="true"
		aria-label="Image viewer"
	>
		<div class="relative flex min-w-0 flex-1 flex-col">
			<div class="flex items-center justify-between gap-2 px-4 py-2 text-sm">
				<span class="tabular-nums opacity-80">{ui.viewer.index + 1} / {count}</span>
				<div class="flex items-center gap-3">
					<button type="button" class="hover:underline" onclick={() => (showInfo = !showInfo)}
						>{showInfo ? 'Hide' : 'Show'} details (I)</button
					>
					<!-- a file download from the API, not an app route -->
					<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -->
					<a class="hover:underline" href={downloadUrl(id)} download>Download</a>
					<button
						type="button"
						class="text-2xl leading-none hover:text-white"
						aria-label="Close"
						onclick={closeViewer}>×</button
					>
				</div>
			</div>
			<div class="relative flex min-h-0 flex-1 items-center justify-center px-12 pb-4">
				<img
					src={imageUrl(id)}
					alt={String(params.prompt ?? 'Image')}
					class="h-full w-full object-contain"
					style:image-rendering={Number(params.width) && Number(params.width) < 384
						? 'pixelated'
						: 'auto'}
					oncontextmenu={(e) =>
						openMenu(
							e,
							imageMenu(id, {
								inViewer: true,
								rating: detail?.rating ?? null,
								favorite: detail?.favorite ?? false
							}),
							String(params.prompt ?? '')
						)}
				/>
				{#if count > 1}
					<button
						type="button"
						class="absolute top-1/2 left-2 -translate-y-1/2 rounded-full bg-white/10 px-3 py-2 text-2xl hover:bg-white/20"
						aria-label="Previous"
						onclick={() => step(-1)}>‹</button
					>
					<button
						type="button"
						class="absolute top-1/2 right-2 -translate-y-1/2 rounded-full bg-white/10 px-3 py-2 text-2xl hover:bg-white/20"
						aria-label="Next"
						onclick={() => step(1)}>›</button
					>
				{/if}
			</div>
		</div>

		{#if showInfo}
			<aside
				class="w-96 flex-none overflow-y-auto border-l border-white/10 bg-gray-900 p-4 text-sm"
			>
				{#if error}
					<p class="text-red-400">{error}</p>
				{:else if detail}
					<div class="flex items-center justify-between">
						<RatingStars value={detail.rating} onchange={(n) => save({ rating: n })} />
						<button
							type="button"
							class="text-2xl leading-none"
							class:text-rose-400={detail.favorite}
							class:text-gray-500={!detail.favorite}
							title="Favourite (F)"
							aria-pressed={detail.favorite}
							onclick={() => save({ favorite: !detail?.favorite })}>♥</button
						>
					</div>

					<div class="mt-4">
						<div class="flex items-center justify-between">
							<h3 class="text-xs font-semibold tracking-wide text-gray-400 uppercase">Prompt</h3>
							<button
								type="button"
								class="text-xs text-indigo-300 hover:underline"
								onclick={copyPrompt}>Copy</button
							>
						</div>
						<p class="mt-1 whitespace-pre-wrap">{params.prompt ?? '—'}</p>
						{#if detail.negative_prompt}
							<h3 class="mt-3 text-xs font-semibold tracking-wide text-gray-400 uppercase">
								Negative
							</h3>
							<p class="mt-1 whitespace-pre-wrap text-gray-300">{detail.negative_prompt}</p>
						{/if}
					</div>

					<div class="mt-4 grid grid-cols-2 gap-2">
						<button
							type="button"
							class="rounded bg-indigo-600 px-2 py-1.5 font-medium hover:bg-indigo-500"
							onclick={() => {
								useSettings(params, true);
								closeViewer();
							}}>Use settings</button
						>
						<button
							type="button"
							class="rounded bg-white/10 px-2 py-1.5 hover:bg-white/20"
							onclick={() => {
								useSettings(params, false);
								closeViewer();
							}}>…with new seed</button
						>
						<button
							type="button"
							class="rounded bg-white/10 px-2 py-1.5 hover:bg-white/20"
							onclick={async () => {
								await useAsInitImage(imageUrl(id), params);
								closeViewer();
							}}>Use as input</button
						>
						<button
							type="button"
							class="rounded bg-white/10 px-2 py-1.5 hover:bg-white/20"
							onclick={openInLab}>Open in Lab</button
						>
					</div>

					<dl class="mt-4 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
						{#each facts as f (f.label)}
							<dt class="text-gray-400">{f.label}</dt>
							<dd class="break-words">{f.value}</dd>
						{/each}
						{#if detail.run_type}
							<dt class="text-gray-400">Run</dt>
							<dd>
								{runTypeLabel(detail.run_type)}
								<button
									type="button"
									class="ml-1 text-indigo-300 hover:underline"
									onclick={() => {
										openGallery({ run: detail?.run_uuid });
										closeViewer();
									}}>show all</button
								>
							</dd>
						{/if}
					</dl>

					<div class="mt-4">
						<h3 class="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">Tags</h3>
						<div class="text-gray-900 dark:text-gray-100">
							<TagInput
								id="viewer-tags"
								tags={manualTags}
								suggestions={(gallery.facets?.tags ?? []).map((t) => t.tag)}
								onchange={(tags) => save({ tags })}
							/>
						</div>
						{#if otherTags.length}
							<div class="mt-2 flex flex-wrap gap-1">
								{#each otherTags as t (t.tag + t.source)}
									<span
										class="inline-flex items-center gap-1 rounded bg-teal-900/60 px-1.5 py-0.5 text-xs"
										title="{tagSourceLabel(t.source)}{t.confidence !== null
											? ` (${(t.confidence * 100).toFixed(0)}%)`
											: ''}"
										>{t.tag}
										<span class="text-teal-300/70">{t.source}</span>
										<button
											type="button"
											class="text-gray-400 hover:text-white"
											aria-label="Remove tag {t.tag}"
											onclick={() => removeTag(t.tag)}>×</button
										></span
									>
								{/each}
							</div>
						{/if}
					</div>

					{#if detail.captions?.length}
						<div class="mt-4">
							<h3 class="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">
								Captions
							</h3>
							<div class="space-y-2">
								{#each detail.captions as c (c.source)}
									<div class="rounded bg-white/5 p-2 text-sm">
										<div class="mb-1 flex items-center justify-between text-xs text-gray-400">
											<span>{c.source}</span>
											<button
												type="button"
												class="hover:text-white"
												onclick={() => copyText(c.text, 'Caption copied')}>Copy</button
											>
										</div>
										<p class="whitespace-pre-wrap">{c.text}</p>
									</div>
								{/each}
							</div>
						</div>
					{/if}

					{#if detail.projects?.length}
						<p class="mt-3 text-xs text-gray-400">
							In project{detail.projects.length > 1 ? 's' : ''}:
							{#each detail.projects as pr, i (pr.uuid)}{i ? ', ' : ''}<button
									type="button"
									class="text-indigo-300 hover:underline"
									title={pr.file_name}
									onclick={() => {
										closeViewer();
										openGallery({ project: pr.uuid });
									}}>{pr.name}</button
								>{/each}
						</p>
					{/if}

					<div class="mt-4">
						<label
							for="viewer-notes"
							class="mb-1 block text-xs font-semibold tracking-wide text-gray-400 uppercase"
							>Notes</label
						>
						<textarea
							id="viewer-notes"
							rows="3"
							class="w-full rounded border border-white/20 bg-gray-800 p-2 text-sm"
							bind:value={notes}
							onblur={() => notes !== (detail?.notes ?? '') && save({ notes })}></textarea>
					</div>

					<div class="mt-4 flex items-center justify-between border-t border-white/10 pt-3">
						<span class="truncate text-xs text-gray-500" title={detail.file_path}
							>{detail.file_name}</span
						>
						<button
							type="button"
							class="rounded px-2 py-1 text-red-300 hover:bg-red-900/50"
							class:bg-red-800={confirmDelete}
							onclick={remove}
						>
							{confirmDelete ? 'Click again to delete' : 'Delete'}
						</button>
					</div>
					{#if !detail.exists}<p class="mt-2 text-xs text-amber-300">
							The file is missing on disk.
						</p>{/if}
				{:else}
					<p class="text-gray-400">Loading…</p>
				{/if}
			</aside>
		{/if}
	</div>
{/if}
