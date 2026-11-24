<!-- frontend/src/lib/components/render/PreviewPanel.svelte -->
<!-- The big view beside a render form: the running job live (step by step), then its result.
     Modes: Live (follows the newest job), Compare (wipe between two results), Batch (all
     images of the newest finished job). The strip below lists this form's jobs. -->
<script lang="ts">
	import ExplorePane from './ExplorePane.svelte';
	import RecentRenders from './RecentRenders.svelte';
	import { imageUrl, previewUrl, thumbUrl } from '#lib/gallery/api.js';
	import { imageMenu, jobMenu } from '#lib/gallery/menus.js';
	import * as ui from '#lib/lab/ui.js';
	import { cancelJob, formJobs, jobFraction, outputIds, queue } from '#lib/stores/queue.svelte.js';
	import { openMenu, openViewer, sendToLab, useSettings } from '#lib/stores/ui.svelte.js';
	import type { QueueJob } from '#lib/types.js';

	let { form }: { form: 'text' | 'image' } = $props();

	type Mode = 'live' | 'compare' | 'batch' | 'explore';
	let mode = $state<Mode>('live');
	/** picked image (null: follow the newest) */
	let picked = $state<number | null>(null);
	/** the other side of Compare (null: the image before the picked one) */
	let compareWith = $state<number | null>(null);
	let wipe = $state(50);
	let fit = $state(true);

	const jobs = $derived(formJobs[form].map((id) => queue.jobs[id]).filter(Boolean) as QueueJob[]);
	const running = $derived(jobs.find((j) => j.status === 'running') ?? null);
	const waiting = $derived(jobs.filter((j) => j.status === 'queued'));
	const allIds = $derived(jobs.flatMap((j) => outputIds(j)));
	const newestDone = $derived(jobs.find((j) => outputIds(j).length > 0) ?? null);
	const shown = $derived(picked ?? allIds[0] ?? null);
	const other = $derived.by(() => {
		if (compareWith !== null && compareWith !== shown) return compareWith;
		const i = shown === null ? -1 : allIds.indexOf(shown);
		return allIds[i + 1] ?? null;
	});
	/** show the live preview: following the newest, and something is rendering */
	const live = $derived(mode === 'live' && picked === null && running !== null);
	const previewSrc = $derived(
		running && running.progress.preview
			? previewUrl(running.id, Number(running.progress.preview))
			: null
	);
	const shownJob = $derived(jobs.find((j) => shown !== null && outputIds(j).includes(shown)));

	// a newly finished render takes over the view when following the newest
	let lastNewest = -1;
	$effect(() => {
		const n = allIds[0] ?? -1;
		if (n !== lastNewest) {
			lastNewest = n;
			if (picked !== null && !allIds.includes(picked)) picked = null;
		}
	});

	function pick(id: number, e?: MouseEvent) {
		if (e?.shiftKey) {
			compareWith = id;
			mode = 'compare';
			return;
		}
		picked = id === allIds[0] ? null : id;
		if (mode === 'batch') mode = 'live';
	}

	function step(delta: number) {
		if (!allIds.length || shown === null) return;
		const i = Math.max(0, Math.min(allIds.length - 1, allIds.indexOf(shown) + delta));
		picked = i === 0 ? null : allIds[i];
	}

	function onKey(e: KeyboardEvent) {
		const t = e.target as HTMLElement;
		if (t.closest('input, textarea, select, [contenteditable]')) return;
		if (e.key === '[') step(1);
		if (e.key === ']') step(-1);
	}

	const caption = (j: QueueJob | undefined) =>
		j
			? `${j.label}${j.result?.performance_ms ? ` · ${(j.result.performance_ms / 1000).toFixed(1)} s` : ''}`
			: '';
</script>

<svelte:window onkeydown={onKey} />

<div class="space-y-3 xl:sticky xl:top-4">
	<div class="{ui.card} !p-3">
		<div class="mb-2 flex flex-wrap items-center justify-between gap-2">
			<div class="inline-flex rounded-md shadow-sm" role="group" aria-label="Preview mode">
				{#each [['live', running ? '● Live' : 'Result'], ['compare', 'Compare'], ['batch', 'Batch'], ['explore', 'Explore']] as [id, label], i (id)}
					<button
						type="button"
						class="border border-gray-300 px-3 py-1 text-sm dark:border-gray-600"
						class:rounded-l-md={i === 0}
						class:rounded-r-md={i === 3}
						class:bg-indigo-600={mode === id}
						class:text-white={mode === id}
						aria-pressed={mode === id}
						onclick={() => (mode = id as Mode)}>{label}</button
					>
				{/each}
			</div>
			<div class="flex items-center gap-3 text-xs text-gray-500">
				{#if picked !== null}
					<button
						type="button"
						class="text-indigo-600 hover:underline"
						onclick={() => (picked = null)}>Follow newest</button
					>
				{/if}
				<label class="flex items-center gap-1"
					><input type="checkbox" bind:checked={fit} /> Fit</label
				>
				{#if waiting.length}<span>{waiting.length} waiting</span>{/if}
			</div>
		</div>

		{#if mode === 'explore'}
			<ExplorePane {form} id={shown} />
		{:else}
			<div
				class="relative flex min-h-80 items-center justify-center overflow-auto rounded bg-gray-100 dark:bg-gray-900"
				class:h-[min(72vh,56rem)]={fit}
			>
				{#if mode === 'batch'}
					{#if newestDone}
						{@const ids = outputIds(newestDone)}
						<div
							class="grid h-full w-full gap-1 p-1"
							style:grid-template-columns="repeat({Math.ceil(Math.sqrt(ids.length))}, minmax(0,
							1fr))"
						>
							{#each ids as id (id)}
								<button
									type="button"
									class="flex min-h-0 items-center justify-center"
									onclick={() => pick(id)}
									ondblclick={() => openViewer(ids, ids.indexOf(id))}
									oncontextmenu={(e) => openMenu(e, imageMenu(id, { ids }), newestDone.label)}
								>
									<img src={imageUrl(id)} alt="" class="max-h-full max-w-full object-contain" />
								</button>
							{/each}
						</div>
					{:else}
						<p class={ui.muted}>The images of the newest finished job show here.</p>
					{/if}
				{:else if mode === 'compare'}
					{#if shown !== null && other !== null}
						<div class="relative h-full w-full select-none">
							<img
								src={imageUrl(other)}
								alt="B"
								class="absolute inset-0 h-full w-full object-contain"
							/>
							<img
								src={imageUrl(shown)}
								alt="A"
								class="absolute inset-0 h-full w-full object-contain"
								style:clip-path="inset(0 {100 - wipe}% 0 0)"
							/>
							<div
								class="pointer-events-none absolute inset-y-0 w-0.5 bg-white/80 shadow"
								style:left="{wipe}%"
							></div>
							<span class="absolute top-2 left-2 rounded bg-black/60 px-1.5 text-xs text-white"
								>A</span
							>
							<span class="absolute top-2 right-2 rounded bg-black/60 px-1.5 text-xs text-white"
								>B</span
							>
							<input
								type="range"
								min="0"
								max="100"
								bind:value={wipe}
								aria-label="Wipe between A and B"
								class="absolute inset-x-6 bottom-3 accent-white"
							/>
						</div>
					{:else}
						<p class="{ui.muted} p-6 text-center">
							Render two images, then compare them here. Click a thumbnail to change A, Shift+click
							to change B.
						</p>
					{/if}
				{:else if live}
					{#if previewSrc}
						<img
							src={previewSrc}
							alt="Live preview"
							class="h-full w-full object-contain"
							style:image-rendering="auto"
							oncontextmenu={(e) => running && openMenu(e, jobMenu(running), running.label)}
						/>
					{:else}
						<div class="text-center text-sm text-gray-500">
							<div
								class="mx-auto mb-3 h-10 w-10 animate-spin rounded-full border-b-2 border-indigo-500"
							></div>
							{running?.progress.total ? 'Starting…' : 'Loading the model…'}
						</div>
					{/if}
					<div class="absolute inset-x-0 bottom-0 bg-black/55 px-3 py-1.5 text-xs text-white">
						<div class="flex items-center justify-between gap-2">
							<span class="truncate">{running?.label}</span>
							<span class="flex-none tabular-nums"
								>{running?.progress.total
									? `step ${running.progress.step ?? 0}/${running.progress.total}`
									: ''}
								<button
									type="button"
									class="ml-2 underline"
									onclick={() => running && cancelJob(running.id)}>Stop</button
								></span
							>
						</div>
						<div class="mt-1 h-1 overflow-hidden rounded bg-white/20">
							<div
								class="h-full bg-indigo-400 transition-all"
								style:width="{(jobFraction(running!) ?? 0) * 100}%"
							></div>
						</div>
					</div>
				{:else if shown !== null}
					<button
						type="button"
						class="flex h-full w-full items-center justify-center"
						title="Double-click for the viewer; right-click for more"
						ondblclick={() => openViewer(allIds, allIds.indexOf(shown))}
						oncontextmenu={(e) => openMenu(e, imageMenu(shown, { ids: allIds }), shownJob?.label)}
					>
						<img
							src={imageUrl(shown)}
							alt={shownJob?.label ?? ''}
							class={fit ? 'max-h-full max-w-full object-contain' : ''}
						/>
					</button>
				{:else if waiting.length}
					<p class={ui.muted}>Waiting in the queue…</p>
				{:else}
					<p class="{ui.muted} p-6 text-center">
						Queue a render: it shows here step by step, then the result.
					</p>
				{/if}
			</div>

			{#if mode === 'live' && !live && shown !== null}
				<div class="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs">
					<span class="min-w-0 truncate text-gray-500">{caption(shownJob)}</span>
					<div class="flex flex-none gap-3">
						<button
							type="button"
							class="text-indigo-600 hover:underline dark:text-indigo-400"
							onclick={() => shownJob?.payload && useSettings(shownJob.payload, true)}
							disabled={!shownJob?.payload}>Use settings</button
						>
						<button
							type="button"
							class="text-indigo-600 hover:underline dark:text-indigo-400"
							onclick={() => shownJob?.payload && sendToLab(shownJob.payload)}
							disabled={!shownJob?.payload}>Send to Lab</button
						>
						<button
							type="button"
							class="text-indigo-600 hover:underline dark:text-indigo-400"
							onclick={() => (mode = 'explore')}>Explore</button
						>
						<button
							type="button"
							class="text-indigo-600 hover:underline dark:text-indigo-400"
							onclick={() => openViewer(allIds, allIds.indexOf(shown))}>Viewer</button
						>
					</div>
				</div>
			{/if}
			{#if mode === 'compare' && shown !== null && other !== null}
				<div class="mt-2 grid grid-cols-2 gap-2 text-xs text-gray-500">
					<div class="flex items-center gap-2">
						<img src={thumbUrl(shown, 128)} alt="" class="h-8 w-8 rounded object-cover" /> A: click a
						thumbnail
					</div>
					<div class="flex items-center justify-end gap-2">
						B: Shift+click a thumbnail <img
							src={thumbUrl(other, 128)}
							alt=""
							class="h-8 w-8 rounded object-cover"
						/>
					</div>
				</div>
			{/if}
		{/if}
	</div>

	<RecentRenders {form} selected={shown} {other} onpick={pick} />
</div>
