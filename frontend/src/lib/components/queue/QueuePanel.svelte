<!-- frontend/src/lib/components/queue/QueuePanel.svelte -->
<!-- Header button + side drawer for the render queue. -->
<script lang="ts">
	import { thumbUrl } from '#lib/gallery/api.js';
	import {
		cancelJob,
		clearFinished,
		finishedJobs,
		jobFraction,
		moveJob,
		outputIds,
		queue,
		retryJob,
		runningJob,
		waitingJobs
	} from '#lib/stores/queue.svelte.js';
	import { jobMenu } from '#lib/gallery/menus.js';
	import { notify, openGallery, openMenu, openViewer, ui } from '#lib/stores/ui.svelte.js';
	import { listSweeps, sweepProgress, writeSweepResults } from '#lib/sweeps.js';
	import type { QueueJob, SweepStatus } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';
	import SweepImportDialog from './SweepImportDialog.svelte';

	const running = $derived(runningJob());
	const waiting = $derived(waitingJobs());
	const finished = $derived(finishedJobs(40));
	const fraction = $derived(running ? jobFraction(running) : null);

	function kindLabel(job: QueueJob): string {
		if (job.kind === 'xyz_grid') return 'Grid';
		if (job.kind.startsWith('lab_')) return 'Lab';
		if (job.kind === 'image_to_image') return 'img2img';
		return 'Render';
	}

	function progressText(job: QueueJob): string {
		const p = job.progress ?? {};
		if (p.phase === 'cancelling') return 'stopping…';
		const parts = [];
		if (p.items && p.items > 1) parts.push(`image ${p.item ?? 1}/${p.items}`);
		if (p.total) parts.push(`step ${p.step ?? 0}/${p.total}`);
		return parts.join(' · ') || 'loading model…';
	}

	function ago(seconds: number | null): string {
		if (!seconds) return '';
		const s = Math.max(0, Date.now() / 1000 - seconds);
		if (s < 60) return 'just now';
		if (s < 3600) return `${Math.floor(s / 60)} min ago`;
		if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
		return new Date(seconds * 1000).toLocaleDateString();
	}

	async function act(fn: () => Promise<unknown>) {
		try {
			await fn();
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		}
	}

	// test sweeps imported from ArtAgents (files): progress and their results folders
	let importing = $state(false);
	let sweeps = $state<SweepStatus[]>([]);

	async function loadSweeps() {
		try {
			sweeps = (await listSweeps()).items.slice(0, 5);
		} catch {
			sweeps = [];
		}
	}

	$effect(() => {
		if (ui.queueOpen) {
			void finished.length; // refresh as jobs finish
			loadSweeps();
		}
	});

	async function writeNow(id: string) {
		await act(async () => {
			const r = await writeSweepResults(id);
			notify(`Results written: ${r.images} images, ${r.errors} missing or failed`, 'success');
			await loadSweeps();
		});
	}

	async function copyPath(path: string) {
		try {
			await navigator.clipboard.writeText(path);
			notify('Folder path copied', 'success', undefined, 1500);
		} catch {
			notify('The browser blocked clipboard access', 'error');
		}
	}
</script>

<svelte:window
	onkeydown={(e) => {
		if (e.key === 'Escape' && ui.queueOpen && !ui.viewer) ui.queueOpen = false;
	}}
/>

<button
	type="button"
	class="relative inline-flex items-center gap-2 rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium shadow-sm hover:bg-gray-50 dark:border-gray-600 dark:bg-gray-800 dark:hover:bg-gray-700"
	aria-expanded={ui.queueOpen}
	onclick={() => (ui.queueOpen = !ui.queueOpen)}
>
	{#if running}
		<span
			class="inline-block h-3 w-3 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent"
		></span>
	{:else}
		<span
			class="inline-block h-2 w-2 rounded-full"
			class:bg-green-500={queue.connected}
			class:bg-gray-400={!queue.connected}
		></span>
	{/if}
	Queue
	{#if waiting.length}
		<span class="rounded-full bg-indigo-600 px-1.5 text-xs text-white tabular-nums"
			>{waiting.length}</span
		>
	{/if}
	{#if running && fraction !== null}
		<span
			class="absolute right-1 bottom-0.5 left-1 h-0.5 overflow-hidden rounded bg-gray-200 dark:bg-gray-700"
		>
			<span class="block h-full bg-indigo-500 transition-all" style:width="{fraction * 100}%"
			></span>
		</span>
	{/if}
</button>

{#if ui.queueOpen}
	<button
		type="button"
		class="fixed inset-0 z-30 cursor-default bg-black/20"
		aria-label="Close the queue"
		onclick={() => (ui.queueOpen = false)}
	></button>
	<aside
		class="fixed top-0 right-0 z-40 flex h-full w-full max-w-md flex-col border-l border-gray-200 bg-white shadow-xl dark:border-gray-700 dark:bg-gray-900"
		aria-label="Render queue"
	>
		<header
			class="flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-700"
		>
			<div>
				<h2 class="text-lg font-semibold">Queue</h2>
				<p class="text-xs text-gray-500">
					{queue.connected ? 'Live' : 'Reconnecting…'} · {waiting.length} waiting
				</p>
			</div>
			<span class="flex items-center gap-3">
				<button
					type="button"
					class="text-xs text-indigo-600 hover:underline dark:text-indigo-400"
					title="Queue a test sweep file from ArtAgents' LoRA Lab"
					onclick={() => (importing = true)}>Import test sweep…</button
				>
				<button
					type="button"
					class="text-2xl leading-none text-gray-500 hover:text-gray-800"
					aria-label="Close"
					onclick={() => (ui.queueOpen = false)}>×</button
				>
			</span>
		</header>

		<div class="flex-1 space-y-5 overflow-y-auto p-4">
			<section>
				<h3 class="mb-2 text-xs font-semibold tracking-wide text-gray-500 uppercase">Running</h3>
				{#if running}
					<div
						class="rounded border border-indigo-200 bg-indigo-50 p-3 dark:border-indigo-800 dark:bg-indigo-950"
						role="group"
						oncontextmenu={(e) => openMenu(e, jobMenu(running), running.label)}
					>
						<div class="flex items-start justify-between gap-2">
							<div class="min-w-0">
								<p class="truncate text-sm font-medium" title={running.label}>{running.label}</p>
								<p class="text-xs text-gray-600 dark:text-gray-400">
									{kindLabel(running)} · {running.model_id ?? ''}
								</p>
							</div>
							<button
								type="button"
								class="text-sm text-red-600 hover:underline"
								onclick={() => act(() => cancelJob(running.id))}>Stop</button
							>
						</div>
						<div class="mt-2 h-1.5 overflow-hidden rounded bg-indigo-100 dark:bg-indigo-900">
							{#if fraction !== null}
								<div
									class="h-full bg-indigo-500 transition-all"
									style:width="{fraction * 100}%"
								></div>
							{:else}
								<div class="h-full w-1/3 animate-pulse bg-indigo-400"></div>
							{/if}
						</div>
						<p class="mt-1 text-xs text-gray-600 tabular-nums dark:text-gray-400">
							{progressText(running)}
						</p>
					</div>
				{:else}
					<p class="text-sm text-gray-500">Nothing running.</p>
				{/if}
			</section>

			{#if waiting.length}
				<section>
					<h3 class="mb-2 text-xs font-semibold tracking-wide text-gray-500 uppercase">
						Waiting ({waiting.length})
					</h3>
					<ol class="space-y-1">
						{#each waiting as job, i (job.id)}
							<li
								class="flex items-center gap-2 rounded border border-gray-200 px-2 py-1.5 text-sm dark:border-gray-700"
								oncontextmenu={(e) => openMenu(e, jobMenu(job), job.label)}
							>
								<span class="w-5 text-right text-xs text-gray-400 tabular-nums">{i + 1}</span>
								<span class="min-w-0 flex-1 truncate" title={job.label}>{job.label}</span>
								<span class="flex flex-none gap-0.5 text-gray-500">
									<button
										type="button"
										class="px-1 hover:text-gray-900 disabled:opacity-30"
										title="Run next"
										disabled={i === 0}
										onclick={() => act(() => moveJob(job.id, 'top'))}>⤒</button
									>
									<button
										type="button"
										class="px-1 hover:text-gray-900 disabled:opacity-30"
										title="Up"
										disabled={i === 0}
										onclick={() => act(() => moveJob(job.id, 'up'))}>▲</button
									>
									<button
										type="button"
										class="px-1 hover:text-gray-900 disabled:opacity-30"
										title="Down"
										disabled={i === waiting.length - 1}
										onclick={() => act(() => moveJob(job.id, 'down'))}>▼</button
									>
									<button
										type="button"
										class="px-1 text-red-600 hover:text-red-800"
										title="Cancel"
										onclick={() => act(() => cancelJob(job.id))}>✕</button
									>
								</span>
							</li>
						{/each}
					</ol>
				</section>
			{/if}

			{#if sweeps.length}
				<section>
					<h3 class="mb-2 text-xs font-semibold tracking-wide text-gray-500 uppercase">
						Test sweeps
					</h3>
					<ul class="space-y-2">
						{#each sweeps as sweep (sweep.id)}
							<li class="rounded border border-gray-200 p-2 text-sm dark:border-gray-700">
								<p class="truncate" title={sweep.name}>{sweep.name}</p>
								<p class="text-xs text-gray-500">
									{sweepProgress(sweep)}
									{#if sweep.written}· results written{:else if sweep.done}· writing results…{/if}
								</p>
								<span class="mt-1 flex gap-3 text-xs">
									<button
										type="button"
										class="text-indigo-600 hover:underline dark:text-indigo-400"
										onclick={() => {
											ui.queueOpen = false;
											openGallery({ run: sweep.run_uuid });
										}}>Gallery</button
									>
									<button
										type="button"
										class="text-indigo-600 hover:underline dark:text-indigo-400"
										title={sweep.results_dir}
										onclick={() => copyPath(sweep.results_dir)}>Copy results path</button
									>
									<button
										type="button"
										class="text-indigo-600 hover:underline dark:text-indigo-400"
										title="Write the results folder with what is finished so far"
										onclick={() => writeNow(sweep.id)}
										>{sweep.written ? 'Write again' : 'Write results now'}</button
									>
								</span>
							</li>
						{/each}
					</ul>
				</section>
			{/if}

			<section>
				<div class="mb-2 flex items-center justify-between">
					<h3 class="text-xs font-semibold tracking-wide text-gray-500 uppercase">Finished</h3>
					{#if finished.length}
						<button
							type="button"
							class="text-xs text-gray-500 hover:underline"
							onclick={() => act(clearFinished)}>Clear</button
						>
					{/if}
				</div>
				{#if !finished.length}
					<p class="text-sm text-gray-500">Finished jobs appear here.</p>
				{/if}
				<ul class="space-y-2">
					{#each finished as job (job.id)}
						{@const ids = outputIds(job)}
						<li
							class="rounded border border-gray-200 p-2 text-sm dark:border-gray-700"
							oncontextmenu={(e) => openMenu(e, jobMenu(job), job.label)}
						>
							<div class="flex items-start justify-between gap-2">
								<div class="min-w-0">
									<p class="truncate" title={job.label}>
										{#if job.status === 'completed'}<span class="text-green-600">✓</span>
										{:else if job.status === 'failed'}<span class="text-red-600">!</span>
										{:else}<span class="text-gray-400">–</span>{/if}
										{job.label}
									</p>
									<p class="text-xs text-gray-500">
										{kindLabel(job)} · {job.status} · {ago(job.finished)}
										{#if job.result?.performance_ms}· {(job.result.performance_ms / 1000).toFixed(
												1
											)} s{/if}
									</p>
								</div>
								<span class="flex flex-none gap-2 text-xs">
									{#if job.run_uuid && job.status === 'completed'}
										<button
											type="button"
											class="text-indigo-600 hover:underline dark:text-indigo-400"
											onclick={() => {
												ui.queueOpen = false;
												openGallery({ run: job.run_uuid });
											}}>Gallery</button
										>
									{/if}
									{#if job.retryable}
										<button
											type="button"
											class="text-indigo-600 hover:underline dark:text-indigo-400"
											onclick={() => act(() => retryJob(job.id))}>Retry</button
										>
									{/if}
								</span>
							</div>
							{#if job.error}<p class="mt-1 text-xs break-words text-red-600">{job.error}</p>{/if}
							{#if job.result?.loras_skipped?.length || job.result?.embeddings_skipped?.length}
								<p class="mt-1 text-xs break-words text-amber-700 dark:text-amber-300">
									Not applied (other model family or missing): {[
										...(job.result.loras_skipped ?? []),
										...(job.result.embeddings_skipped ?? [])
									].join(', ')}
								</p>
							{/if}
							{#if ids.length}
								<div class="mt-2 flex gap-1">
									{#each ids.slice(0, 6) as id, k (id)}
										<button
											type="button"
											class="overflow-hidden rounded hover:ring-2 hover:ring-indigo-500"
											onclick={() => openViewer(ids, k)}
										>
											<img
												src={thumbUrl(id, 128)}
												alt="Output {id}"
												class="h-12 w-12 object-cover"
												loading="lazy"
											/>
										</button>
									{/each}
									{#if ids.length > 6}<span class="self-center text-xs text-gray-500"
											>+{ids.length - 6}</span
										>{/if}
								</div>
							{/if}
						</li>
					{/each}
				</ul>
			</section>
		</div>
	</aside>
{/if}

{#if importing}
	<SweepImportDialog onclose={() => (importing = false)} onqueued={() => loadSweeps()} />
{/if}
