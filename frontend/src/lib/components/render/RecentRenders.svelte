<!-- frontend/src/lib/components/render/RecentRenders.svelte -->
<!-- Jobs queued from one render form: waiting, running (with progress) and finished images. -->
<script lang="ts">
	import { thumbUrl } from '#lib/gallery/api.js';
	import * as ui from '#lib/lab/ui.js';
	import {
		cancelJob,
		formJobs,
		jobFraction,
		outputIds,
		queue,
		retryJob
	} from '#lib/stores/queue.svelte.js';
	import { openGallery, openViewer } from '#lib/stores/ui.svelte.js';

	let { form }: { form: 'text' | 'image' } = $props();

	const jobs = $derived(formJobs[form].map((id) => queue.jobs[id]).filter(Boolean));
	const allIds = $derived(jobs.flatMap((j) => outputIds(j)));
	// the latest error, spelled out under the tiles
	const failed = $derived(jobs.find((j) => j.status === 'failed' && j.error));
</script>

{#if jobs.length}
	<div class={ui.card}>
		<div class="mb-3 flex items-center justify-between">
			<h3 class="font-medium">Recent renders</h3>
			<div class="flex gap-2">
				<button
					type="button"
					class="text-sm text-indigo-600 hover:underline dark:text-indigo-400"
					onclick={() => openGallery({})}>Open gallery</button
				>
				<button
					type="button"
					class="text-sm text-gray-500 hover:underline"
					onclick={() =>
						(formJobs[form] = formJobs[form].filter(
							(id) => !['completed', 'failed', 'cancelled'].includes(queue.jobs[id]?.status)
						))}>Clear finished</button
				>
			</div>
		</div>
		<!-- one tile per image; jobs without images yet show their status in a tile -->
		<div class="grid gap-2" style:grid-template-columns="repeat(auto-fill, minmax(112px, 1fr))">
			{#each jobs as job (job.id)}
				{#if outputIds(job).length}
					{#each outputIds(job) as id (id)}
						<button
							type="button"
							class="group relative overflow-hidden rounded bg-gray-100 hover:ring-2 hover:ring-indigo-500 dark:bg-gray-700"
							title={job.label}
							onclick={() => openViewer(allIds, allIds.indexOf(id))}
						>
							<img
								src={thumbUrl(id, 256)}
								alt={job.label}
								class="aspect-square w-full object-cover"
								loading="lazy"
							/>
							<span
								class="absolute inset-x-0 bottom-0 truncate bg-black/50 px-1 py-0.5 text-left text-[10px] text-white opacity-0 transition group-hover:opacity-100"
								>{job.label}</span
							>
						</button>
					{/each}
				{:else}
					<div
						class="flex aspect-square flex-col justify-between rounded border border-dashed border-gray-300 p-2 text-xs dark:border-gray-600"
						title={job.label}
					>
						<span class="line-clamp-3 text-gray-600 dark:text-gray-300">{job.label}</span>
						<div class="space-y-1">
							{#if job.status === 'running'}
								<div class="h-1.5 overflow-hidden rounded bg-gray-200 dark:bg-gray-700">
									{#if jobFraction(job) !== null}
										<div
											class="h-full bg-indigo-500 transition-all"
											style:width="{(jobFraction(job) ?? 0) * 100}%"
										></div>
									{:else}
										<div class="h-full w-1/3 animate-pulse bg-indigo-400"></div>
									{/if}
								</div>
							{/if}
							<div class="flex items-center justify-between gap-1">
								{#if job.status === 'queued'}
									<span class="text-gray-500">#{job.position ?? '–'} waiting</span>
									<button
										type="button"
										class="text-red-600 hover:underline"
										onclick={() => cancelJob(job.id)}>Cancel</button
									>
								{:else if job.status === 'running'}
									<span class="text-indigo-600 tabular-nums"
										>{job.progress.phase === 'cancelling'
											? 'stopping…'
											: job.progress.total
												? `${job.progress.step ?? 0}/${job.progress.total}`
												: 'loading…'}</span
									>
									<button
										type="button"
										class="text-red-600 hover:underline"
										onclick={() => cancelJob(job.id)}>Stop</button
									>
								{:else}
									<span
										class:text-red-600={job.status === 'failed'}
										class:text-gray-500={job.status !== 'failed'}
										title={job.error ?? ''}
										>{job.status === 'completed' ? 'no images' : job.status}</span
									>
									{#if job.retryable}<button
											type="button"
											class="text-indigo-600 hover:underline"
											onclick={() => retryJob(job.id)}>Retry</button
										>{/if}
								{/if}
							</div>
						</div>
					</div>
				{/if}
			{/each}
		</div>
		{#if failed}
			<p class="mt-2 text-xs break-words text-red-600">{failed.label}: {failed.error}</p>
		{/if}
	</div>
{/if}
