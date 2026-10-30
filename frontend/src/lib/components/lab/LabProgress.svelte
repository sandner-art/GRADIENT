<!-- frontend/src/lib/components/lab/LabProgress.svelte -->
<!-- Shows the running Lab job and the sampler's step progress (polls /system/status). -->
<script lang="ts">
	import { get } from '#lib/apiClient.js';
	import { lab } from '#lib/lab/labState.svelte.js';
	import type { RuntimeStatus } from '#lib/types.js';
	import { formatDuration } from '#lib/utils/format.js';

	let status = $state<RuntimeStatus | null>(null);
	let elapsed = $state(0);

	$effect(() => {
		if (!lab.busy) {
			status = null;
			return;
		}
		const started = Date.now();
		let stopped = false;
		const tick = async () => {
			elapsed = (Date.now() - started) / 1000;
			try {
				const s = await get<RuntimeStatus>('/system/status');
				if (!stopped) status = s;
			} catch {
				/* the job's own request reports errors */
			}
		};
		tick();
		const timer = setInterval(tick, 700);
		return () => {
			stopped = true;
			clearInterval(timer);
		};
	});

	const current = $derived(status?.current ?? null);
	const fraction = $derived(
		current && current.total_steps ? Math.min(1, current.step / current.total_steps) : null
	);
</script>

{#if lab.busy}
	<div
		class="rounded-lg border border-indigo-200 bg-indigo-50 p-3 text-sm text-indigo-900 dark:border-indigo-800 dark:bg-indigo-950 dark:text-indigo-100"
		role="status"
		aria-live="polite"
	>
		<div class="flex items-center justify-between gap-2">
			<span class="flex items-center gap-2 font-medium">
				<span
					class="inline-block h-3 w-3 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent"
				></span>
				{lab.busy}
			</span>
			<span class="tabular-nums">{formatDuration(elapsed)}</span>
		</div>
		<div class="mt-2 h-1.5 overflow-hidden rounded bg-indigo-100 dark:bg-indigo-900">
			{#if fraction !== null}
				<div class="h-full bg-indigo-500 transition-all" style:width="{fraction * 100}%"></div>
			{:else}
				<div class="h-full w-1/3 animate-pulse bg-indigo-400"></div>
			{/if}
		</div>
		{#if current}
			<p class="mt-1 text-xs opacity-80">
				{current.task}: step {current.step}/{current.total_steps}
				{#if status && status.queue_depth > 1}· {status.queue_depth - 1} more queued{/if}
			</p>
		{:else}
			<p class="mt-1 text-xs opacity-80">
				Loading the model or computing (the first run of a model takes longest).
			</p>
		{/if}
	</div>
{/if}
