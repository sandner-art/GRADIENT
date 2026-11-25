<!-- frontend/src/lib/components/MemoryButton.svelte -->
<!-- What is loaded (image models, vision models for metrics, LLMs) and the VRAM in use;
     free all of it, or one kind. Unloading waits for the running GPU job. -->
<script lang="ts">
	import { get, post } from '#lib/apiClient.js';
	import { notify } from '#lib/stores/ui.svelte.js';
	import { apiErrorMessage } from '#lib/utils/format.js';
	import { freedMessage, type MemoryReport, type UnloadResult } from '#lib/memory.js';

	let open = $state(false);
	let report = $state<MemoryReport | null>(null);
	let busy = $state(false);
	let error = $state<string | null>(null);

	async function refresh() {
		try {
			report = await get<MemoryReport>('/system/memory');
			error = null;
		} catch (e) {
			error = apiErrorMessage(e);
		}
	}

	async function toggle() {
		open = !open;
		if (open) await refresh();
	}

	async function unload(targets: string[] | null) {
		busy = true;
		try {
			const r = await post<UnloadResult>('/system/unload', targets ? { targets } : {});
			notify(freedMessage(r), 'success', undefined, 4000);
			await refresh();
		} catch (e) {
			error = apiErrorMessage(e);
		} finally {
			busy = false;
		}
	}

	const groups = $derived(
		report
			? ([
					['diffusion', 'Image models', report.loaded.diffusion],
					['vision', 'Vision models (metrics)', report.loaded.vision],
					['llm', 'LLMs', report.loaded.llm]
				] as const)
			: []
	);
	const anything = $derived(groups.some(([, , items]) => items.length));
	const mb = (v: number | null | undefined) =>
		v ? (v >= 1024 ? `${(v / 1024).toFixed(1)} GB` : `${v} MB`) : '';
</script>

<svelte:window
	onkeydown={(e) => {
		if (e.key === 'Escape' && open) open = false;
	}}
/>

<div class="relative">
	<button
		type="button"
		class="rounded-md border border-gray-300 px-2.5 py-1.5 text-sm text-gray-600 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-300 dark:hover:bg-gray-800"
		aria-expanded={open}
		title="Loaded models and VRAM"
		onclick={toggle}>Memory</button
	>
	{#if open}
		<button
			type="button"
			class="fixed inset-0 z-30 cursor-default"
			aria-label="Close the memory panel"
			onclick={() => (open = false)}
		></button>
		<div
			class="absolute right-0 z-40 mt-2 w-80 rounded-md border border-gray-200 bg-white p-3 text-sm shadow-lg dark:border-gray-700 dark:bg-gray-900"
			role="dialog"
			aria-label="Memory"
		>
			{#if report?.vram}
				<p class="mb-2">
					VRAM: {(report.vram.total_gb - report.vram.free_gb).toFixed(1)} of {report.vram.total_gb}
					GB in use
					<span class="text-xs text-gray-500">(PyTorch holds {report.vram.reserved_gb} GB)</span>
				</p>
			{:else if report}
				<p class="mb-2 text-xs text-gray-500">{report.device}: no VRAM figures (not CUDA)</p>
			{/if}
			{#each groups as [kind, label, items] (kind)}
				{#if items.length}
					<div class="mb-2">
						<div class="flex items-center justify-between">
							<span class="font-medium">{label}</span>
							<button
								type="button"
								class="text-xs text-indigo-600 hover:underline disabled:opacity-50 dark:text-indigo-400"
								disabled={busy}
								onclick={() => unload([kind])}>Unload</button
							>
						</div>
						<ul class="text-xs text-gray-600 dark:text-gray-400">
							{#each items as item, i (i)}
								<li>
									{item.name}{item.detail ? ` · ${item.detail}` : ''}{item.vram_mb
										? ` · ${mb(item.vram_mb)}`
										: ''}
								</li>
							{/each}
						</ul>
					</div>
				{/if}
			{/each}
			{#if report && !anything}<p class="mb-2 text-xs text-gray-500">No models loaded.</p>{/if}
			<div
				class="flex items-center justify-between border-t border-gray-200 pt-2 dark:border-gray-700"
			>
				<button
					type="button"
					class="rounded bg-indigo-600 px-2.5 py-1 text-white hover:bg-indigo-700 disabled:opacity-50"
					disabled={busy || !anything}
					onclick={() => unload(null)}>{busy ? 'Freeing…' : 'Free VRAM'}</button
				>
				<button
					type="button"
					class="text-xs text-gray-500 hover:underline disabled:opacity-50"
					disabled={busy}
					title="Explorer paths, word vocabularies and tokenizers (memory, not VRAM)"
					onclick={() => unload(['caches'])}>Clear caches</button
				>
			</div>
			<p class="mt-1 text-xs text-gray-500">
				Models load again when a render or the Lab needs them. A running job finishes first.
			</p>
			{#if error}<p class="mt-1 text-xs text-red-600">{error}</p>{/if}
		</div>
	{/if}
</div>
