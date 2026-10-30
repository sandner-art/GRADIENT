<!-- frontend/src/lib/components/lab/AttentionTool.svelte -->
<!-- Attention maps: ConceptAttention for DiT models, cross-attention for UNet models.
     Picks the method from the model family; the user can override it. -->
<script lang="ts">
	import AttentionMaps from './AttentionMaps.svelte';
	import ConceptAttention from './ConceptAttention.svelte';
	import { lab, modelFamilies } from '#lib/lab/labState.svelte.js';
	import { attentionMethodFor } from '#lib/lab/math.js';
	import * as ui from '#lib/lab/ui.js';

	type Method = 'concept' | 'cross';
	let chosen = $state<{ model: string; method: Method } | null>(null);

	const family = $derived(modelFamilies[lab.base.model_id] ?? null);
	const suggested = $derived(attentionMethodFor(family));
	// a manual choice sticks until the model changes
	const method = $derived<Method>(
		chosen && chosen.model === lab.base.model_id ? chosen.method : (suggested ?? 'concept')
	);

	const options: { id: Method; label: string; hint: string }[] = [
		{ id: 'concept', label: 'ConceptAttention', hint: 'FLUX.1, FLUX.2, SD 3.5' },
		{ id: 'cross', label: 'Cross-attention', hint: 'SD 1.x, SDXL' }
	];
</script>

<div class="space-y-4">
	<div class="flex flex-wrap items-center gap-2">
		<div class="inline-flex rounded-md shadow-sm" role="group" aria-label="Attention method">
			{#each options as o, i (o.id)}
				<button
					type="button"
					aria-pressed={method === o.id}
					class="border border-gray-300 px-3 py-1.5 text-sm font-medium dark:border-gray-600"
					class:rounded-l-md={i === 0}
					class:rounded-r-md={i === options.length - 1}
					class:bg-indigo-600={method === o.id}
					class:text-white={method === o.id}
					class:bg-white={method !== o.id}
					class:text-gray-700={method !== o.id}
					class:dark:bg-gray-700={method !== o.id}
					class:dark:text-gray-200={method !== o.id}
					onclick={() => (chosen = { model: lab.base.model_id, method: o.id })}
				>
					{o.label} <span class="font-normal opacity-75">({o.hint})</span>
				</button>
			{/each}
		</div>
		{#if family}
			<span class={ui.muted}>
				{#if suggested}
					Model family <code>{family}</code>: {suggested === 'concept'
						? 'ConceptAttention'
						: 'cross-attention'} fits.
				{:else}
					Model family <code>{family}</code> isn't supported by either method yet; token ablation works
					for every model.
				{/if}
			</span>
		{/if}
	</div>

	<div hidden={method !== 'concept'}><ConceptAttention /></div>
	<div hidden={method !== 'cross'}><AttentionMaps /></div>
</div>
