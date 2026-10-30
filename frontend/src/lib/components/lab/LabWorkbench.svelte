<!-- frontend/src/lib/components/lab/LabWorkbench.svelte -->
<!-- The Lab tab: shared base settings on the left, one research tool on the right. -->
<script lang="ts">
	import AttentionMaps from './AttentionMaps.svelte';
	import LabBaseSettings from './LabBaseSettings.svelte';
	import LabProgress from './LabProgress.svelte';
	import ModulationLanes from './ModulationLanes.svelte';
	import TaylorPreview from './TaylorPreview.svelte';
	import TokenAblation from './TokenAblation.svelte';

	type Tool = 'taylor' | 'modulation' | 'attention' | 'ablation';
	const tools: { id: Tool; label: string; hint: string }[] = [
		{ id: 'taylor', label: 'Taylor preview', hint: 'Preview other CFG values instantly' },
		{ id: 'modulation', label: 'Modulation lanes', hint: 'Change things for part of the run' },
		{ id: 'attention', label: 'Attention maps', hint: 'Where each word acts' },
		{ id: 'ablation', label: 'Token ablation', hint: 'How much each word matters' }
	];
	let tool = $state<Tool>('taylor');
</script>

<div class="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
	<section class="space-y-4 lg:sticky lg:top-4 lg:self-start">
		<LabBaseSettings />
		<LabProgress />
	</section>

	<section class="space-y-4">
		<div class="flex flex-wrap gap-2" role="tablist" aria-label="Lab tools">
			{#each tools as t (t.id)}
				<button
					type="button"
					role="tab"
					aria-selected={tool === t.id}
					title={t.hint}
					class="rounded-full border px-3 py-1 text-sm font-medium transition"
					class:border-indigo-500={tool === t.id}
					class:bg-indigo-600={tool === t.id}
					class:text-white={tool === t.id}
					class:border-gray-300={tool !== t.id}
					class:text-gray-700={tool !== t.id}
					class:dark:border-gray-600={tool !== t.id}
					class:dark:text-gray-200={tool !== t.id}
					onclick={() => (tool = t.id)}>{t.label}</button
				>
			{/each}
		</div>

		<!-- Keep every tool mounted so results survive switching between them. -->
		<div hidden={tool !== 'taylor'}><TaylorPreview /></div>
		<div hidden={tool !== 'modulation'}><ModulationLanes /></div>
		<div hidden={tool !== 'attention'}><AttentionMaps /></div>
		<div hidden={tool !== 'ablation'}><TokenAblation /></div>
	</section>
</div>
