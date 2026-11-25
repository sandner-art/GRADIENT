<!-- frontend/src/lib/components/lab/LabWorkbench.svelte -->
<!-- The Lab tab: shared base settings on the left, one research tool on the right. -->
<script lang="ts">
	import AttentionTool from './AttentionTool.svelte';
	import Explorer from './Explorer.svelte';
	import LabBaseSettings from './LabBaseSettings.svelte';
	import LabProgress from './LabProgress.svelte';
	import ModulationLanes from './ModulationLanes.svelte';
	import QuantReport from './QuantReport.svelte';
	import StyleCheck from './StyleCheck.svelte';
	import TaylorPreview from './TaylorPreview.svelte';
	import TokenAblation from './TokenAblation.svelte';
	import { lab, type LabTool } from '#lib/lab/labState.svelte.js';

	const tools: { id: LabTool; label: string; hint: string }[] = [
		{ id: 'explore', label: 'Explorer', hint: 'Travel through the model, region by region' },
		{ id: 'taylor', label: 'Taylor preview', hint: 'Preview other CFG values instantly' },
		{ id: 'modulation', label: 'Modulation lanes', hint: 'Change things for part of the run' },
		{
			id: 'attention',
			label: 'Attention maps',
			hint: 'Where each word or concept is in the image'
		},
		{ id: 'ablation', label: 'Token ablation', hint: 'How much each word matters' },
		{
			id: 'quantization',
			label: 'Quantization check',
			hint: 'What loading the model quantized changes'
		},
		{ id: 'style', label: 'Style check', hint: 'Which styles work, and where in the prompt' }
	];
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
					aria-selected={lab.tool === t.id}
					title={t.hint}
					class="rounded-full border px-3 py-1 text-sm font-medium transition"
					class:border-indigo-500={lab.tool === t.id}
					class:bg-indigo-600={lab.tool === t.id}
					class:text-white={lab.tool === t.id}
					class:border-gray-300={lab.tool !== t.id}
					class:text-gray-700={lab.tool !== t.id}
					class:dark:border-gray-600={lab.tool !== t.id}
					class:dark:text-gray-200={lab.tool !== t.id}
					onclick={() => (lab.tool = t.id)}>{t.label}</button
				>
			{/each}
		</div>

		<!-- Keep every tool mounted so results survive switching between them. -->
		<div hidden={lab.tool !== 'explore'}><Explorer /></div>
		<div hidden={lab.tool !== 'taylor'}><TaylorPreview /></div>
		<div hidden={lab.tool !== 'modulation'}><ModulationLanes /></div>
		<div hidden={lab.tool !== 'attention'}><AttentionTool /></div>
		<div hidden={lab.tool !== 'ablation'}><TokenAblation /></div>
		<div hidden={lab.tool !== 'quantization'}><QuantReport /></div>
		<div hidden={lab.tool !== 'style'}><StyleCheck /></div>
	</section>
</div>
