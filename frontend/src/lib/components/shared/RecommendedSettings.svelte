<!-- frontend/src/lib/components/shared/RecommendedSettings.svelte -->
<!-- Shows "Use recommended settings" when the selected model has catalog defaults
     (e.g. FLUX.2 klein: 1024x1024, 4 steps, CFG 1). -->
<script lang="ts">
	import { get } from '#lib/apiClient.js';
	import type { GenerationDefaults, LocalModel } from '#lib/types.js';
	import { plainSchedulerName } from '#lib/utils/format.js';

	interface Props {
		modelId: string | null | undefined;
		onapply: (defaults: GenerationDefaults & { scheduler: string | null }) => void;
	}
	let { modelId, onapply }: Props = $props();

	let models = $state<LocalModel[]>([]);

	$effect(() => {
		Promise.all([
			get<LocalModel[]>('/models/local?type=diffusers'),
			get<LocalModel[]>('/models/local?type=checkpoints')
		])
			.then(([a, b]) => (models = [...a, ...b]))
			.catch(() => (models = []));
	});

	const current = $derived(models.find((m) => m.name === modelId));
	const defaults = $derived(current?.defaults ?? null);
	const summary = $derived(
		defaults
			? [
					defaults.width && defaults.height ? `${defaults.width}×${defaults.height}` : null,
					defaults.steps ? `${defaults.steps} steps` : null,
					defaults.guidance_scale !== null && defaults.guidance_scale !== undefined
						? `CFG ${defaults.guidance_scale}`
						: null
				]
					.filter(Boolean)
					.join(', ')
			: ''
	);
</script>

{#if defaults && summary}
	<div
		class="flex flex-wrap items-center justify-between gap-2 rounded-md border border-indigo-200 bg-indigo-50 px-3 py-2 text-xs text-indigo-900 dark:border-indigo-800 dark:bg-indigo-950 dark:text-indigo-200"
	>
		<span>
			Recommended for {current?.family ?? 'this model'}: {summary}
		</span>
		<button
			type="button"
			class="rounded bg-indigo-600 px-2 py-0.5 font-medium text-white hover:bg-indigo-700"
			onclick={() => onapply({ ...defaults, scheduler: plainSchedulerName(defaults.scheduler) })}
		>
			Use these
		</button>
	</div>
{/if}
