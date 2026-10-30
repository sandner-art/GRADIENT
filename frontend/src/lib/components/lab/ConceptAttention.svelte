<!-- frontend/src/lib/components/lab/ConceptAttention.svelte -->
<!-- ConceptAttention (FLUX.1, FLUX.2, SD3/3.5): where each concept is in the image, from the
     attention outputs of the double-stream blocks. Concepts don't have to be in the prompt. -->
<script lang="ts">
	import { post } from '#lib/apiClient.js';
	import { baseProblem, baseRequest, lab, runLabJob } from '#lib/lab/labState.svelte.js';
	import { cachedLayers, fetchLayers } from '#lib/lab/layers.svelte.js';
	import { laneSteps, parseWordList } from '#lib/lab/math.js';
	import * as ui from '#lib/lab/ui.js';
	import type { ConceptAttentionRequest, ConceptAttentionResponse } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let conceptsText = $state('');
	let blocks = $state('');
	let start = $state(0);
	let end = $state(1);
	let show = $state<'overlay' | 'heatmap'>('overlay');
	let result = $state<ConceptAttentionResponse | null>(null);
	let resultModel = $state('');
	let selected = $state<string | null>(null);
	let error = $state<string | null>(null);
	let layerError = $state<string | null>(null);

	const known = $derived(cachedLayers(lab.base.model_id));
	const [firstStep, lastStep] = $derived(laneSteps(start, end, lab.base.num_inference_steps));
	const focus = $derived(result?.concepts.find((c) => c.concept === selected) ?? null);
	const pct = (v: number) => `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%`;

	async function loadBlocks() {
		layerError = null;
		await runLabJob('Loading the model to read its blocks…', async () => {
			try {
				await fetchLayers(lab.base.model_id);
			} catch (e) {
				layerError = apiErrorMessage(e);
			}
		});
	}

	async function run() {
		error = baseProblem();
		if (!error && !(end > start)) error = 'The step range is empty.';
		if (error) return;
		const body: ConceptAttentionRequest = {
			base_request: baseRequest(),
			concepts: parseWordList(conceptsText),
			blocks: blocks.trim() || null,
			start,
			end
		};
		await runLabJob('Running ConceptAttention…', async () => {
			try {
				result = await post<ConceptAttentionResponse>('/lab/concept-attention', body);
				resultModel = body.base_request.model_id;
				selected = result.concepts[0]?.concept ?? null;
			} catch (e) {
				error = apiErrorMessage(e);
			}
		});
	}
</script>

<div class="space-y-4">
	<div class={ui.card}>
		<h2 class={ui.heading}>ConceptAttention</h2>
		<p class="{ui.muted} mt-1">
			Locates concepts in the image from the attention outputs of the double-stream blocks (FLUX.1,
			FLUX.2, SD 3.5). Each concept runs as an extra token through the text stream; the image is not
			changed. Concepts don't have to appear in the prompt. Adding “background” helps.
		</p>

		<div class="mt-3 space-y-3">
			<div>
				<label for="ca_concepts" class={ui.label}
					>Concepts (comma-separated, empty = words of the prompt + background)</label
				>
				<input
					id="ca_concepts"
					class={ui.input}
					placeholder="lighthouse, sea, sky, rocks, background"
					bind:value={conceptsText}
				/>
			</div>
			<div>
				<label for="ca_blocks" class={ui.label}>Blocks</label>
				<div class="flex gap-2">
					<input
						id="ca_blocks"
						class={ui.input}
						placeholder="default: the deeper half"
						bind:value={blocks}
					/>
					<button
						type="button"
						class={ui.secondaryButton}
						disabled={!!lab.busy || !lab.base.model_id}
						onclick={loadBlocks}>Count blocks</button
					>
				</div>
				<p class={ui.help}>
					Double-stream block indices, e.g. <code>9-18</code>, <code>12,15,18</code> or
					<code>all</code>. Deeper blocks localize better; <code>all</code> is worth a try too.
					{#if known}
						{#if known.concept_blocks}
							This model has <strong>{known.concept_blocks}</strong> blocks (0–{known.concept_blocks -
								1}).
						{:else}
							This model has no double-stream blocks; use cross-attention maps or token ablation.
						{/if}
					{/if}
				</p>
				{#if layerError}<p class="mt-1 text-xs text-red-600">{layerError}</p>{/if}
			</div>
			<div class="grid grid-cols-2 gap-3">
				<div>
					<label for="ca_start" class={ui.label}>From {Math.round(start * 100)}%</label>
					<input
						id="ca_start"
						type="range"
						min="0"
						max="1"
						step="0.05"
						class="w-full"
						bind:value={start}
					/>
				</div>
				<div>
					<label for="ca_end" class={ui.label}>To {Math.round(end * 100)}%</label>
					<input
						id="ca_end"
						type="range"
						min="0"
						max="1"
						step="0.05"
						class="w-full"
						bind:value={end}
					/>
				</div>
			</div>
			<p class={ui.help}>
				Averages steps {firstStep + 1}–{lastStep} of {lab.base.num_inference_steps}.
			</p>
			<button type="button" class={ui.primaryButton} disabled={!!lab.busy} onclick={run}
				>Compute concept maps</button
			>
		</div>
		{#if error}<div class="{ui.errorBox} mt-3" role="alert">{error}</div>{/if}
	</div>

	{#if result}
		<div class={ui.card}>
			<div class="mb-3 flex flex-wrap items-center justify-between gap-2">
				<p class={ui.muted}>
					{resultModel} · seed {result.seed} · blocks {result
						.blocks_recorded[0]}–{result.blocks_recorded.at(-1)} of
					{result.num_blocks} · {result.steps_recorded} steps · {(
						result.performance_ms / 1000
					).toFixed(1)} s
				</p>
				<div class="flex gap-1" role="group" aria-label="Map style">
					<button
						type="button"
						class={ui.secondaryButton}
						class:ring-2={show === 'overlay'}
						onclick={() => (show = 'overlay')}>Overlay</button
					>
					<button
						type="button"
						class={ui.secondaryButton}
						class:ring-2={show === 'heatmap'}
						onclick={() => (show = 'heatmap')}>Heatmap</button
					>
				</div>
			</div>

			<div class="grid grid-cols-2 gap-3">
				<figure>
					<img src={ui.png(result.image_base64)} alt="Generated" class="h-auto w-full rounded" />
					<figcaption class={ui.help}>Image</figcaption>
				</figure>
				<figure>
					<img
						src={ui.png(result.segmentation_base64)}
						alt="Segmentation: each patch coloured by its strongest concept"
						class="h-auto w-full rounded"
					/>
					<figcaption class={ui.help}>Strongest concept per patch</figcaption>
				</figure>
			</div>

			<ul class="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-sm" aria-label="Legend">
				{#each result.concepts as c (c.concept)}
					<li class="flex items-center gap-1.5">
						<span class="inline-block h-3 w-3 rounded-sm" style:background-color={c.color}></span>
						{c.concept}
						<span class="text-gray-500 tabular-nums dark:text-gray-400">{pct(c.area)}</span>
					</li>
				{/each}
			</ul>

			<div class="mt-4 grid grid-cols-1 gap-3 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
				{#if focus}
					<figure>
						<img
							src={ui.png(show === 'overlay' ? focus.overlay_base64 : focus.heatmap_base64)}
							alt="Concept map for {focus.concept}"
							class="h-auto w-full rounded"
						/>
						<figcaption class={ui.help}>
							“{focus.concept}” · wins {pct(focus.area)} of the image · mean probability
							{pct(focus.mean_probability)}
						</figcaption>
					</figure>
				{/if}
				<div class="grid grid-cols-3 content-start gap-2 sm:grid-cols-4">
					{#each result.concepts as c (c.concept)}
						<button
							type="button"
							class="overflow-hidden rounded border border-gray-200 text-left dark:border-gray-600"
							class:ring-2={c.concept === selected}
							class:ring-indigo-500={c.concept === selected}
							onclick={() => (selected = c.concept)}
						>
							<img
								src={ui.png(show === 'overlay' ? c.overlay_base64 : c.heatmap_base64)}
								alt="Concept map for {c.concept}"
								class="h-auto w-full"
							/>
							<span class="flex items-center gap-1 truncate px-1 py-1 text-xs font-medium">
								<span
									class="inline-block h-2 w-2 flex-none rounded-sm"
									style:background-color={c.color}
								></span>
								{c.concept}
							</span>
						</button>
					{/each}
				</div>
			</div>
			<p class={ui.help}>
				Maps are probabilities over the concepts you listed (they add up to 1 in every patch), so
				they share one scale. A concept that is missing from the image still takes some probability
				when nothing else fits better.
			</p>
		</div>
	{/if}
</div>
