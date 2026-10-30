<!-- frontend/src/lib/components/lab/AttentionMaps.svelte -->
<!-- Per-word cross-attention maps, recorded only on the chosen layers and steps. -->
<script lang="ts">
	import { post } from '#lib/apiClient.js';
	import { baseProblem, baseRequest, lab, runLabJob } from '#lib/lab/labState.svelte.js';
	import { cachedLayers, fetchLayers } from '#lib/lab/layers.svelte.js';
	import { countMatches, laneSteps, parseWordList } from '#lib/lab/math.js';
	import * as ui from '#lib/lab/ui.js';
	import type { AttentionMapRequest, AttentionMapResponse } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let wordsText = $state('');
	let layers = $state('*attn2');
	let start = $state(0);
	let end = $state(1);
	let show = $state<'overlay' | 'heatmap'>('overlay');
	let result = $state<AttentionMapResponse | null>(null);
	let resultModel = $state('');
	// index into result.maps: a prompt can contain the same word twice ("a ... a ...")
	let selected = $state(0);
	let error = $state<string | null>(null);
	let layerError = $state<string | null>(null);

	const known = $derived(cachedLayers(lab.base.model_id));
	const matched = $derived(countMatches(known, layers));
	const [firstStep, lastStep] = $derived(laneSteps(start, end, lab.base.num_inference_steps));
	const maxShare = $derived(Math.max(1e-9, ...(result?.maps.map((m) => m.attention_share) ?? [0])));
	const focus = $derived(result?.maps[selected] ?? null);

	async function listLayers() {
		layerError = null;
		if (!lab.base.model_id) return;
		await runLabJob('Loading the model to list its layers…', async () => {
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
		const body: AttentionMapRequest = {
			base_request: baseRequest(),
			words: parseWordList(wordsText),
			layers,
			start,
			end
		};
		await runLabJob('Recording cross-attention…', async () => {
			try {
				result = await post<AttentionMapResponse>('/lab/attention-maps', body);
				resultModel = body.base_request.model_id;
				selected = 0;
			} catch (e) {
				error = apiErrorMessage(e);
			}
		});
	}
</script>

<div class="space-y-4">
	<div class={ui.card}>
		<h2 class={ui.heading}>Cross-attention maps</h2>
		<p class="{ui.muted} mt-1">
			Where each word of the prompt is attended to. Probabilities are computed only on the selected
			cross-attention layers and steps; the rest of the model runs on its fast attention, and the
			image matches a normal render. Available for SD 1.x/SDXL (UNet models).
		</p>

		<div class="mt-3 space-y-3">
			<div>
				<label for="att_words" class={ui.label}>Words (comma-separated, empty = every word)</label>
				<input
					id="att_words"
					class={ui.input}
					placeholder="lighthouse, coast, sunset"
					bind:value={wordsText}
				/>
			</div>
			<div>
				<label for="att_layers" class={ui.label}>Layers</label>
				<div class="flex gap-2">
					<input
						id="att_layers"
						class={ui.input}
						list="att_layer_suggestions"
						bind:value={layers}
						placeholder="*attn2"
					/>
					<button
						type="button"
						class={ui.secondaryButton}
						disabled={!!lab.busy || !lab.base.model_id}
						onclick={listLayers}>List layers</button
					>
				</div>
				<datalist id="att_layer_suggestions">
					{#each known?.suggestions ?? ['*attn2', 'down_blocks.*attn2', 'mid_block.*attn2', 'up_blocks.*attn2'] as s (s)}
						<option value={s}></option>
					{/each}
				</datalist>
				<p class={ui.help}>
					Glob over layer names; comma-separate several patterns.
					{#if matched !== null}
						<strong>{matched}</strong> of {known?.layers.length} attention layers match.
					{/if}
				</p>
				{#if layerError}<p class="mt-1 text-xs text-red-600">{layerError}</p>{/if}
			</div>
			<div class="grid grid-cols-2 gap-3">
				<div>
					<label for="att_start" class={ui.label}>From {Math.round(start * 100)}%</label>
					<input
						id="att_start"
						type="range"
						min="0"
						max="1"
						step="0.05"
						class="w-full"
						bind:value={start}
					/>
				</div>
				<div>
					<label for="att_end" class={ui.label}>To {Math.round(end * 100)}%</label>
					<input
						id="att_end"
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
				Records steps {firstStep + 1}–{lastStep} of {lab.base.num_inference_steps}. Early steps show
				layout, late steps show detail.
			</p>
			<button type="button" class={ui.primaryButton} disabled={!!lab.busy} onclick={run}
				>Compute maps</button
			>
		</div>
		{#if error}<div class="{ui.errorBox} mt-3" role="alert">{error}</div>{/if}
	</div>

	{#if result}
		<div class={ui.card}>
			<div class="mb-3 flex flex-wrap items-center justify-between gap-2">
				<p class={ui.muted}>
					{resultModel} · seed {result.seed} · {result.layers_recorded.length} layers ·
					{(result.performance_ms / 1000).toFixed(1)} s
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
				{#if focus}
					<figure>
						<img
							src={ui.img(show === 'overlay' ? focus.overlay_base64 : focus.heatmap_base64)}
							alt="Attention map for {focus.word}"
							class="h-auto w-full rounded"
						/>
						<figcaption class={ui.help}>
							“{focus.word}” · {(focus.attention_share * 100).toFixed(2)}% of attention · tokens
							{focus.token_positions.join(', ')}
						</figcaption>
					</figure>
				{/if}
			</div>

			<div class="mt-4 grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6">
				{#each result.maps as m, i (`${m.word}@${m.token_positions[0]}`)}
					<button
						type="button"
						class="overflow-hidden rounded border border-gray-200 text-left dark:border-gray-600"
						class:ring-2={i === selected}
						class:ring-indigo-500={i === selected}
						onclick={() => (selected = i)}
					>
						<img
							src={ui.img(show === 'overlay' ? m.overlay_base64 : m.heatmap_base64)}
							alt="Attention map for {m.word}"
							class="h-auto w-full"
						/>
						<span class="block truncate px-1 pt-1 text-xs font-medium">{m.word}</span>
						<span class="mx-1 mb-1 block h-1 rounded bg-gray-200 dark:bg-gray-700">
							<span
								class="block h-1 rounded bg-indigo-500"
								style:width="{(m.attention_share / maxShare) * 100}%"
							></span>
						</span>
					</button>
				{/each}
			</div>
			<p class={ui.help}>
				Each map is normalized on its own; the bar compares how much attention the words receive.
			</p>
		</div>
	{/if}
</div>
