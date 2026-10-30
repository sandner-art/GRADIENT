<!-- frontend/src/lib/components/lab/ModulationLanes.svelte -->
<!-- Structural modulation: lanes on the denoising timeline that switch or blend the prompt,
     change guidance, scale attention layers or a LoRA weight for part of the run. -->
<script lang="ts">
	import { post } from '#lib/apiClient.js';
	import { baseProblem, baseRequest, lab, runLabJob } from '#lib/lab/labState.svelte.js';
	import { cachedLayers, fetchLayers } from '#lib/lab/layers.svelte.js';
	import { cleanLane, countMatches, laneProblem, laneSteps } from '#lib/lab/math.js';
	import * as ui from '#lib/lab/ui.js';
	import type {
		ImageResult,
		ModulationLane,
		ModulationLaneType,
		Text2ImageResponse
	} from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	const TYPES: { id: ModulationLaneType; label: string; color: string }[] = [
		{ id: 'prompt', label: 'Prompt', color: 'bg-indigo-500' },
		{ id: 'guidance', label: 'Guidance', color: 'bg-amber-500' },
		{ id: 'attention_scale', label: 'Attention scale', color: 'bg-teal-500' },
		{ id: 'lora_weight', label: 'LoRA weight', color: 'bg-pink-500' }
	];
	const colorOf = (t: ModulationLaneType) => TYPES.find((x) => x.id === t)?.color ?? 'bg-gray-500';

	const PRESETS: { label: string; hint: string; lanes: () => ModulationLane[] }[] = [
		{
			label: 'Layout from another prompt',
			hint: 'The first 30% of the steps follow a different prompt, which sets the composition.',
			lanes: () => [
				{ type: 'prompt', start: 0, end: 0.3, prompt: 'a castle on a hill', mode: 'replace' }
			]
		},
		{
			label: 'Blend in a style late',
			hint: 'From 40% on, mix in a second prompt that mostly changes texture and color.',
			lanes: () => [
				{
					type: 'prompt',
					start: 0.4,
					end: 1,
					prompt: 'an oil painting, thick impasto brush strokes',
					mode: 'blend',
					value: 0.5
				}
			]
		},
		{
			label: 'Guidance only early',
			hint: 'High CFG for structure, then CFG 1.5 for natural detail (SD/SDXL/SD3).',
			lanes: () => [{ type: 'guidance', start: 0.6, end: 1, value: 1.5 }]
		},
		{
			label: 'Mute up-block text attention',
			hint: 'Scale the decoder’s cross-attention to 0 for the first half (UNet models).',
			lanes: () => [
				{ type: 'attention_scale', start: 0, end: 0.5, value: 0, targets: 'up_blocks.*attn2' }
			]
		},
		{
			label: 'LoRA fades in',
			hint: 'No LoRA for layout, full LoRA for detail (pick a LoRA in the base settings).',
			lanes: () => [{ type: 'lora_weight', start: 0, end: 0.4, value: 0, lora_index: 0 }]
		}
	];

	let baseline = $state<{ key: string; image: ImageResult } | null>(null);
	let modulated = $state<ImageResult | null>(null);
	let skipBaseline = $state(false);
	let error = $state<string | null>(null);
	let layerError = $state<string | null>(null);

	const steps = $derived(lab.base.num_inference_steps);
	const loraCount = $derived(lab.base.lora ? 1 : 0);
	const problems = $derived(lab.lanes.map((l) => laneProblem(l, loraCount)));
	const known = $derived(cachedLayers(lab.base.model_id));

	function addLane(type: ModulationLaneType = 'prompt') {
		const defaults: Record<ModulationLaneType, ModulationLane> = {
			prompt: { type, start: 0, end: 0.3, prompt: '', mode: 'replace' },
			guidance: { type, start: 0.5, end: 1, value: 2 },
			attention_scale: { type, start: 0, end: 0.5, value: 0.5, targets: '*attn2' },
			lora_weight: { type, start: 0, end: 0.5, value: 0, lora_index: 0 }
		};
		lab.lanes.push(defaults[type]);
	}

	function changeType(i: number, type: ModulationLaneType) {
		const { start, end } = lab.lanes[i];
		addLane(type);
		const fresh = lab.lanes.pop()!;
		lab.lanes[i] = { ...fresh, start, end };
	}

	function numberOrNull(v: string): number | null {
		return v === '' ? null : Number(v);
	}

	async function listLayers() {
		layerError = null;
		await runLabJob('Loading the model to list its layers…', async () => {
			try {
				await fetchLayers(lab.base.model_id);
			} catch (e) {
				layerError = apiErrorMessage(e);
			}
		});
	}

	async function compare() {
		error = baseProblem();
		if (!error && !lab.lanes.length) error = 'Add at least one lane.';
		if (!error) {
			const bad = problems.findIndex(Boolean);
			if (bad >= 0) error = `Lane ${bad + 1}: ${problems[bad]}`;
		}
		if (error) return;

		await runLabJob(
			skipBaseline ? 'Rendering with lanes…' : 'Rendering baseline and lanes…',
			async () => {
				try {
					let seed = lab.base.seed;
					const plain = baseRequest({ seed });
					const key = JSON.stringify(plain);
					if (!skipBaseline && (seed === null || baseline?.key !== key)) {
						const r = await post<Text2ImageResponse>('/generate/text-to-image', plain);
						seed = r.images[0].seed;
						baseline = { key: JSON.stringify(baseRequest({ seed })), image: r.images[0] };
					}
					const r = await post<Text2ImageResponse>(
						'/generate/text-to-image',
						baseRequest({
							seed: seed ?? baseline?.image.seed ?? null,
							modulation: lab.lanes.map(cleanLane)
						})
					);
					modulated = r.images[0];
				} catch (e) {
					error = apiErrorMessage(e);
				}
			}
		);
	}

	const applied = $derived(
		Object.keys((modulated?.parameters.modulation_applied as Record<string, boolean>) ?? {})
	);
	const notes = $derived((modulated?.parameters.modulation_notes as string[] | undefined) ?? []);
	const layerCounts = $derived(
		(modulated?.parameters.modulation_layers as Record<string, number> | undefined) ?? {}
	);
</script>

<div class="space-y-4">
	<div class={ui.card}>
		<h2 class={ui.heading}>Structural modulation</h2>
		<p class="{ui.muted} mt-1">
			Lanes change one thing for part of the denoising run. Early steps decide composition, late
			steps decide texture, so the same edit has very different effects depending on when it runs.
			Lanes also work in every generation request (<code>modulation</code> field of the API).
		</p>

		<div class="mt-3 flex flex-wrap gap-2">
			{#each PRESETS as p (p.label)}
				<button
					type="button"
					class={ui.secondaryButton}
					title={p.hint}
					onclick={() => (lab.lanes = p.lanes())}>{p.label}</button
				>
			{/each}
		</div>

		<!-- Timeline -->
		<div class="mt-4" aria-label="Lane timeline">
			<div class="mb-1 flex justify-between text-xs text-gray-500 dark:text-gray-400">
				<span>step 1 · noise, layout</span><span>step {steps} · detail</span>
			</div>
			<div class="space-y-1 rounded bg-gray-100 p-2 dark:bg-gray-900">
				{#each lab.lanes as lane, i (i)}
					{@const [a, b] = laneSteps(lane.start, lane.end, steps)}
					<div class="flex items-center gap-2">
						<span class="w-4 text-right text-[10px] font-medium text-gray-500 tabular-nums"
							>{i + 1}</span
						>
						<div class="relative h-4 flex-1 rounded bg-white dark:bg-gray-800">
							<div
								class="absolute top-0 h-4 rounded {colorOf(lane.type)}"
								class:opacity-40={!!problems[i]}
								style:left="{(a / steps) * 100}%"
								style:width="{((b - a) / steps) * 100}%"
								title="Lane {i + 1}: steps {a + 1}–{b}"
							></div>
						</div>
					</div>
				{:else}
					<p class="text-xs text-gray-500">No lanes.</p>
				{/each}
			</div>
		</div>

		<!-- Lane editor -->
		<div class="mt-4 space-y-3">
			{#each lab.lanes as lane, i (i)}
				<div class="rounded border border-gray-200 p-3 dark:border-gray-700">
					<div class="flex flex-wrap items-end gap-2">
						<span class="mb-1 inline-block h-3 w-3 rounded-full {colorOf(lane.type)}"></span>
						<div class="w-40">
							<label for="lane_type_{i}" class="text-xs text-gray-600 dark:text-gray-400"
								>Lane {i + 1}</label
							>
							<select
								id="lane_type_{i}"
								class={ui.smallInput}
								value={lane.type}
								onchange={(e) => changeType(i, e.currentTarget.value as ModulationLaneType)}
							>
								{#each TYPES as t (t.id)}
									<option value={t.id}>{t.label}</option>
								{/each}
							</select>
						</div>
						<div class="w-20">
							<label for="lane_start_{i}" class="text-xs text-gray-600 dark:text-gray-400"
								>From</label
							>
							<input
								id="lane_start_{i}"
								type="number"
								min="0"
								max="1"
								step="0.05"
								class={ui.smallInput}
								bind:value={lane.start}
							/>
						</div>
						<div class="w-20">
							<label for="lane_end_{i}" class="text-xs text-gray-600 dark:text-gray-400">To</label>
							<input
								id="lane_end_{i}"
								type="number"
								min="0"
								max="1"
								step="0.05"
								class={ui.smallInput}
								bind:value={lane.end}
							/>
						</div>
						{#if lane.type === 'prompt'}
							<div class="w-28">
								<label for="lane_mode_{i}" class="text-xs text-gray-600 dark:text-gray-400"
									>Mode</label
								>
								<select id="lane_mode_{i}" class={ui.smallInput} bind:value={lane.mode}>
									<option value="replace">Replace</option>
									<option value="blend">Blend</option>
								</select>
							</div>
						{/if}
						{#if lane.type !== 'prompt' || lane.mode === 'blend'}
							<div class="w-24">
								<label for="lane_value_{i}" class="text-xs text-gray-600 dark:text-gray-400">
									{lane.type === 'guidance'
										? 'CFG'
										: lane.type === 'attention_scale'
											? 'Scale'
											: lane.type === 'lora_weight'
												? 'Weight'
												: 'Mix (0–1)'}
								</label>
								<input
									id="lane_value_{i}"
									type="number"
									step="0.1"
									class={ui.smallInput}
									value={lane.value ?? ''}
									oninput={(e) => (lane.value = numberOrNull(e.currentTarget.value))}
								/>
							</div>
						{/if}
						<button
							type="button"
							class="ml-auto text-sm text-red-600 hover:underline dark:text-red-400"
							onclick={() => lab.lanes.splice(i, 1)}>Remove</button
						>
					</div>

					{#if lane.type === 'prompt'}
						<input
							class="{ui.smallInput} mt-2"
							placeholder="prompt used while the lane is active"
							aria-label="Lane {i + 1} prompt"
							bind:value={lane.prompt}
						/>
					{:else if lane.type === 'attention_scale'}
						<div class="mt-2 flex gap-2">
							<input
								class={ui.smallInput}
								list="lane_layer_suggestions"
								placeholder="up_blocks.*attn2"
								aria-label="Lane {i + 1} target layers"
								bind:value={lane.targets}
							/>
							<button
								type="button"
								class={ui.secondaryButton}
								disabled={!!lab.busy || !lab.base.model_id}
								onclick={listLayers}>List layers</button
							>
						</div>
						{#if known}
							<p class={ui.help}>
								{countMatches(known, lane.targets ?? '')} of {known.layers.length} layers match.
							</p>
						{/if}
					{:else if lane.type === 'lora_weight'}
						<p class={ui.help}>
							{lab.base.lora
								? `Sets the weight of ${lab.base.lora} while active (base weight ${lab.base.lora_weight} otherwise).`
								: 'Pick a LoRA in the base settings first.'}
						</p>
					{/if}
					{#if problems[i]}
						<p class="mt-1 text-xs text-red-600 dark:text-red-400">{problems[i]}</p>
					{/if}
				</div>
			{/each}
			<datalist id="lane_layer_suggestions">
				{#each known?.suggestions ?? ['*attn2', 'down_blocks.*attn2', 'mid_block.*attn2', 'up_blocks.*attn2', 'transformer_blocks.*'] as s (s)}
					<option value={s}></option>
				{/each}
			</datalist>
			{#if layerError}<p class="text-xs text-red-600">{layerError}</p>{/if}
		</div>

		<div class="mt-3 flex flex-wrap items-center gap-3">
			<button type="button" class={ui.secondaryButton} onclick={() => addLane()}>+ Add lane</button>
			<button type="button" class={ui.primaryButton} disabled={!!lab.busy} onclick={compare}
				>{skipBaseline ? 'Render with lanes' : 'Compare with baseline'}</button
			>
			<label class="flex items-center gap-1 text-sm">
				<input type="checkbox" bind:checked={skipBaseline} /> Skip baseline
			</label>
		</div>
		{#if error}<div class="{ui.errorBox} mt-3" role="alert">{error}</div>{/if}
	</div>

	{#if modulated}
		<div class={ui.card}>
			<div class="grid grid-cols-2 gap-3">
				{#if baseline && !skipBaseline}
					<figure>
						<img
							src={ui.png(baseline.image.image_base64)}
							alt="Without lanes"
							class="h-auto w-full rounded"
						/>
						<figcaption class={ui.help}>Without lanes · seed {baseline.image.seed}</figcaption>
					</figure>
				{/if}
				<figure>
					<img
						src={ui.png(modulated.image_base64)}
						alt="With lanes"
						class="h-auto w-full rounded"
					/>
					<figcaption class={ui.help}>
						With lanes · seed {modulated.seed}{applied.length ? ` · ${applied.join(', ')}` : ''}
					</figcaption>
				</figure>
			</div>
			{#if Object.keys(layerCounts).length}
				<p class="{ui.help} mt-2">
					Layers scaled: {Object.entries(layerCounts)
						.map(([k, v]) => `${k} → ${v}`)
						.join(', ')}
				</p>
			{/if}
			{#each notes as n, i (i)}
				<p class="{ui.noteBox} mt-2">{n}</p>
			{/each}
		</div>
	{/if}
</div>
