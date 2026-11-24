<!-- frontend/src/lib/components/lab/QuantReport.svelte -->
<!-- Quantization check: render the Lab's settings at full precision and quantized, on the same
     seeds, and compare cost, images and the Taylor tangent (do Lab analyses still hold?). -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { get, post } from '#lib/apiClient.js';
	import { quantOptions } from '#lib/components/shared/QuantizationSelect.svelte';
	import { baseProblem, baseRequest, runLabJob } from '#lib/lab/labState.svelte.js';
	import * as ui from '#lib/lab/ui.js';
	import type { LocalModel, QuantOption, QuantReportResponse } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let options = $state<QuantOption[]>([]);
	let picked = $state<string[]>([]);
	let reference = $state('none');
	let seeds = $state(3);
	let taylor = $state(true);
	let featureModel = $state<string | null>(null);
	let visionModels = $state<string[]>([]);
	let result = $state<QuantReportResponse | null>(null);
	let error = $state<string | null>(null);
	let open = $state<string | null>(null);

	onMount(async () => {
		options = await quantOptions(true);
		const first = options.find((o) => o.available && o.value !== 'none');
		if (first) picked = [first.value];
		try {
			visionModels = (await get<LocalModel[]>('/models/local?type=vision')).map((m) => m.name);
		} catch {
			visionModels = [];
		}
	});

	const variants = $derived(picked.filter((v) => v !== reference));
	const renders = $derived((variants.length + 1) * (seeds + (taylor ? 3 : 0)));

	function toggle(value: string, on: boolean) {
		picked = on ? [...picked, value].slice(-4) : picked.filter((v) => v !== value);
	}

	async function run() {
		error = baseProblem();
		if (!error && !variants.length) error = 'Pick at least one quantization to compare.';
		if (error) return;
		const body = {
			base_request: baseRequest(),
			variants,
			reference,
			num_seeds: seeds,
			taylor,
			feature_model: featureModel
		};
		await runLabJob(
			`Rendering ${renders} images with ${variants.length + 1} precisions…`,
			async () => {
				try {
					result = await post<QuantReportResponse>('/lab/quantization-report', body);
					open = null;
				} catch (e) {
					error = apiErrorMessage(e);
				}
			}
		);
	}

	const num = (v: number | null | undefined, digits = 2) =>
		v === null || v === undefined ? '—' : v.toFixed(digits);
	/** megabytes: decimals only for small values */
	const mb = (v: number | null | undefined) =>
		v === null || v === undefined ? '—' : `${v.toFixed(v < 10 ? 2 : 0)} MB`;
	const tone = (rel: number | null) =>
		rel === null
			? ''
			: rel < 0.1
				? 'text-emerald-700 dark:text-emerald-400'
				: rel < 0.35
					? 'text-amber-700 dark:text-amber-400'
					: 'text-red-700 dark:text-red-400';
</script>

<div class="space-y-4">
	<div class={ui.card}>
		<h2 class={ui.heading}>Quantization check</h2>
		<p class="{ui.muted} mt-1">
			Renders these settings at full precision and quantized, with the same seeds, then compares
			memory, speed, the images (PSNR, SSIM, DINO features) and the Taylor tangent ∂image/∂guidance,
			which shows whether Lab measurements made on the quantized model still hold. "Relative change"
			puts the feature distance on a scale where 1 means as different as another seed.
		</p>

		<fieldset class="mt-3">
			<legend class={ui.label}>Compare</legend>
			<div class="grid grid-cols-1 gap-1 sm:grid-cols-2">
				{#each options.filter((o) => o.value !== 'none') as o (o.value)}
					<label
						class="flex items-start gap-2 text-sm"
						class:opacity-50={!o.available}
						title={o.reason ?? o.about}
					>
						<input
							type="checkbox"
							class="mt-1"
							disabled={!o.available}
							checked={picked.includes(o.value)}
							onchange={(e) => toggle(o.value, e.currentTarget.checked)}
						/>
						<span
							>{o.label}{#if !o.available}<span class="text-xs text-gray-500">
									· {o.reason?.includes('GPU') ? 'needs an NVIDIA GPU' : 'not installed'}</span
								>{/if}</span
						>
					</label>
				{/each}
			</div>
			{#if options.length && options.every((o) => !o.available || o.value === 'none')}
				<p class={ui.help}>
					No quantization backend is installed. Install GRADIENT's quant extras: <code
						>pip install -e .[quant]</code
					>.
				</p>
			{/if}
		</fieldset>

		<div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
			<div>
				<label for="qr_reference" class={ui.label}>Reference</label>
				<select id="qr_reference" class={ui.input} bind:value={reference}>
					<option value="none">Full precision</option>
					{#each options.filter((o) => o.available && o.value !== 'none') as o (o.value)}
						<option value={o.value}>{o.label}</option>
					{/each}
				</select>
			</div>
			<div>
				<label for="qr_seeds" class={ui.label}>Seeds</label>
				<input id="qr_seeds" type="number" min="2" max="8" class={ui.input} bind:value={seeds} />
			</div>
			<div>
				<label for="qr_fm" class={ui.label}>Feature model</label>
				<select id="qr_fm" class={ui.input} bind:value={featureModel}>
					<option value={null}>Automatic (DINOv3, else DINOv2)</option>
					{#each visionModels as m (m)}<option value={m}>{m}</option>{/each}
				</select>
			</div>
		</div>
		<label class="mt-2 flex items-center gap-2 text-sm">
			<input type="checkbox" bind:checked={taylor} /> Compare the Taylor tangent (3 more renders each,
			or one exact JVP on SD/SDXL)
		</label>

		{#if error}<div class="{ui.errorBox} mt-3" role="alert">{error}</div>{/if}
		<div class="mt-3 flex items-center justify-between gap-2">
			<p class={ui.help}>
				About {renders} renders; each precision loads the model once. Seeds start at the Lab seed.
			</p>
			<button type="button" class={ui.primaryButton} onclick={run}>Run the check</button>
		</div>
	</div>

	{#if result}
		<div class="{ui.card} space-y-3">
			<p class="text-sm">{result.summary}</p>
			{#if result.feature_note}<p class={ui.help}>{result.feature_note}</p>{/if}
			<div class="overflow-x-auto">
				<table class="w-full min-w-[44rem] text-left text-sm">
					<thead class="text-xs text-gray-500">
						<tr>
							<th class="py-1 pr-2 font-medium">Precision</th>
							<th class="pr-2 font-medium">Denoiser</th>
							<th class="pr-2 font-medium">s / image</th>
							<th class="pr-2 font-medium">Peak GPU</th>
							<th class="pr-2 font-medium">PSNR</th>
							<th class="pr-2 font-medium">SSIM</th>
							<th class="pr-2 font-medium" title="Feature distance / distance between two seeds"
								>Relative change</th
							>
							<th class="pr-2 font-medium" title="Cosine of ∂image/∂guidance with the reference"
								>Tangent</th
							>
						</tr>
					</thead>
					<tbody>
						{#each result.variants as v, i (v.quantization)}
							<tr class="border-t border-gray-200 align-top dark:border-gray-700">
								<td class="py-1.5 pr-2">
									<div class="font-medium">{v.label}</div>
									<div class="text-xs text-gray-500">{v.error ?? v.verdict}</div>
								</td>
								<td class="pr-2 tabular-nums">{mb(v.denoiser_mb)}</td>
								<td class="pr-2 tabular-nums">{num(v.seconds_per_image)}</td>
								<td class="pr-2 tabular-nums">{mb(v.peak_memory_mb)}</td>
								<td class="pr-2 tabular-nums">{i === 0 ? '' : num(v.psnr, 1)}</td>
								<td class="pr-2 tabular-nums">{i === 0 ? '' : num(v.ssim, 3)}</td>
								<td class="pr-2 tabular-nums {tone(v.relative_change)}"
									>{i === 0 ? '' : num(v.relative_change, 3)}</td
								>
								<td class="pr-2 tabular-nums">{i === 0 ? '' : num(v.taylor_cosine, 3)}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<img
				src="data:image/png;base64,{result.image_base64}"
				alt="One row per precision, one column per seed"
				class="max-w-full rounded border border-gray-200 dark:border-gray-700"
			/>
			{#each result.variants.filter((v) => v.strip_base64) as v (v.quantization)}
				<div>
					<button
						type="button"
						class="text-sm text-indigo-600 hover:underline dark:text-indigo-400"
						onclick={() => (open = open === v.quantization ? null : v.quantization)}
						>{open === v.quantization ? '▾' : '▸'}
						{v.label}: reference | quantized | difference</button
					>
					{#if open === v.quantization}
						<img
							src="data:image/png;base64,{v.strip_base64}"
							alt="Per seed: reference, quantized and the difference"
							class="mt-2 max-w-md rounded border border-gray-200 dark:border-gray-700"
						/>
					{/if}
				</div>
			{/each}
			<p class={ui.help}>
				Seeds {result.seeds.join(', ')} · {result.feature_model ?? 'no feature model'} · saved to the
				gallery as a Lab run · {(result.performance_ms / 1000).toFixed(1)} s
			</p>
		</div>
	{/if}
</div>
