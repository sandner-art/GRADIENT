<!-- frontend/src/lib/components/lab/TaylorPreview.svelte -->
<!-- Taylor preview: one tangent computation, then any CFG nearby is previewed instantly
     as image(w + Δ) ≈ image(w) + Δ · ∂image/∂w. -->
<script lang="ts">
	import { post } from '#lib/apiClient.js';
	import { decodePixels } from '#lib/lab/canvas.js';
	import { baseProblem, baseRequest, lab, runLabJob } from '#lib/lab/labState.svelte.js';
	import { composeTaylor, meanAbsDiff } from '#lib/lab/math.js';
	import * as ui from '#lib/lab/ui.js';
	import type { TaylorRequest, TaylorResponse, Text2ImageResponse } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let method = $state<'auto' | 'jvp' | 'finite_difference'>('auto');
	let fdDelta = $state(0.5);
	let result = $state<TaylorResponse | null>(null);
	let error = $state<string | null>(null);
	let delta = $state(0);
	let view = $state<'preview' | 'magnitude'>('preview');

	let basePx = $state.raw<ImageData | null>(null);
	let tangentPx = $state.raw<ImageData | null>(null);
	let canvas = $state<HTMLCanvasElement | null>(null);

	let exact = $state<{
		delta: number;
		image: string;
		taylorError: number;
		plainError: number;
	} | null>(null);

	const w = $derived(result?.guidance_scale ?? lab.base.guidance_scale);
	const minDelta = $derived(-Math.min(6, Math.max(0, w - (result?.method === 'jvp' ? 1 : 0))));

	async function compute() {
		error = baseProblem();
		if (error) return;
		const body: TaylorRequest = {
			base_request: baseRequest(),
			method,
			delta: method === 'finite_difference' ? fdDelta : null
		};
		await runLabJob('Computing the guidance tangent…', async () => {
			try {
				const r = await post<TaylorResponse>('/lab/taylor-preview', body);
				[basePx, tangentPx] = await Promise.all([
					decodePixels(ui.png(r.base_image_base64)),
					decodePixels(ui.png(r.tangent_png_base64))
				]);
				result = r;
				delta = 0;
				exact = null;
				view = 'preview';
			} catch (e) {
				error = apiErrorMessage(e);
			}
		});
	}

	function previewPixels(d: number): Uint8ClampedArray<ArrayBuffer> | null {
		if (!basePx || !tangentPx || !result) return null;
		return composeTaylor(basePx.data, tangentPx.data, result.tangent_scale, d);
	}

	// Redraw the preview whenever Δ or the data changes (a few ms even at 1024²).
	$effect(() => {
		if (!canvas || !basePx || view !== 'preview') return;
		const px = previewPixels(delta);
		if (!px) return;
		canvas.width = basePx.width;
		canvas.height = basePx.height;
		canvas.getContext('2d')?.putImageData(new ImageData(px, basePx.width, basePx.height), 0, 0);
	});

	async function renderExact() {
		if (!result) return;
		const d = delta;
		const body = baseRequest({ guidance_scale: result.guidance_scale + d, seed: result.seed });
		await runLabJob(
			`Rendering CFG ${(result.guidance_scale + d).toFixed(2)} for comparison…`,
			async () => {
				try {
					const r = await post<Text2ImageResponse>('/generate/text-to-image', body);
					const image = r.images[0].image_base64;
					const exactPx = await decodePixels(ui.png(image));
					const predicted = previewPixels(d);
					if (!predicted || !basePx) return;
					exact = {
						delta: d,
						image,
						taylorError: meanAbsDiff(predicted, exactPx.data),
						plainError: meanAbsDiff(basePx.data, exactPx.data)
					};
				} catch (e) {
					error = apiErrorMessage(e);
				}
			}
		);
	}
</script>

<div class="space-y-4">
	<div class={ui.card}>
		<h2 class={ui.heading}>Taylor preview</h2>
		<p class="{ui.muted} mt-1">
			Computes how every pixel changes with the guidance scale, then previews nearby CFG values
			instantly with a slider. SD 1.x/SDXL use an exact forward-mode derivative through the whole
			sampler (about the cost of 2–3 renders); other families use finite differences (3 renders).
		</p>
		<div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
			<div>
				<label for="taylor_method" class={ui.label}>Method</label>
				<select id="taylor_method" class={ui.input} bind:value={method}>
					<option value="auto">Auto</option>
					<option value="jvp">Exact JVP (SD/SDXL)</option>
					<option value="finite_difference">Finite differences</option>
				</select>
			</div>
			{#if method === 'finite_difference'}
				<div>
					<label for="taylor_fd" class={ui.label}>Difference step ±</label>
					<input
						id="taylor_fd"
						type="number"
						min="0.05"
						step="0.05"
						class={ui.input}
						bind:value={fdDelta}
					/>
				</div>
			{/if}
			<div class="flex items-end">
				<button type="button" class={ui.primaryButton} disabled={!!lab.busy} onclick={compute}
					>Compute tangent</button
				>
			</div>
		</div>
		{#if error}<div class="{ui.errorBox} mt-3" role="alert">{error}</div>{/if}
	</div>

	{#if result}
		<div class={ui.card}>
			<div class="mb-3 flex flex-wrap items-center justify-between gap-2">
				<div class={ui.muted}>
					{result.method === 'jvp' ? 'Exact JVP' : 'Finite differences'} · seed {result.seed} ·
					{(result.performance_ms / 1000).toFixed(1)} s · mean |∂image/∂w| {(
						result.mean_abs_tangent * 255
					).toFixed(2)} levels per CFG unit
				</div>
				<div class="flex gap-1" role="group" aria-label="View">
					<button
						type="button"
						class={ui.secondaryButton}
						class:ring-2={view === 'preview'}
						onclick={() => (view = 'preview')}>Preview</button
					>
					<button
						type="button"
						class={ui.secondaryButton}
						class:ring-2={view === 'magnitude'}
						onclick={() => (view = 'magnitude')}>Sensitivity map</button
					>
				</div>
			</div>

			{#if view === 'preview'}
				<canvas
					bind:this={canvas}
					class="mx-auto block h-auto max-h-[70vh] w-full max-w-full rounded object-contain"
				></canvas>
				<div class="mt-4">
					<label for="taylor_delta" class={ui.label}>
						CFG {(result.guidance_scale + delta).toFixed(2)}
						<span class="font-normal text-gray-500">
							(computed at {result.guidance_scale}, Δ {delta >= 0 ? '+' : ''}{delta.toFixed(2)})
						</span>
					</label>
					<input
						id="taylor_delta"
						type="range"
						class="w-full"
						min={minDelta}
						max="6"
						step="0.05"
						bind:value={delta}
					/>
					<p class={ui.help}>
						A first-order prediction: accurate for small Δ, it drifts as composition starts to
						change. Check it against a real render.
					</p>
					<div class="mt-2 flex flex-wrap gap-2">
						<button
							type="button"
							class={ui.secondaryButton}
							disabled={!!lab.busy || delta === 0}
							onclick={renderExact}>Render exact at this CFG</button
						>
						<button
							type="button"
							class={ui.secondaryButton}
							disabled={delta === 0}
							onclick={() =>
								(lab.base.guidance_scale = +(result!.guidance_scale + delta).toFixed(2))}
							>Use this CFG in base settings</button
						>
						<button type="button" class={ui.secondaryButton} onclick={() => (delta = 0)}
							>Reset</button
						>
					</div>
				</div>
			{:else}
				<img
					src={ui.png(result.magnitude_heatmap_base64)}
					alt="Per-pixel sensitivity to the guidance scale"
					class="mx-auto block h-auto w-full rounded"
				/>
				<p class={ui.help}>Bright regions change most when the guidance scale changes.</p>
			{/if}
		</div>

		{#if exact}
			<div class={ui.card}>
				<h3 class="mb-2 font-medium">
					Prediction vs real render at CFG {(result.guidance_scale + exact.delta).toFixed(2)}
				</h3>
				<div class="grid grid-cols-2 gap-3">
					<figure>
						<img src={ui.png(exact.image)} alt="Real render" class="h-auto w-full rounded" />
						<figcaption class={ui.help}>Real render</figcaption>
					</figure>
					<div class="text-sm">
						<p>
							Mean error of the Taylor preview:
							<strong>{exact.taylorError.toFixed(2)}</strong> levels
						</p>
						<p>
							Without the tangent (just the base image):
							<strong>{exact.plainError.toFixed(2)}</strong> levels
						</p>
						<p class="{ui.help} mt-2">
							{exact.taylorError < exact.plainError
								? `The tangent removes ${Math.round((1 - exact.taylorError / exact.plainError) * 100)}% of the change.`
								: 'At this Δ the linear prediction no longer helps; the image changes non-linearly.'}
						</p>
					</div>
				</div>
			</div>
		{/if}
	{/if}
</div>
