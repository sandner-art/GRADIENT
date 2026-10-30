<!-- frontend/src/lib/components/lab/TokenAblation.svelte -->
<!-- Leave-one-word-out: re-render without each word (same seed) and measure how far the
     image moves in DINO feature space. A causal, model-agnostic influence measure. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { get, post } from '#lib/apiClient.js';
	import { baseProblem, baseRequest, lab, runLabJob } from '#lib/lab/labState.svelte.js';
	import { parseWordList } from '#lib/lab/math.js';
	import * as ui from '#lib/lab/ui.js';
	import type {
		AblationRequest,
		AblationResponse,
		AblationResult,
		LocalModel
	} from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let mode = $state<'remove_word' | 'mask_embedding'>('remove_word');
	let wordsText = $state('');
	let maxWords = $state(8);
	let featureModel = $state<string | null>(null);
	let visionModels = $state<string[]>([]);
	let show = $state<'overlay' | 'heatmap' | 'image'>('overlay');
	let result = $state<AblationResponse | null>(null);
	let error = $state<string | null>(null);

	onMount(async () => {
		try {
			visionModels = (await get<LocalModel[]>('/models/local?type=vision')).map((m) => m.name);
		} catch {
			visionModels = [];
		}
	});

	const words = $derived(parseWordList(wordsText));
	const renders = $derived((words?.length ?? maxWords) + 1);
	const maxDistance = $derived(
		Math.max(1e-9, ...(result?.results.map((r) => r.patch_distance) ?? [0]))
	);

	async function run() {
		error = baseProblem();
		if (error) return;
		const body: AblationRequest = {
			base_request: baseRequest(),
			mode,
			words,
			max_words: maxWords,
			feature_model: featureModel
		};
		await runLabJob(`Rendering ${renders} images and comparing features…`, async () => {
			try {
				result = await post<AblationResponse>('/lab/token-ablation', body);
			} catch (e) {
				error = apiErrorMessage(e);
			}
		});
	}

	const fmt = (v: number) => (Math.abs(v) >= 0.01 || v === 0 ? v.toFixed(3) : v.toExponential(1));

	const picture = (r: AblationResult) =>
		show === 'overlay'
			? r.overlay_base64
			: show === 'heatmap'
				? r.heatmap_base64
				: r.ablated_image_base64;
</script>

<div class="space-y-4">
	<div class={ui.card}>
		<h2 class={ui.heading}>Token ablation</h2>
		<p class="{ui.muted} mt-1">
			Removes one word at a time, renders again with the same seed and measures how much the image
			changes in DINOv3 feature space (DINOv2 if v3 isn't installed). Large distance = the word
			matters; the map shows where. Works for every model family.
		</p>
		<div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
			<div>
				<label for="abl_mode" class={ui.label}>Ablation</label>
				<select id="abl_mode" class={ui.input} bind:value={mode}>
					<option value="remove_word">Remove word (any model)</option>
					<option value="mask_embedding">Mask embedding (SD/SDXL)</option>
				</select>
				<p class={ui.help}>
					{mode === 'remove_word'
						? 'Re-tokenizes the shorter prompt, so later words shift position.'
						: 'Keeps every other token in place; replaces only this word with the empty prompt’s embedding.'}
				</p>
			</div>
			<div>
				<label for="abl_feature" class={ui.label}>Feature model</label>
				<select id="abl_feature" class={ui.input} bind:value={featureModel}>
					<option value={null}>Default (DINOv3 ViT-S/16 → DINOv2)</option>
					{#each visionModels as v (v)}
						<option value={v}>{v}</option>
					{/each}
				</select>
				<p class={ui.help}>
					Install DINOv3 (gated, accept the license on Hugging Face) or DINOv2 from the Models tab
					under “Vision”.
				</p>
			</div>
			<div>
				<label for="abl_words" class={ui.label}>Words (empty = content words of the prompt)</label>
				<input
					id="abl_words"
					class={ui.input}
					placeholder="lighthouse, sunset"
					bind:value={wordsText}
				/>
			</div>
			<div>
				<label for="abl_max" class={ui.label}>Max words</label>
				<input
					id="abl_max"
					type="number"
					min="1"
					max="40"
					class={ui.input}
					bind:value={maxWords}
					disabled={!!words}
				/>
			</div>
		</div>
		<div class="mt-3 flex items-center gap-3">
			<button type="button" class={ui.primaryButton} disabled={!!lab.busy} onclick={run}
				>Run ablation</button
			>
			<span class={ui.muted}>Up to {renders} renders.</span>
		</div>
		{#if error}<div class="{ui.errorBox} mt-3" role="alert">{error}</div>{/if}
	</div>

	{#if result}
		<div class={ui.card}>
			<div class="mb-3 flex flex-wrap items-center justify-between gap-2">
				<p class={ui.muted}>
					{result.feature_model} · seed {result.seed} · {(result.performance_ms / 1000).toFixed(1)} s
				</p>
				<div class="flex gap-1" role="group" aria-label="Show">
					{#each [['overlay', 'Where'], ['heatmap', 'Heatmap'], ['image', 'Without word']] as [id, text] (id)}
						<button
							type="button"
							class={ui.secondaryButton}
							class:ring-2={show === id}
							onclick={() => (show = id as typeof show)}>{text}</button
						>
					{/each}
				</div>
			</div>
			{#if result.feature_note}<div class="{ui.noteBox} mb-3">{result.feature_note}</div>{/if}

			<div class="grid grid-cols-1 gap-4 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
				<figure>
					<img
						src={ui.png(result.base_image_base64)}
						alt="Full prompt"
						class="h-auto w-full rounded"
					/>
					<figcaption class={ui.help}>Full prompt</figcaption>
				</figure>
				<ol class="space-y-2">
					{#each result.results as r, i (r.word)}
						<li class="flex gap-3 rounded border border-gray-200 p-2 dark:border-gray-700">
							<img
								src={ui.img(picture(r))}
								alt="Effect of {r.word}"
								class="h-20 w-20 flex-none rounded object-cover"
							/>
							<div class="min-w-0 flex-1">
								<div class="flex items-baseline justify-between gap-2">
									<span class="font-medium">{i + 1}. {r.word}</span>
									<span class="text-xs text-gray-500 tabular-nums dark:text-gray-400">
										patch {fmt(r.patch_distance)} · global {fmt(r.global_distance)}
									</span>
								</div>
								<div class="mt-1 h-1.5 rounded bg-gray-200 dark:bg-gray-700">
									<div
										class="h-1.5 rounded bg-indigo-500"
										style:width="{(r.patch_distance / maxDistance) * 100}%"
									></div>
								</div>
								{#if r.ablated_prompt}
									<p
										class="mt-1 truncate text-xs text-gray-500 dark:text-gray-400"
										title={r.ablated_prompt}
									>
										“{r.ablated_prompt}”
									</p>
								{/if}
							</div>
						</li>
					{/each}
				</ol>
			</div>
			<p class={ui.help}>
				Ranked by mean per-patch feature distance (1 − cosine). Global = distance of the image-level
				(CLS) features. Each map is scaled on its own.
			</p>
		</div>
	{/if}
</div>
