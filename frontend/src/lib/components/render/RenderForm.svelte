<!-- frontend/src/lib/components/render/RenderForm.svelte -->
<!-- Everyday rendering (text-to-image and image-to-image). Queues jobs instead of waiting,
     so you can keep changing settings and queue more while the GPU works. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { get } from '#lib/apiClient.js';
	import QuantizationSelect from '#lib/components/shared/QuantizationSelect.svelte';
	import RecommendedSettings from '#lib/components/shared/RecommendedSettings.svelte';
	import StylePicker from './StylePicker.svelte';
	import TokenView from './TokenView.svelte';
	import { TILINGS, type Tiling } from '#lib/tiling.js';
	import type { StylePreview } from '#lib/styles.js';
	import { addToken, removeToken, usedEmbeddings, type EmbeddingDetail } from '#lib/embeddings.js';
	import * as ui from '#lib/lab/ui.js';
	import { rememberFormJobs, submitJob } from '#lib/stores/queue.svelte.js';
	import { notify, sendToLab } from '#lib/stores/ui.svelte.js';
	import type {
		ImageToImageRequest,
		LoRAParameter,
		PresetRead,
		PromptSyntax,
		Text2ImageRequest
	} from '#lib/types.js';
	import { apiErrorMessage, plainSchedulerName } from '#lib/utils/format.js';
	import {
		applyPresetToStore,
		fetchPresetsForModel,
		savePresetFromStore
	} from '#lib/utils/presetManager.js';

	type Req = Text2ImageRequest & Partial<Pick<ImageToImageRequest, 'image_base64' | 'strength'>>;
	interface Props {
		mode: 'text' | 'image';
		store: Writable<Req>;
	}
	let { mode, store }: Props = $props();

	let models = $state<string[]>([]);
	let schedulers = $state<string[]>([]);
	let loraNames = $state<string[]>([]);
	let presets = $state<PresetRead[]>([]);
	let presetId = $state('none');
	let batchCount = $state(1);
	let perLine = $state(false);
	let showAdvanced = $state(false);
	let embeddingDetails = $state<EmbeddingDetail[]>([]);
	let pickedEmbedding = $state('');
	let stylePreview = $state<StylePreview | null>(null);
	let sigmasText = $state('');
	let error = $state<string | null>(null);
	let busy = $state(false);
	let dragOver = $state(false);

	const prompts = $derived(
		($store.prompt ?? '')
			.split('\n')
			.map((l) => l.trim())
			.filter(Boolean)
	);
	const jobsToQueue = $derived((perLine && prompts.length > 1 ? prompts.length : 1) * batchCount);
	const imagesToQueue = $derived(jobsToQueue * (Number($store.num_outputs) || 1));
	const SIZES: [number, number][] = [
		[512, 512],
		[768, 768],
		[1024, 1024],
		[832, 1216],
		[1216, 832],
		[896, 1152],
		[1152, 896]
	];

	onMount(async () => {
		const [m, s, l] = await Promise.all([
			get<string[]>('/system/models/diffusion').catch((): string[] => []),
			get<string[]>('/system/schedulers').catch(() => ['Default']),
			get<string[]>('/system/models/loras').catch((): string[] => [])
		]);
		models = m;
		schedulers = s.filter((x) => x !== 'Default');
		loraNames = l;
		if (models.length && !models.includes($store.model_id))
			store.update((v) => ({ ...v, model_id: models[0] }));
	});

	// custom sigmas: keep the text box and the store in step (e.g. after "use settings")
	$effect(() => {
		const stored = $store.custom_sigmas ?? null;
		if (JSON.stringify(stored) !== JSON.stringify(parseSigmas(sigmasText))) {
			sigmasText = stored?.join(', ') ?? '';
		}
		if (stored) showAdvanced = true;
	});

	// embeddings: the list says which ones fit the selected model
	let embeddingsFor: string | null = null;
	$effect(() => {
		const id = $store.model_id;
		if (id === embeddingsFor) return;
		embeddingsFor = id;
		get<{ embeddings: EmbeddingDetail[] }>(
			`/system/models/embeddings/details?model_id=${encodeURIComponent(id ?? '')}`
		)
			.then((r) => {
				if (embeddingsFor === id) embeddingDetails = r.embeddings;
			})
			.catch(() => (embeddingDetails = []));
	});

	// presets follow the model
	let presetsFor = '';
	$effect(() => {
		const id = $store.model_id;
		if (id === presetsFor) return;
		presetsFor = id;
		presetId = 'none';
		fetchPresetsForModel(id).then((p) => {
			if (presetsFor === id) presets = p;
		});
	});

	function set<K extends keyof Req>(key: K, value: Req[K]) {
		store.update((v) => ({ ...v, [key]: value }));
	}

	function applyPreset(value: string) {
		presetId = value;
		const preset = presets.find((p) => String(p.id) === value) ?? null;
		applyPresetToStore(preset, store, () => (sigmasText = $store.custom_sigmas?.join(', ') ?? ''));
	}

	async function savePreset() {
		await savePresetFromStore(store, (s) => ({
			negative_prompt: s.negative_prompt,
			height: s.height,
			width: s.width,
			num_inference_steps: s.num_inference_steps,
			guidance_scale: s.guidance_scale,
			scheduler_id: s.scheduler_id
		}));
		presets = await fetchPresetsForModel($store.model_id);
	}

	// ---- LoRAs and embeddings live in the store, so "use settings" can restore them
	const loras = $derived($store.loras ?? []);
	function setLoras(next: LoRAParameter[]) {
		set('loras', next);
	}
	// embeddings are used by writing their token in a prompt, as in A1111
	const embeddingsInUse = $derived(
		usedEmbeddings($store.prompt, $store.negative_prompt, embeddingDetails)
	);
	function insertEmbedding(where: 'prompt' | 'negative_prompt') {
		const picked = embeddingDetails.find((e) => e.name === pickedEmbedding);
		if (!picked) return;
		set(where, addToken($store[where], picked.token));
		pickedEmbedding = '';
	}

	// ---- init image (image mode)
	function readFile(file: File | undefined | null) {
		if (!file || !file.type.startsWith('image/')) return;
		const reader = new FileReader();
		reader.onload = () => set('image_base64', String(reader.result));
		reader.readAsDataURL(file);
	}
	function onPaste(e: ClipboardEvent) {
		if (mode !== 'image') return;
		const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith('image/'));
		if (item) readFile(item.getAsFile());
	}

	function parseSigmas(text: string): number[] | null {
		const values = text
			.split(/[\s,]+/)
			.filter(Boolean)
			.map(Number);
		return values.length && values.every((v) => Number.isFinite(v)) ? values : null;
	}

	function problem(): string | null {
		const s = $store;
		if (!s.model_id) return 'Pick a model.';
		if (!s.prompt?.trim()) return 'The prompt is empty.';
		const w = Number(s.width ?? 0);
		const h = Number(s.height ?? 0);
		if (w % 8 || h % 8 || w <= 0 || h <= 0)
			return 'Width and height must be positive multiples of 8.';
		if (!(Number(s.num_inference_steps ?? 0) > 0)) return 'Steps must be positive.';
		if (mode === 'image' && !s.image_base64)
			return 'Add an input image (drop, paste or choose a file).';
		if (loras.some((l) => !l.name))
			return 'Choose a LoRA for every LoRA row, or remove the empty row.';
		return null;
	}

	async function queueRender() {
		error = problem();
		if (error) return;
		busy = true;
		const s = $store;
		const base: Record<string, unknown> = {
			model_id: s.model_id,
			negative_prompt: s.negative_prompt?.trim() || null,
			width: Number(s.width),
			height: Number(s.height),
			num_inference_steps: Number(s.num_inference_steps),
			guidance_scale: s.guidance_scale,
			seed:
				s.seed === null || s.seed === undefined || Number.isNaN(Number(s.seed))
					? null
					: Math.floor(Number(s.seed)),
			scheduler_id: s.scheduler_id || null,
			num_outputs: Number(s.num_outputs) || 1,
			custom_sigmas: parseSigmas(sigmasText),
			loras: loras.length ? loras : null,
			embeddings: s.embeddings?.length ? s.embeddings : null,
			vae_id: s.vae_id || null,
			quantization: s.quantization || null,
			prompt_syntax: s.prompt_syntax || undefined,
			styles: s.styles?.length ? s.styles : null,
			style_position: s.styles?.length ? (s.style_position ?? 'template') : undefined,
			style_start: s.styles?.length ? (s.style_start ?? 0) : undefined,
			tiling: s.tiling && s.tiling !== 'none' ? s.tiling : undefined
		};
		if (mode === 'image') {
			base.image_base64 = s.image_base64;
			base.strength = s.strength ?? 0.75;
		}
		const kind = mode === 'image' ? 'image_to_image' : 'text_to_image';
		const lines = perLine && prompts.length > 1 ? prompts : [s.prompt.trim()];
		try {
			for (const line of lines) {
				const jobs = await submitJob(kind, { ...base, prompt: line }, { count: batchCount });
				rememberFormJobs(mode, jobs);
			}
			notify(
				`Queued ${jobsToQueue} job${jobsToQueue > 1 ? 's' : ''} (${imagesToQueue} image${imagesToQueue > 1 ? 's' : ''})`,
				'success',
				undefined,
				2500
			);
		} catch (e) {
			error = apiErrorMessage(e);
		} finally {
			busy = false;
		}
	}

	/** The form's settings into the Lab (first prompt line; the Lab renders one image). */
	function toLab() {
		const s = $store;
		// the Lab has no styles: it gets the prompts as they would be rendered
		const styled = s.styles?.length && stylePreview;
		sendToLab({
			...s,
			prompt: styled ? stylePreview!.prompt : (prompts[0] ?? s.prompt ?? '').trim(),
			negative_prompt: styled ? (stylePreview!.negative_prompt ?? '') : s.negative_prompt,
			loras: loras.filter((l) => l.name),
			custom_sigmas: parseSigmas(sigmasText)
		});
	}

	function onKey(e: KeyboardEvent) {
		if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
			e.preventDefault();
			queueRender();
		}
	}
</script>

<svelte:window onpaste={onPaste} />

<div class="space-y-4">
	<form
		class={ui.card}
		onsubmit={(e) => {
			e.preventDefault();
			queueRender();
		}}
	>
		<div class="mb-4 flex items-center justify-between gap-2">
			<h2 class={ui.heading}>{mode === 'image' ? 'Image-to-image' : 'Text-to-image'}</h2>
			<span class={ui.muted}>Ctrl+Enter queues</span>
		</div>

		<div class="space-y-4">
			<!-- model + presets -->
			<div class="grid grid-cols-1 gap-3 sm:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
				<div>
					<label for="{mode}_model" class={ui.label}>Model</label>
					<select
						id="{mode}_model"
						class={ui.input}
						value={$store.model_id}
						onchange={(e) => set('model_id', e.currentTarget.value)}
					>
						{#if !models.length}<option value="">No models installed</option>{/if}
						{#each models as m (m)}<option value={m}>{m}</option>{/each}
					</select>
				</div>
				<div>
					<label for="{mode}_preset" class={ui.label}>Preset</label>
					<div class="flex gap-1">
						<select
							id="{mode}_preset"
							class={ui.input}
							value={presetId}
							onchange={(e) => applyPreset(e.currentTarget.value)}
						>
							<option value="none">—</option>
							{#each presets as p (p.id)}<option value={String(p.id)}>{p.preset_name}</option
								>{/each}
						</select>
						<button
							type="button"
							class={ui.secondaryButton}
							title="Save these settings as a preset"
							onclick={savePreset}>💾</button
						>
					</div>
				</div>
			</div>
			<QuantizationSelect
				id="{mode}_quantization"
				modelId={$store.model_id || null}
				value={$store.quantization}
				onchange={(v) => store.update((s) => ({ ...s, quantization: v }))}
			/>
			<RecommendedSettings
				modelId={$store.model_id}
				onapply={(d) =>
					store.update((v) => ({
						...v,
						width: d.width ?? v.width,
						height: d.height ?? v.height,
						num_inference_steps: d.steps ?? v.num_inference_steps,
						guidance_scale: d.guidance_scale ?? v.guidance_scale,
						scheduler_id: plainSchedulerName(d.scheduler)
					}))}
			/>

			<!-- input image -->
			{#if mode === 'image'}
				<div>
					<span class={ui.label}>Input image</span>
					<label
						class="flex cursor-pointer items-center gap-3 rounded-md border-2 border-dashed p-3 text-sm transition"
						class:border-indigo-400={dragOver}
						class:bg-indigo-50={dragOver}
						class:border-gray-300={!dragOver}
						ondragover={(e) => {
							e.preventDefault();
							dragOver = true;
						}}
						ondragleave={() => (dragOver = false)}
						ondrop={(e) => {
							e.preventDefault();
							dragOver = false;
							readFile(e.dataTransfer?.files?.[0]);
						}}
					>
						{#if $store.image_base64}
							<img src={$store.image_base64} alt="Input" class="h-20 w-20 rounded object-cover" />
						{/if}
						<span class="text-gray-600 dark:text-gray-300">
							Drop an image here, paste one, or <span class="text-indigo-600 underline"
								>choose a file</span
							>. Gallery images: “Use as input” in the viewer.
						</span>
						<input
							type="file"
							accept="image/*"
							class="sr-only"
							onchange={(e) => readFile(e.currentTarget.files?.[0])}
						/>
					</label>
					<label for="{mode}_strength" class="{ui.label} mt-2"
						>Strength {($store.strength ?? 0.75).toFixed(2)}
						<span class="font-normal text-gray-500">(how much the image may change)</span></label
					>
					<input
						id="{mode}_strength"
						type="range"
						min="0"
						max="1"
						step="0.01"
						class="w-full"
						value={$store.strength ?? 0.75}
						oninput={(e) => set('strength', Number(e.currentTarget.value))}
					/>
				</div>
			{/if}

			<!-- prompts -->
			<div>
				<label for="{mode}_prompt" class={ui.label}>Prompt</label>
				<textarea
					id="{mode}_prompt"
					rows="4"
					class={ui.input}
					value={$store.prompt}
					oninput={(e) => set('prompt', e.currentTarget.value)}
					onkeydown={onKey}></textarea>
				{#if prompts.length > 1}
					<label class="mt-1 flex items-center gap-2 text-sm">
						<input type="checkbox" bind:checked={perLine} />
						One job per line ({prompts.length} prompts)
					</label>
				{/if}
			</div>
			<div>
				<label for="{mode}_negative" class={ui.label}>Negative prompt</label>
				<textarea
					id="{mode}_negative"
					rows="2"
					class={ui.input}
					value={$store.negative_prompt ?? ''}
					oninput={(e) => set('negative_prompt', e.currentTarget.value)}
					onkeydown={onKey}></textarea>
			</div>

			<StylePicker {store} bind:preview={stylePreview} />

			<TokenView
				modelId={$store.model_id}
				prompt={stylePreview ? stylePreview.prompt : $store.prompt}
				negative={stylePreview ? stylePreview.negative_prompt : $store.negative_prompt}
				syntax={$store.prompt_syntax}
				styled={!!stylePreview && !!$store.styles?.length}
			/>

			<!-- LoRAs -->
			<div>
				<div class="mb-1 flex items-center justify-between">
					<span class={ui.label}>LoRAs</span>
					<button
						type="button"
						class="text-sm text-indigo-600 hover:underline disabled:opacity-50 dark:text-indigo-400"
						disabled={!loraNames.length}
						onclick={() => setLoras([...loras, { name: '', weight: 0.8 }])}>+ Add LoRA</button
					>
				</div>
				{#if !loraNames.length}
					<p class={ui.help}>No LoRAs installed (Models tab → LoRA).</p>
				{/if}
				{#each loras as lora, i (i)}
					<div class="mb-2 flex items-center gap-2">
						<select
							class={ui.smallInput}
							aria-label="LoRA {i + 1}"
							value={lora.name}
							onchange={(e) =>
								setLoras(
									loras.map((l, j) => (j === i ? { ...l, name: e.currentTarget.value } : l))
								)}
						>
							<option value="">— choose —</option>
							{#each loraNames as n (n)}<option value={n}>{n}</option>{/each}
						</select>
						<input
							type="range"
							min="-1"
							max="2"
							step="0.05"
							class="w-28 flex-none"
							aria-label="Weight of LoRA {i + 1}"
							value={lora.weight ?? 1}
							oninput={(e) =>
								setLoras(
									loras.map((l, j) =>
										j === i ? { ...l, weight: Number(e.currentTarget.value) } : l
									)
								)}
						/>
						<span class="w-10 text-right font-mono text-xs"
							>{Number(lora.weight ?? 1).toFixed(2)}</span
						>
						<button
							type="button"
							class="text-red-600 hover:underline"
							aria-label="Remove LoRA {i + 1}"
							onclick={() => setLoras(loras.filter((_, j) => j !== i))}>✕</button
						>
					</div>
				{/each}
				{#if loras.length}
					<p class={ui.help}>LoRAs stay loaded: changing weights or switching back is instant.</p>
				{/if}
			</div>

			<!-- textual inversion embeddings -->
			<div>
				<div class="mb-1 flex flex-wrap items-center justify-between gap-2">
					<span class={ui.label}>Embeddings</span>
					{#if embeddingDetails.length}
						<div class="flex items-center gap-2">
							<select
								class={ui.sized(ui.smallInput, 'w-48')}
								aria-label="Embedding to add"
								bind:value={pickedEmbedding}
							>
								<option value="">— choose —</option>
								{#each embeddingDetails as e (e.name)}
									<option value={e.name}
										>{e.token}{e.family_label ? ` · ${e.family_label}` : ''}{e.fits === false
											? ' (other model family)'
											: ''}</option
									>
								{/each}
							</select>
							<button
								type="button"
								class="text-sm text-indigo-600 hover:underline disabled:opacity-50 dark:text-indigo-400"
								disabled={!pickedEmbedding}
								onclick={() => insertEmbedding('prompt')}>+ Prompt</button
							>
							<button
								type="button"
								class="text-sm text-indigo-600 hover:underline disabled:opacity-50 dark:text-indigo-400"
								disabled={!pickedEmbedding}
								onclick={() => insertEmbedding('negative_prompt')}>+ Negative</button
							>
						</div>
					{/if}
				</div>
				{#if !embeddingDetails.length}
					<p class={ui.help}>No embeddings installed (Models tab → Embedding).</p>
				{:else if embeddingsInUse.length}
					<div class="flex flex-wrap gap-1">
						{#each embeddingsInUse as used (used.where + used.embedding.name)}
							<span
								class="rounded px-2 py-0.5 text-xs {used.embedding.fits === false
									? 'bg-amber-100 text-amber-900 dark:bg-amber-900 dark:text-amber-100'
									: 'bg-gray-100 dark:bg-gray-700'}"
								title={used.embedding.fits === false
									? `Made for ${used.embedding.family_label ?? 'another model family'}: it will be skipped with this model`
									: used.embedding.name}
								>{used.embedding.token}
								<span class="opacity-60"
									>{used.where === 'prompt' ? 'prompt' : 'negative'}{used.embedding.fits === false
										? ' · other model family'
										: ''}</span
								>
								<button
									type="button"
									class="ml-1 text-red-600"
									aria-label="Remove {used.embedding.token} from the {used.where === 'prompt'
										? 'prompt'
										: 'negative prompt'}"
									onclick={() =>
										set(used.where, removeToken($store[used.where], used.embedding.token))}
									>✕</button
								></span
							>
						{/each}
					</div>
				{:else}
					<p class={ui.help}>
						Add one to the prompt or the negative prompt, or type its name there as in A1111.
					</p>
				{/if}
			</div>

			<!-- size -->
			<div>
				<span class={ui.label}>Size</span>
				<div class="flex flex-wrap items-center gap-2">
					<input
						type="number"
						step="8"
						min="64"
						class={ui.sized(ui.input, 'w-24')}
						aria-label="Width"
						value={$store.width}
						oninput={(e) => set('width', Number(e.currentTarget.value))}
					/>
					<button
						type="button"
						class={ui.secondaryButton}
						title="Swap width and height"
						onclick={() => store.update((v) => ({ ...v, width: v.height, height: v.width }))}
						>⇄</button
					>
					<input
						type="number"
						step="8"
						min="64"
						class={ui.sized(ui.input, 'w-24')}
						aria-label="Height"
						value={$store.height}
						oninput={(e) => set('height', Number(e.currentTarget.value))}
					/>
					<div class="flex flex-wrap gap-1">
						{#each SIZES as [w, h] (`${w}x${h}`)}
							<button
								type="button"
								class="rounded border border-gray-300 px-1.5 py-0.5 text-xs hover:bg-gray-100 dark:border-gray-600 dark:hover:bg-gray-700"
								class:ring-2={$store.width === w && $store.height === h}
								onclick={() => store.update((v) => ({ ...v, width: w, height: h }))}>{w}×{h}</button
							>
						{/each}
					</div>
				</div>
				<label class="mt-2 flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300"
					>Tiling
					<select
						class={ui.sized(ui.smallInput, 'w-52')}
						aria-label="Seamless tiling"
						value={$store.tiling ?? 'none'}
						onchange={(e) => set('tiling', e.currentTarget.value as Tiling)}
					>
						{#each TILINGS as t (t.value)}<option value={t.value}>{t.label}</option>{/each}
					</select>
					{#if ($store.tiling ?? 'none') !== 'none'}<span class={ui.help}
							>Check the seams in the preview's Tile view.</span
						>{/if}
				</label>
			</div>

			<!-- sampling -->
			<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
				<div>
					<label for="{mode}_steps" class={ui.label}>Steps</label>
					<input
						id="{mode}_steps"
						type="number"
						min="1"
						max="150"
						class={ui.input}
						value={$store.num_inference_steps}
						oninput={(e) => set('num_inference_steps', Number(e.currentTarget.value))}
					/>
				</div>
				<div>
					<label for="{mode}_cfg" class={ui.label}>Guidance</label>
					<input
						id="{mode}_cfg"
						type="number"
						step="0.5"
						min="0"
						class={ui.input}
						value={typeof $store.guidance_scale === 'number' ? $store.guidance_scale : 7}
						oninput={(e) => set('guidance_scale', Number(e.currentTarget.value))}
					/>
				</div>
				<div class="col-span-2">
					<label for="{mode}_scheduler" class={ui.label}>Sampler</label>
					<select
						id="{mode}_scheduler"
						class={ui.input}
						value={$store.scheduler_id ?? ''}
						onchange={(e) => set('scheduler_id', e.currentTarget.value || null)}
					>
						<option value="">Model default</option>
						{#each schedulers as s (s)}<option value={s}>{s}</option>{/each}
					</select>
				</div>
			</div>

			<!-- seed and amounts -->
			<div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
				<div class="col-span-2">
					<label for="{mode}_seed" class={ui.label}>Seed</label>
					<div class="flex gap-1">
						<input
							id="{mode}_seed"
							type="number"
							min="0"
							placeholder="random"
							class={ui.input}
							value={$store.seed ?? ''}
							oninput={(e) =>
								set('seed', e.currentTarget.value === '' ? null : Number(e.currentTarget.value))}
						/>
						<button
							type="button"
							class={ui.secondaryButton}
							title="New fixed seed"
							onclick={() => set('seed', Math.floor(Math.random() * 2 ** 32))}>🎲</button
						>
						<button
							type="button"
							class={ui.secondaryButton}
							title="Random seed for every image"
							onclick={() => set('seed', null)}>✕</button
						>
					</div>
				</div>
				<div>
					<label for="{mode}_n" class={ui.label}>Images per job</label>
					<input
						id="{mode}_n"
						type="number"
						min="1"
						max="4"
						class={ui.input}
						value={$store.num_outputs}
						oninput={(e) => set('num_outputs', Number(e.currentTarget.value))}
					/>
				</div>
				<div>
					<label for="{mode}_batch" class={ui.label}>Batch count</label>
					<input
						id="{mode}_batch"
						type="number"
						min="1"
						max="50"
						class={ui.input}
						bind:value={batchCount}
					/>
				</div>
			</div>
			{#if batchCount > 1 && $store.seed !== null && $store.seed !== undefined}
				<p class={ui.help}>With a fixed seed, each job of the batch uses the next seed.</p>
			{/if}

			<!-- advanced -->
			<details bind:open={showAdvanced}>
				<summary class="cursor-pointer text-sm text-gray-600 dark:text-gray-300">Advanced</summary>
				<div class="mt-2 space-y-3">
					<div>
						<label for="{mode}_syntax" class={ui.label}>Prompt syntax (SD 1.x/2.x, SDXL)</label>
						<select
							id="{mode}_syntax"
							class={ui.input}
							value={$store.prompt_syntax ?? 'a1111'}
							onchange={(e) => set('prompt_syntax', e.currentTarget.value as PromptSyntax)}
						>
							<option value="a1111">A1111: weights, long prompts, [from:to:when]</option>
							<option value="comfy">ComfyUI weights (same syntax)</option>
							<option value="plain">Plain text (cut at 77 tokens)</option>
						</select>
						<p class={ui.help}>
							(word:1.2) and [word] weigh words, BREAK starts a new 75-token chunk, [cat:dog:0.5]
							switches halfway, [cat|dog] alternates every step.
						</p>
					</div>
					<div>
						<label for="{mode}_sigmas" class={ui.label}>Custom sigmas (overrides steps)</label>
						<textarea
							id="{mode}_sigmas"
							rows="2"
							class="{ui.input} font-mono text-sm"
							placeholder="e.g. 14.6, 8.4, 5.2, …"
							bind:value={sigmasText}
							oninput={() => set('custom_sigmas', parseSigmas(sigmasText))}></textarea>
					</div>
				</div>
			</details>

			<div class="flex gap-2">
				<button type="submit" class="{ui.primaryButton} flex-1 py-3 text-base" disabled={busy}>
					Queue {imagesToQueue} image{imagesToQueue > 1 ? 's' : ''}{jobsToQueue > 1
						? ` (${jobsToQueue} jobs)`
						: ''}
				</button>
				<button
					type="button"
					class={ui.secondaryButton}
					title="Open these settings in the Lab (Taylor preview, attention maps, ablation …)"
					disabled={!$store.model_id}
					onclick={toLab}>Send to Lab</button
				>
			</div>
			{#if error}<div class={ui.errorBox} role="alert">{error}</div>{/if}
		</div>
	</form>
</div>
