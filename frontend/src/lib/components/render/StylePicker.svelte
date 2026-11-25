<!-- frontend/src/lib/components/render/StylePicker.svelte -->
<!-- Prompt styles (A1111 styles.csv): pick styles in order, choose where their text goes,
     preview the prompts the render will use (token counts), apply them to the prompt. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { get, post } from '#lib/apiClient.js';
	import * as ui from '#lib/lab/ui.js';
	import {
		POSITIONS,
		groupStyles,
		needsPreview,
		type PromptStyle,
		type StylePosition,
		type StylePreview,
		type StylesList
	} from '#lib/styles.js';
	import type { LoRAParameter, Text2ImageRequest } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	interface Props {
		store: Writable<Text2ImageRequest>;
		preview?: StylePreview | null;
	}
	let { store, preview = $bindable(null) }: Props = $props();

	let list = $state<StylesList | null>(null);
	let search = $state('');
	let picked = $state('');
	let error = $state<string | null>(null);
	let showPreview = $state(false);
	let fileInput = $state<HTMLInputElement | null>(null);
	let importedNote = $state('');

	const byName = $derived(new Map((list?.styles ?? []).map((s) => [s.name, s])));
	const groups = $derived(groupStyles(list?.styles ?? [], search));
	const selected = $derived($store.styles ?? []);
	const plain = $derived(($store.prompt_syntax ?? 'a1111') === 'plain');

	async function load() {
		try {
			list = await get<StylesList>('/styles');
		} catch (e) {
			error = apiErrorMessage(e);
		}
	}
	onMount(load);

	function set<K extends keyof Text2ImageRequest>(key: K, value: Text2ImageRequest[K]) {
		store.update((v) => ({ ...v, [key]: value }));
	}

	function add(name: string) {
		if (name && !selected.includes(name)) set('styles', [...selected, name]);
		picked = '';
	}

	function remove(name: string) {
		const next = selected.filter((n) => n !== name);
		set('styles', next.length ? next : null);
	}

	async function importFile(file: File | undefined) {
		if (!file) return;
		error = null;
		try {
			const r = await post<{ file: string; styles: number }>('/styles/import', {
				filename: file.name,
				content: await file.text()
			});
			await load();
			error = null;
			search = '';
			picked = '';
			importedNote = `${r.styles} styles from ${r.file}`;
		} catch (e) {
			error = apiErrorMessage(e);
		} finally {
			if (fileInput) fileInput.value = '';
		}
	}

	// the prompts a render will use (debounced)
	function previewBody() {
		return {
			model_id: $store.model_id || null,
			prompt: $store.prompt ?? '',
			negative_prompt: $store.negative_prompt ?? '',
			styles: $store.styles ?? null,
			style_position: $store.style_position ?? 'template',
			style_start: $store.style_start ?? 0,
			prompt_syntax: $store.prompt_syntax ?? 'a1111',
			seed: $store.seed ?? null
		};
	}
	let timer: ReturnType<typeof setTimeout> | undefined;
	let asked = 0;
	$effect(() => {
		const body = previewBody();
		clearTimeout(timer);
		if (!needsPreview(body.prompt, body.negative_prompt, body.styles)) {
			preview = null;
			return;
		}
		const ticket = ++asked;
		timer = setTimeout(async () => {
			try {
				const r = await post<StylePreview>('/styles/preview', body);
				if (ticket === asked) preview = r;
			} catch {
				if (ticket === asked) preview = null;
			}
		}, 350);
		return () => clearTimeout(timer);
	});

	/** Write the styled prompts into the prompt boxes and clear the styles. */
	async function applyToPrompt() {
		let fresh: StylePreview;
		try {
			fresh = await post<StylePreview>('/styles/preview', previewBody()); // never a stale one
		} catch (e) {
			error = apiErrorMessage(e);
			return;
		}
		const loras: LoRAParameter[] = [...($store.loras ?? [])];
		for (const l of fresh.loras)
			if (!loras.some((x) => x.name === l.name)) loras.push({ name: l.name, weight: l.weight });
		store.update((v) => ({
			...v,
			prompt: fresh.prompt,
			negative_prompt: fresh.negative_prompt ?? '',
			loras,
			styles: null,
			style_start: 0
		}));
	}

	const missing = $derived(selected.filter((n) => list && !byName.has(n)));
	const startPercent = $derived(Math.round(($store.style_start ?? 0) * 100));
	function title(style: PromptStyle | undefined): string {
		if (!style) return 'Not in the styles folder';
		return `${style.prompt}${style.negative_prompt ? `\nNegative: ${style.negative_prompt}` : ''}\n(${style.file})`;
	}
</script>

<div>
	<div class="mb-1 flex flex-wrap items-center justify-between gap-2">
		<span class={ui.label}>Styles</span>
		<div class="flex items-center gap-2">
			<button
				type="button"
				class="text-sm text-indigo-600 hover:underline dark:text-indigo-400"
				onclick={() => fileInput?.click()}>Import styles…</button
			>
			<input
				bind:this={fileInput}
				type="file"
				accept=".csv,.json"
				class="hidden"
				aria-label="Styles file to import"
				onchange={(e) => importFile(e.currentTarget.files?.[0])}
			/>
		</div>
	</div>

	{#if list && !list.styles.length}
		<p class={ui.help}>
			No styles yet. Import an A1111 <code>styles.csv</code> (for example from
			github.com/sandner-art/ai-research,
			<code>A1111_Styles</code>) or put .csv/.json files into <code>{list.folder}</code>.
		</p>
	{:else if list}
		<div class="flex flex-wrap items-center gap-2">
			<input
				class={ui.sized(ui.smallInput, 'w-32')}
				placeholder="Search…"
				aria-label="Search styles"
				bind:value={search}
			/>
			<select
				class={ui.sized(ui.smallInput, 'min-w-0 flex-1')}
				aria-label="Style to add"
				bind:value={picked}
				onchange={() => add(picked)}
			>
				<option value="">— add a style ({list.styles.length}) —</option>
				{#each groups as [group, items] (group)}
					<optgroup label={group}>
						{#each items as s (s.name)}
							<option value={s.name} disabled={selected.includes(s.name)}>{s.name}</option>
						{/each}
					</optgroup>
				{/each}
			</select>
		</div>
	{/if}
	{#if importedNote}<p class={ui.help}>Imported {importedNote}.</p>{/if}

	{#if selected.length}
		<div class="mt-2 flex flex-wrap gap-1">
			{#each selected as name, i (name)}
				<span
					class="rounded px-2 py-0.5 text-xs {byName.has(name) || !list
						? 'bg-indigo-50 text-indigo-900 dark:bg-indigo-950 dark:text-indigo-100'
						: 'bg-amber-100 text-amber-900 dark:bg-amber-900 dark:text-amber-100'}"
					title={title(byName.get(name))}
					><span class="opacity-60">{i + 1}.</span>
					{name}
					<button
						type="button"
						class="ml-1 text-red-600"
						aria-label="Remove style {name}"
						onclick={() => remove(name)}>✕</button
					></span
				>
			{/each}
		</div>
		{#if missing.length}
			<p class="mt-1 text-xs text-amber-700 dark:text-amber-300">
				Not in the styles folder (skipped): {missing.join(', ')}
			</p>
		{/if}
		<div class="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
			<label class="text-xs text-gray-600 dark:text-gray-400"
				>Position
				<select
					class={ui.smallInput}
					value={$store.style_position ?? 'template'}
					onchange={(e) => set('style_position', e.currentTarget.value as StylePosition)}
				>
					{#each POSITIONS as p (p.value)}<option value={p.value}>{p.label}</option>{/each}
				</select>
			</label>
			<label class="text-xs text-gray-600 dark:text-gray-400"
				>Style from {startPercent ? `${startPercent} % of the steps` : 'the first step'}
				<input
					type="range"
					min="0"
					max="0.9"
					step="0.05"
					class="w-full"
					value={$store.style_start ?? 0}
					disabled={plain}
					oninput={(e) => set('style_start', Number(e.currentTarget.value))}
				/>
			</label>
		</div>
	{/if}

	{#if preview}
		<div class="mt-2 text-xs">
			<div class="flex flex-wrap items-center gap-3">
				<button
					type="button"
					class="text-indigo-600 hover:underline dark:text-indigo-400"
					aria-expanded={showPreview}
					onclick={() => (showPreview = !showPreview)}
					>{showPreview ? 'Hide' : 'Show'} the prompt as rendered</button
				>

				{#if selected.length}
					<button
						type="button"
						class="text-indigo-600 hover:underline dark:text-indigo-400"
						title="Write the styled prompts into the prompt boxes and clear the styles"
						onclick={applyToPrompt}>Apply to prompt</button
					>
				{/if}
			</div>
			{#if showPreview}
				<div
					class="mt-1 rounded border border-gray-200 bg-gray-50 p-2 dark:border-gray-700 dark:bg-gray-800"
				>
					<p class="break-words whitespace-pre-wrap">{preview.prompt}</p>
					{#if preview.negative_prompt}
						<p class="mt-1 break-words whitespace-pre-wrap text-gray-500 dark:text-gray-400">
							Negative: {preview.negative_prompt}
						</p>
					{/if}
					{#if preview.loras.length}
						<p class="mt-1 text-gray-500 dark:text-gray-400">
							LoRAs from tags: {preview.loras.map((l) => `${l.name} ${l.weight}`).join(', ')}
						</p>
					{/if}
					{#if Array.isArray(preview.notes.lora_tags_missing)}
						<p class="mt-1 text-amber-700 dark:text-amber-300">
							LoRA tags not found: {(preview.notes.lora_tags_missing as string[]).join(', ')}
						</p>
					{/if}
					{#if !preview.reads_syntax && preview.model_family}
						<p class="mt-1 text-gray-500 dark:text-gray-400">
							This model reads the text as it is: weights, BREAK and the style start apply to SD
							1.x/2.x and SDXL.
						</p>
					{/if}
				</div>
			{/if}
		</div>
	{/if}
	{#if error}<p class="mt-1 text-xs text-red-600">{error}</p>{/if}
</div>
