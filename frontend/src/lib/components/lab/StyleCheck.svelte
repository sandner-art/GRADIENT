<!-- frontend/src/lib/components/lab/StyleCheck.svelte -->
<!-- Style check: render the Lab's settings without a style and with each picked style
     (× position × start) on the same seeds; score style strength (DINO) and content kept
     (CLIP) and show them as a scatter, a table and the images. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { get, post } from '#lib/apiClient.js';
	import { baseProblem, baseRequest, runLabJob } from '#lib/lab/labState.svelte.js';
	import {
		axisMax,
		labelOf,
		parseStarts,
		sortRows,
		strengthOf,
		type SortKey,
		type StyleCheckResponse,
		type StyleCheckRow
	} from '#lib/lab/styleCheck.js';
	import * as ui from '#lib/lab/ui.js';
	import {
		POSITIONS,
		groupStyles,
		type PromptStyle,
		type StylePosition,
		type StylesList
	} from '#lib/styles.js';
	import type { LocalModel } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let all = $state<PromptStyle[]>([]);
	let search = $state('');
	let picked = $state<string[]>([]);
	let positions = $state<StylePosition[]>(['template']);
	let startsText = $state('0');
	let seeds = $state(2);
	let featureModel = $state<string | null>(null);
	let clipModel = $state<string | null>(null);
	let visionModels = $state<string[]>([]);
	let result = $state<StyleCheckResponse | null>(null);
	let error = $state<string | null>(null);
	let sortKey = $state<SortKey>('order');
	let selected = $state<StyleCheckRow | null>(null);

	onMount(async () => {
		try {
			all = (await get<StylesList>('/styles')).styles;
		} catch {
			all = [];
		}
		try {
			visionModels = (await get<LocalModel[]>('/models/local?type=vision')).map((m) => m.name);
		} catch {
			visionModels = [];
		}
	});

	const groups = $derived(groupStyles(all, search));
	const starts = $derived(parseStarts(startsText));
	const variants = $derived(picked.length * positions.length * starts.length);
	const renders = $derived(seeds * (variants + 1));
	const rows = $derived(result ? sortRows(result.rows, sortKey) : []);

	function pick(name: string) {
		if (name && !picked.includes(name) && picked.length < 16) picked = [...picked, name];
	}

	function togglePosition(p: StylePosition, on: boolean) {
		positions = on ? [...positions, p] : positions.filter((x) => x !== p);
	}

	async function run() {
		error = baseProblem();
		if (!error && !picked.length) error = 'Pick at least one style.';
		if (!error && !positions.length) error = 'Pick at least one position.';
		if (!error && variants > 32)
			error = `${variants} variants; at most 32 (styles × positions × starts).`;
		if (error) return;
		const body = {
			base_request: baseRequest(),
			styles: picked,
			positions,
			style_starts: starts,
			num_seeds: seeds,
			feature_model: featureModel,
			clip_model: clipModel
		};
		await runLabJob(`Style check: ${renders} renders…`, async () => {
			try {
				result = await post<StyleCheckResponse>('/lab/style-check', body);
				selected = null;
			} catch (e) {
				error = apiErrorMessage(e);
			}
		});
	}

	// scatter: x = strength (1 = a new seed), y = content kept (1 = as without the style)
	const W = 520,
		H = 300,
		L = 48,
		B = 36,
		T = 12,
		R = 16;
	const hasContent = $derived(!!result?.rows.some((r) => r.content_rel !== null));
	const xMax = $derived(axisMax(result?.rows.map(strengthOf) ?? []));
	const yMin = $derived(
		Math.min(0.8, ...(result?.rows.map((r) => r.content_rel ?? 1) ?? [1])) - 0.02
	);
	const yMax = $derived(Math.max(1.05, ...(result?.rows.map((r) => r.content_rel ?? 1) ?? [1])));
	const sx = (v: number) => L + (v / xMax) * (W - L - R);
	const sy = (v: number) => T + (1 - (v - yMin) / (yMax - yMin)) * (H - T - B);
	const pointLabel = (r: StyleCheckRow) =>
		`${r.style}${r.position !== 'template' ? ` · ${r.position}` : ''}`;
	// label the dots that have room; the others show their name on hover
	const labelled = $derived.by(() => {
		const placed: { x0: number; x1: number; y: number }[] = [];
		const out: StyleCheckRow[] = [];
		for (const r of result?.rows ?? []) {
			const x0 = sx(strengthOf(r)) + 8;
			const x1 = x0 + pointLabel(r).length * 6;
			const y = hasContent ? sy(r.content_rel ?? 1) : sy(1);
			if (x1 > W) continue;
			if (placed.some((b) => Math.abs(b.y - y) < 13 && x0 < b.x1 && b.x0 < x1)) continue;
			const covers = (result?.rows ?? []).some((o) => {
				const ox = sx(strengthOf(o)),
					oy = hasContent ? sy(o.content_rel ?? 1) : sy(1);
				return o !== r && ox > x0 - 6 && ox < x1 + 6 && Math.abs(oy - y) < 9;
			});
			if (covers) continue;
			placed.push({ x0, x1, y });
			out.push(r);
		}
		return out;
	});
	const fmt = (v: number | null | undefined, d = 2) =>
		v === null || v === undefined ? '—' : v.toFixed(d);
	const png = (b64: string) => `data:image/png;base64,${b64}`;
</script>

<div class="space-y-4">
	<div class={ui.card}>
		<h2 class={ui.heading}>Style check</h2>
		<p class="{ui.muted} mt-1">
			Renders the Lab's settings without a style and with each style you pick, on the same seeds.
			<b>Strength</b> is the DINO distance from the unstyled image, where 1 means the style changes
			the image as much as a new seed. <b>Content kept</b> is how well CLIP still finds your prompt in
			the image, where 1 means as well as without the style. Effective styles are strong and keep the
			content.
		</p>

		<div class="mt-3">
			<span class={ui.label}>Styles ({picked.length}/16)</span>
			{#if !all.length}
				<p class={ui.help}>No styles yet: import a styles.csv in Text-to-Image → Styles.</p>
			{:else}
				<div class="flex flex-wrap items-center gap-2">
					<input
						class={ui.sized(ui.smallInput, 'w-32')}
						placeholder="Search…"
						aria-label="Search styles"
						bind:value={search}
					/>
					<select
						class={ui.sized(ui.smallInput, 'min-w-0 flex-1')}
						aria-label="Style to check"
						value=""
						onchange={(e) => {
							pick(e.currentTarget.value);
							e.currentTarget.value = '';
						}}
					>
						<option value="">— add a style —</option>
						{#each groups as [group, items] (group)}
							<optgroup label={group}>
								{#each items as s (s.name)}
									<option value={s.name} disabled={picked.includes(s.name)}>{s.name}</option>
								{/each}
							</optgroup>
						{/each}
					</select>
				</div>
				{#if picked.length}
					<div class="mt-2 flex flex-wrap gap-1">
						{#each picked as name (name)}
							<span class="rounded bg-indigo-50 px-2 py-0.5 text-xs dark:bg-indigo-950"
								>{name}
								<button
									type="button"
									class="ml-1 text-red-600"
									aria-label="Remove {name}"
									onclick={() => (picked = picked.filter((n) => n !== name))}>✕</button
								></span
							>
						{/each}
					</div>
				{/if}
			{/if}
		</div>

		<fieldset class="mt-3">
			<legend class={ui.label}>Positions</legend>
			<div class="flex flex-wrap gap-3">
				{#each POSITIONS as p (p.value)}
					<label class="flex items-center gap-1 text-sm"
						><input
							type="checkbox"
							checked={positions.includes(p.value)}
							onchange={(e) => togglePosition(p.value, e.currentTarget.checked)}
						/>{p.label}</label
					>
				{/each}
			</div>
		</fieldset>

		<div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-4">
			<div>
				<label for="sc_starts" class={ui.label}>Style from</label>
				<input id="sc_starts" class={ui.input} placeholder="0, 0.3" bind:value={startsText} />
				<p class={ui.help}>Fractions of the steps, up to 4.</p>
			</div>
			<div>
				<label for="sc_seeds" class={ui.label}>Seeds</label>
				<input id="sc_seeds" type="number" min="1" max="4" class={ui.input} bind:value={seeds} />
			</div>
			<div>
				<label for="sc_fm" class={ui.label}>Strength model</label>
				<select id="sc_fm" class={ui.input} bind:value={featureModel}>
					<option value={null}>DINO (automatic)</option>
					{#each visionModels as m (m)}<option value={m}>{m}</option>{/each}
				</select>
			</div>
			<div>
				<label for="sc_clip" class={ui.label}>Content model</label>
				<select id="sc_clip" class={ui.input} bind:value={clipModel}>
					<option value={null}>First CLIP/SigLIP installed</option>
					<option value="none">Skip content kept</option>
					{#each visionModels as m (m)}<option value={m}>{m}</option>{/each}
				</select>
			</div>
		</div>

		{#if error}<div class="{ui.errorBox} mt-3" role="alert">{error}</div>{/if}
		<div class="mt-3 flex items-center justify-between gap-2">
			<p class={ui.help}>
				{variants} variant{variants === 1 ? '' : 's'} × {seeds} seed{seeds === 1 ? '' : 's'}, plus
				the unstyled images: {renders} renders. Seeds start at the Lab seed.
			</p>
			<button type="button" class={ui.primaryButton} onclick={run}>Run the check</button>
		</div>
	</div>

	{#if result}
		<div class="{ui.card} space-y-3">
			<p class="text-sm">
				{result.rows.length} variants on {result.seeds.length} seed{result.seeds.length === 1
					? ''
					: 's'} in {result.elapsed_s.toFixed(0)} s · strength by {result.feature_model}{result.clip_model
					? `, content by ${result.clip_model}`
					: ''}.
			</p>
			{#each result.notes as note, i (i)}<p class={ui.help}>{note}</p>{/each}

			<svg
				viewBox="0 0 {W} {H}"
				class="w-full max-w-xl"
				role="img"
				aria-label="Style strength against content kept"
			>
				<line x1={L} x2={W - R} y1={H - B} y2={H - B} stroke="currentColor" opacity="0.3" />
				<line x1={L} x2={L} y1={T} y2={H - B} stroke="currentColor" opacity="0.3" />
				<line
					x1={sx(1)}
					x2={sx(1)}
					y1={T}
					y2={H - B}
					stroke="currentColor"
					opacity="0.35"
					stroke-dasharray="4 4"
				/>
				{#if hasContent}
					<line
						x1={L}
						x2={W - R}
						y1={sy(1)}
						y2={sy(1)}
						stroke="currentColor"
						opacity="0.35"
						stroke-dasharray="4 4"
					/>
					<text x={L - 6} y={sy(1) + 4} text-anchor="end" font-size="11" fill="currentColor">1</text
					>
				{/if}
				<text x={sx(1)} y={H - B + 14} text-anchor="middle" font-size="11" fill="currentColor"
					>1 = a new seed</text
				>
				<text x={L} y={H - 6} font-size="11" fill="currentColor" opacity="0.7"
					>Strength (DINO) →</text
				>
				<text
					x="14"
					y={(T + H - B) / 2}
					font-size="11"
					fill="currentColor"
					opacity="0.7"
					text-anchor="middle"
					transform="rotate(-90 14 {(T + H - B) / 2})"
					>{hasContent ? 'Content kept (CLIP) →' : ''}</text
				>
				{#each result.rows as r (labelOf(r))}
					{@const y = hasContent ? sy(r.content_rel ?? 1) : sy(1)}
					<g
						role="button"
						tabindex="0"
						class="cursor-pointer"
						onclick={() => (selected = r)}
						onkeydown={(e) => e.key === 'Enter' && (selected = r)}
					>
						<title
							>{labelOf(r)}: strength {fmt(strengthOf(r))}, content kept {fmt(r.content_rel)}</title
						>
						<circle
							cx={sx(strengthOf(r))}
							cy={y}
							r={selected === r ? 7 : 5}
							fill="rgb(99 102 241)"
							fill-opacity="0.8"
							stroke="white"
							stroke-width="1.5"
						/>
						{#if labelled.includes(r)}
							<text
								x={sx(strengthOf(r)) + 8}
								y={y + 4}
								font-size="10.5"
								fill="currentColor"
								opacity="0.85">{pointLabel(r)}</text
							>
						{/if}
					</g>
				{/each}
			</svg>

			<div class="overflow-x-auto">
				<table class="w-full min-w-[40rem] text-left text-sm">
					<thead class="text-xs text-gray-500">
						<tr>
							<th class="py-1 pr-2 font-medium">Style</th>
							<th class="pr-2 font-medium">Position</th>
							<th class="pr-2 font-medium"
								><button
									type="button"
									class="hover:underline"
									onclick={() => (sortKey = 'strength')}
									>Strength{sortKey === 'strength' ? ' ↓' : ''}</button
								></th
							>
							<th class="pr-2 font-medium"
								><button type="button" class="hover:underline" onclick={() => (sortKey = 'content')}
									>Content kept{sortKey === 'content' ? ' ↓' : ''}</button
								></th
							>
							<th class="pr-2 font-medium" title="Similarity of the style's images across seeds"
								><button
									type="button"
									class="hover:underline"
									onclick={() => (sortKey = 'consistency')}
									>Consistency{sortKey === 'consistency' ? ' ↓' : ''}</button
								></th
							>
							<th class="pr-2 font-medium">Tokens</th>
							<th class="pr-2 font-medium">Notes</th>
						</tr>
					</thead>
					<tbody>
						{#each rows as r (labelOf(r))}
							<tr
								class="cursor-pointer border-t border-gray-200 align-top hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800"
								class:bg-indigo-50={selected === r}
								onclick={() => (selected = r)}
							>
								<td class="py-1 pr-2">{r.style}</td>
								<td class="pr-2"
									>{r.position}{r.start ? ` · from ${Math.round(r.start * 100)} %` : ''}</td
								>
								<td class="pr-2 tabular-nums">{fmt(strengthOf(r))}</td>
								<td class="pr-2 tabular-nums">{fmt(r.content_rel)}</td>
								<td class="pr-2 tabular-nums">{fmt(r.consistency)}</td>
								<td class="pr-2 tabular-nums"
									>{r.tokens ?? '—'}{r.chunks && r.chunks > 1 ? ` (${r.chunks} chunks)` : ''}</td
								>
								<td class="pr-2 text-xs text-amber-700 dark:text-amber-300">{r.notes.join('; ')}</td
								>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
			<p class={ui.help}>
				Unstyled: content {fmt(result.baseline_content, 3)} (CLIP), consistency {fmt(
					result.baseline_consistency
				)} across seeds{result.seed_distance !== null
					? `, seed-to-seed distance ${fmt(result.seed_distance, 3)}`
					: ''}. Click a row or a dot to see its images.
			</p>

			{#if selected}
				<div>
					<p class="text-sm font-medium">{labelOf(selected)}</p>
					<div
						class="mt-1 grid gap-2"
						style="grid-template-columns: repeat({result.seeds.length}, minmax(0, 1fr))"
					>
						{#each result.baseline as index (index)}
							<figure>
								<img
									src={png(result.cells[index].image_base64)}
									alt="No style, seed {result.cells[index].seed}"
									class="w-full rounded"
								/>
								<figcaption class={ui.help}>no style · seed {result.cells[index].seed}</figcaption>
							</figure>
						{/each}
						{#each selected.cells as index (index)}
							<figure>
								<img
									src={png(result.cells[index].image_base64)}
									alt="{selected.style}, seed {result.cells[index].seed}"
									class="w-full rounded"
								/>
								<figcaption class={ui.help}>
									{selected.style} · seed {result.cells[index].seed}
								</figcaption>
							</figure>
						{/each}
					</div>
				</div>
			{/if}

			<details>
				<summary class="cursor-pointer text-sm text-indigo-600 dark:text-indigo-400"
					>Contact sheet (saved in the gallery with every image)</summary
				>
				<img
					src={png(result.image_base64)}
					alt="All variants by seed"
					class="mt-2 max-w-full rounded"
				/>
			</details>
		</div>
	{/if}
</div>
