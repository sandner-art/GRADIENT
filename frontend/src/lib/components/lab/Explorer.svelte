<!-- frontend/src/lib/components/lab/Explorer.svelte -->
<!-- Travel through the model, one region at a time: mark a region, see when it settles,
     find the words acting there, and branch it (new noise, other words, a prompt mix,
     LoRA weights) from a chosen step while the rest of the image stays. Every branch can
     be continued, so the path through the model grows step by step.
     `compact` is the version beside the render forms (no map section, fewer options). -->
<script lang="ts">
	import { onMount } from 'svelte';
	import BranchViewer from './BranchViewer.svelte';
	import RegionCanvas from './RegionCanvas.svelte';
	import SettlingChart from './SettlingChart.svelte';
	import { get, post } from '#lib/apiClient.js';
	import {
		backTo,
		continueFrom,
		explore,
		loadImage,
		measure,
		openSession,
		renderBranches,
		replaceWord,
		suggestWords
	} from '#lib/lab/explore.svelte.js';
	import { baseProblem, baseRequest, lab } from '#lib/lab/labState.svelte.js';
	import * as ui from '#lib/lab/ui.js';
	import { goTo, notify, useAsInitImage } from '#lib/stores/ui.svelte.js';
	import type { ExploreBranch, ExploreBranchSpec, LocalModel } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let { compact = false }: { compact?: boolean } = $props();

	let numSteps = $state(6);
	let perStep = $state(3);
	// svelte-ignore state_referenced_locally
	let variations = $state(compact ? 4 : 3); // a starting value only
	let featureModel = $state<string | null>(null);
	let visionModels = $state<string[]>([]);
	let method = $state<'auto' | 'attention' | 'ablation'>('auto');
	/** chosen alternatives per word */
	let picked = $state<Record<string, string[]>>({});
	/** alternatives added by hand or by the LLM, per word */
	let extra = $state<Record<string, string[]>>({});
	let typed = $state<Record<string, string>>({});
	let mixPrompt = $state('');
	let mixWeight = $state(0.5);
	let loraWeights = $state('');
	let overlayChoice = $state<'none' | 'free' | 'measured' | 'word'>('none');
	let overlayWord = $state<string | null>(null);
	/** the branch viewer: null = closed, 0 = the explored image, i = branch i */
	let viewing = $state<number | null>(null);
	let llms = $state<string[]>([]);
	let asking = $state<string | null>(null);

	onMount(async () => {
		try {
			visionModels = (await get<LocalModel[]>('/models/local?type=vision')).map((m) => m.name);
		} catch {
			visionModels = [];
		}
		try {
			llms = await get<string[]>('/system/models/llm');
		} catch {
			llms = [];
		}
	});

	const start = $derived(explore.start);
	const steps = $derived(start?.steps ?? 0);
	const session = $derived(explore.session);
	const hasLora = $derived(!!session?.base_request.loras?.length);
	const busy = $derived(lab.busy !== null);

	async function beginFromLab() {
		const problem = baseProblem();
		if (problem) {
			explore.error = problem;
			return;
		}
		openSession({
			base_request: baseRequest(),
			init_image_base64: null,
			strength: null,
			path: [],
			source: 'the Lab settings'
		});
		picked = {};
		extra = {};
		await loadImage();
	}

	/** expected change at a step, interpolated from the measured map */
	function expectedAt(step: number): number | null {
		const pts = explore.map?.points;
		if (!pts?.length) return null;
		for (let i = 0; i < pts.length - 1; i++) {
			const a = pts[i];
			const b = pts[i + 1];
			if (step >= a.step && step <= b.step)
				return a.change + ((b.change - a.change) * (step - a.step)) / Math.max(1, b.step - a.step);
		}
		return step <= pts[0].step ? pts[0].change : pts[pts.length - 1].change;
	}

	const nearestFrame = $derived.by(() => {
		const frames = start?.frames ?? [];
		let best = frames[0];
		for (const f of frames) if (f.step <= explore.step) best = f;
		return best ?? null;
	});
	const nearestPoint = $derived.by(() => {
		const pts = explore.map?.points ?? [];
		let best = pts[0];
		for (const p of pts)
			if (Math.abs(p.step - explore.step) < Math.abs(best.step - explore.step)) best = p;
		return best ?? null;
	});

	const overlaySrc = $derived.by(() => {
		if (overlayChoice === 'free' && nearestFrame) return ui.img(nearestFrame.still_changing_base64);
		if (overlayChoice === 'measured' && nearestPoint) return ui.img(nearestPoint.heatmap_base64);
		if (overlayChoice === 'word' && overlayWord) {
			const w = explore.tokens?.words.find((t) => t.word === overlayWord);
			if (w?.overlay_base64) return ui.img(w.overlay_base64);
		}
		return null;
	});

	const alternativesFor = (word: string, base: string[]) => [
		...new Set([...base, ...(extra[word] ?? [])])
	];

	function toggle(word: string, alt: string) {
		const cur = picked[word] ?? [];
		picked[word] = cur.includes(alt) ? cur.filter((a) => a !== alt) : [...cur, alt];
	}

	function addTyped(word: string) {
		const value = (typed[word] ?? '').trim();
		if (!value) return;
		extra[word] = [...new Set([...(extra[word] ?? []), value])];
		toggle(word, value);
		typed[word] = '';
	}

	async function askLlm(word: string) {
		if (!llms.length || !start) return;
		asking = word;
		try {
			const res = await post<{ choices: { message: { content: string } }[] }>(
				'/llm/chat/completions',
				{
					llm_config: { model_id: llms[0] },
					messages: [
						{
							role: 'system',
							content:
								'You suggest single words or short phrases for image prompts. Answer with a comma-separated list only.'
						},
						{
							role: 'user',
							content: `Prompt: "${start.prompt}". Give 6 visually different alternatives for "${word}" that fit the prompt.`
						}
					],
					generation_params: { max_tokens: 80, temperature: 0.8 },
					stream: false
				}
			);
			const text = res.choices?.[0]?.message?.content ?? '';
			const words = text
				.split(/[,\n]/)
				.map((s) => s.replace(/^[\s\d.\-*"']+|["'.\s]+$/g, '').trim())
				.filter((s) => s && s.length < 40 && s.toLowerCase() !== word.toLowerCase())
				.slice(0, 6);
			if (!words.length) notify('The LLM gave no usable words', 'info');
			extra[word] = [...new Set([...(extra[word] ?? []), ...words])];
		} catch (e) {
			notify(`LLM: ${apiErrorMessage(e)}`, 'error');
		} finally {
			asking = null;
		}
	}

	const loraList = $derived(
		loraWeights
			.split(/[,\s]+/)
			.filter(Boolean)
			.map(Number)
			.filter((v) => Number.isFinite(v))
	);

	const specs = $derived.by((): ExploreBranchSpec[] => {
		const out: ExploreBranchSpec[] = [];
		for (let i = 0; i < variations; i++) out.push({ kind: 'seed' });
		const prompt = start?.prompt ?? '';
		for (const [word, alts] of Object.entries(picked))
			for (const alt of alts)
				out.push({
					kind: 'prompt',
					prompt: replaceWord(prompt, word, alt),
					label: `${word} → ${alt}`
				});
		if (mixPrompt.trim())
			out.push({ kind: 'mix', mix_prompt: mixPrompt.trim(), mix_weight: mixWeight });
		if (hasLora) for (const w of loraList) out.push({ kind: 'lora', lora_weight: w });
		return out;
	});

	async function branch() {
		if (specs.length > 12) {
			explore.error = `At most 12 branches at once (${specs.length} chosen).`;
			return;
		}
		viewing = null;
		await renderBranches(specs, featureModel);
		if (explore.branches?.branches.length) viewing = 1; // step through them with ← →
	}

	async function useAsInput(b: ExploreBranch) {
		await useAsInitImage(ui.png(b.image_base64), b.parameters);
	}

	const viewerItems = $derived(
		explore.branches
			? [
					{
						label: 'This image',
						src: ui.png(explore.branches.reference_image_base64),
						change: null
					},
					...explore.branches.branches.map((b) => ({
						label: b.label,
						src: ui.png(b.image_base64),
						change: b.change
					}))
				]
			: []
	);
	const branchAt = (i: number) => explore.branches?.branches[i - 1];

	const regionText = $derived(explore.mask ? 'in the marked region' : 'whole image');
	const changeText = (v: number | null) => (v === null ? '' : v.toFixed(3));
	const maxChange = $derived(
		Math.max(1e-9, ...(explore.branches?.branches.map((b) => b.change ?? 0) ?? [0]))
	);
</script>

<div class="space-y-4">
	{#if !compact}
		<div class={ui.card}>
			<h2 class={ui.heading}>Explorer</h2>
			<p class="{ui.muted} mt-1">
				Mark a region and travel through the model there: see at which step it settles, which words
				act on it, and branch it from any step (new noise, other words, a prompt mix, LoRA weights)
				while the rest of the image stays. Continue from any branch to go further.
			</p>
			<div class="mt-3 flex flex-wrap items-center gap-2">
				<button type="button" class={ui.primaryButton} disabled={busy} onclick={beginFromLab}
					>{session ? 'Start again from the Lab settings' : 'Explore the Lab settings'}</button
				>
				<span class={ui.help}>…or right-click any image → Explore.</span>
			</div>
		</div>
	{/if}

	{#if explore.error}<div class={ui.errorBox} role="alert">{explore.error}</div>{/if}

	{#if session && !start && !busy}
		<div class={ui.card}>
			<p class={ui.muted}>Exploring {session.source}.</p>
			<button type="button" class="{ui.secondaryButton} mt-2" onclick={loadImage}>Render it</button>
		</div>
	{/if}

	{#if start && session}
		<div class="{ui.card} space-y-3 {compact ? '!p-3' : ''}">
			<nav class="flex flex-wrap items-center gap-1 text-xs" aria-label="Path through the model">
				<button
					type="button"
					class="rounded px-1.5 py-0.5 hover:bg-gray-100 dark:hover:bg-gray-700"
					class:font-semibold={session.path.length === 0}
					disabled={busy}
					onclick={() => backTo(0)}>{session.source}</button
				>
				{#each session.path as op, i (i)}
					<span class="text-gray-400">›</span>
					<button
						type="button"
						class="rounded px-1.5 py-0.5 hover:bg-gray-100 dark:hover:bg-gray-700"
						class:font-semibold={i === session.path.length - 1}
						title="Branched at step {op.step}"
						disabled={busy}
						onclick={() => backTo(i + 1)}>{op.label ?? op.kind}</button
					>
				{/each}
			</nav>

			<RegionCanvas
				src={ui.png(start.image_base64)}
				mask={explore.mask}
				overlay={overlaySrc}
				{compact}
				onchange={(m) => (explore.mask = m)}
			/>

			<div class="flex flex-wrap items-center gap-3 text-xs">
				<label class="flex items-center gap-1 text-gray-600 dark:text-gray-300"
					>Overlay
					<select class={ui.sized(ui.smallInput, 'w-44')} bind:value={overlayChoice}>
						<option value="none">none</option>
						<option value="free" disabled={!start.frames.length}
							>still changing (free estimate)</option
						>
						<option value="measured" disabled={!explore.map}>measured change</option>
						<option value="word" disabled={!explore.tokens}>a word's map</option>
					</select>
				</label>
				{#if overlayChoice === 'word' && explore.tokens}
					<select class={ui.sized(ui.smallInput, 'w-32')} bind:value={overlayWord}>
						{#each explore.tokens.words as w (w.word)}<option value={w.word}>{w.word}</option
							>{/each}
					</select>
				{/if}
				<span class="text-gray-500"
					>seed {start.seed} · {start.steps} steps · {start.scheduler}</span
				>
			</div>
			{#if start.note}<p class={ui.help}>{start.note}</p>{/if}

			{#if !compact && start.frames.length}
				<div>
					<p class="mb-1 text-xs font-medium text-gray-600 dark:text-gray-300">
						How it formed: what the model predicted at each step (click to branch from there)
					</p>
					<div class="flex gap-1 overflow-x-auto pb-1">
						{#each start.frames as f (f.step)}
							<button
								type="button"
								class="shrink-0 rounded border-2 text-center text-[10px] text-gray-500"
								class:border-indigo-500={nearestFrame?.step === f.step}
								class:border-transparent={nearestFrame?.step !== f.step}
								onclick={() => (explore.step = f.step)}
							>
								<img
									src={ui.img(f.image_base64)}
									alt="Prediction at step {f.step}"
									class="h-20 w-20 rounded object-cover"
								/>
								step {f.step} · {Math.round(f.still_changing * 100)}%
							</button>
						{/each}
					</div>
				</div>
			{/if}
		</div>

		{#if !compact}
			<div class="{ui.card} space-y-3">
				<h3 class="font-semibold">1 · Where can it move?</h3>
				<p class={ui.help}>
					Branches the region with new noise at several steps and measures how much it changes (DINO
					features, or pixels without a vision model). Early steps change everything, late ones only
					details; the curve shows when this region settles.
				</p>
				<div class="flex flex-wrap items-end gap-3">
					<label class="text-xs"
						>Steps to test <input
							type="number"
							min="2"
							max="12"
							class={ui.sized(ui.smallInput, 'w-16')}
							bind:value={numSteps}
						/></label
					>
					<label class="text-xs"
						>Branches per step <input
							type="number"
							min="1"
							max="8"
							class={ui.sized(ui.smallInput, 'w-16')}
							bind:value={perStep}
						/></label
					>
					<label class="text-xs"
						>Vision model
						<select class={ui.sized(ui.smallInput, 'w-48')} bind:value={featureModel}>
							<option value={null}>Automatic (DINOv3, else DINOv2)</option>
							{#each visionModels as m (m)}<option value={m}>{m}</option>{/each}
						</select>
					</label>
					<button
						type="button"
						class={ui.secondaryButton}
						disabled={busy}
						onclick={() => measure(numSteps, perStep, featureModel)}
						>Measure ({numSteps * perStep} partial renders)</button
					>
				</div>
				{#if explore.map}
					<p class="text-sm">{explore.map.summary}</p>
					{#if explore.map.feature_note}<p class={ui.help}>{explore.map.feature_note}</p>{/if}
					<SettlingChart
						points={explore.map.points}
						{steps}
						window={explore.map.window}
						selected={explore.step}
						onpick={(s) => (explore.step = s)}
					/>
					<div class="flex flex-wrap gap-2">
						{#each [['big', 'Big changes'], ['variations', 'Variations'], ['details', 'Details']] as [key, text] (key)}
							{@const s = explore.map.suggested[key as 'big' | 'variations' | 'details']}
							<button type="button" class={ui.secondaryButton} onclick={() => (explore.step = s)}
								>{text}: step {s}</button
							>
						{/each}
					</div>
					<details>
						<summary class="cursor-pointer text-xs text-gray-500">Branches per step</summary>
						<img
							src={ui.png(explore.map.image_base64)}
							alt="One row per step: the branches"
							class="mt-2 max-w-full rounded"
						/>
					</details>
				{/if}
			</div>
		{/if}

		<div class="{ui.card} space-y-3 {compact ? '!p-3' : ''}">
			<div class="flex flex-wrap items-center justify-between gap-2">
				<h3 class="font-semibold">{compact ? 'Words acting here' : '2 · What drives it?'}</h3>
				<div class="flex items-center gap-2">
					{#if !compact}
						<select
							class={ui.sized(ui.smallInput, 'w-40')}
							bind:value={method}
							title="attention: one render; ablation: one render per word (any model)"
						>
							<option value="auto">Automatic</option>
							<option value="attention">Attention (1 render)</option>
							<option value="ablation">Ablation (1 per word)</option>
						</select>
					{/if}
					<button
						type="button"
						class={ui.secondaryButton}
						disabled={busy}
						onclick={() => suggestWords(method)}>Suggest words</button
					>
				</div>
			</div>
			{#if explore.tokens}
				{#if explore.tokens.note}<p class={ui.help}>{explore.tokens.note}</p>{/if}
				<ul class="space-y-2">
					{#each explore.tokens.words as w (w.word)}
						<li class="rounded border border-gray-200 p-2 dark:border-gray-700">
							<div class="flex items-center gap-2 text-sm">
								<span class="w-28 truncate font-medium" title={w.word}>{w.word}</span>
								<span
									class="h-2 flex-1 rounded bg-gray-100 dark:bg-gray-700"
									title="How much it acts in the region"
								>
									<span
										class="block h-2 rounded bg-indigo-500"
										style:width="{Math.round(w.score * 100)}%"
									></span>
								</span>
								<span
									class="w-24 text-right text-xs text-gray-500"
									title="1 = evenly spread; higher = concentrated in the region"
									>{w.lift.toFixed(1)}× here</span
								>
							</div>
							<div class="mt-1.5 flex flex-wrap items-center gap-1">
								{#each alternativesFor(w.word, w.alternatives) as alt (alt)}
									{@const on = (picked[w.word] ?? []).includes(alt)}
									<button
										type="button"
										class="rounded-full border px-2 py-0.5 text-xs"
										class:border-indigo-500={on}
										class:bg-indigo-600={on}
										class:text-white={on}
										class:border-gray-300={!on}
										class:dark:border-gray-600={!on}
										aria-pressed={on}
										onclick={() => toggle(w.word, alt)}>{alt}</button
									>
								{/each}
								<input
									class={ui.sized(ui.smallInput, 'w-28')}
									placeholder="other…"
									aria-label="Another word instead of {w.word}"
									bind:value={typed[w.word]}
									onkeydown={(e) => e.key === 'Enter' && addTyped(w.word)}
								/>
								{#if llms.length}
									<button
										type="button"
										class="text-xs text-indigo-600 hover:underline disabled:opacity-50 dark:text-indigo-400"
										disabled={asking !== null}
										onclick={() => askLlm(w.word)}
										>{asking === w.word ? 'asking…' : 'ask the LLM'}</button
									>
								{/if}
							</div>
						</li>
					{/each}
				</ul>
				<p class={ui.help}>
					Picked words become branches: the same noise, only the word differs. Alternatives come
					from the text encoder's vocabulary{llms.length ? ' or the LLM' : ''}.
				</p>
			{:else}
				<p class={ui.help}>
					Ranks the prompt's words by how much they act in the region (attention maps; ablation for
					models without them) and suggests other words for each.
				</p>
			{/if}
		</div>

		<div class="{ui.card} space-y-3 {compact ? '!p-3' : ''}">
			<h3 class="font-semibold">{compact ? 'Branch' : '3 · Branch'}</h3>
			<div>
				<label
					class="flex items-center justify-between text-xs text-gray-600 dark:text-gray-300"
					for="explore_step"
					><span>Branch from step <b>{explore.step}</b> of {steps}</span>
					<span class="text-gray-500"
						>{expectedAt(explore.step) !== null
							? `expected change ≈ ${(expectedAt(explore.step) as number).toFixed(2)}`
							: 'earlier = bigger changes, later = details'}</span
					></label
				>
				<input
					id="explore_step"
					type="range"
					min="0"
					max={Math.max(0, steps - 1)}
					bind:value={explore.step}
					class="w-full"
				/>
			</div>
			<div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
				<label class="text-xs"
					>New noise (variations)
					<input type="number" min="0" max="12" class={ui.smallInput} bind:value={variations} />
				</label>
				{#if !compact}
					<label class="text-xs"
						>Mix in a prompt (embedding average)
						<div class="flex gap-2">
							<input
								class={ui.smallInput}
								placeholder="e.g. made of glass"
								bind:value={mixPrompt}
							/>
							<input
								type="number"
								min="0"
								max="1"
								step="0.1"
								class={ui.sized(ui.smallInput, 'w-16')}
								bind:value={mixWeight}
								title="weight"
							/>
						</div>
					</label>
				{/if}
				{#if hasLora && !compact}
					<label class="text-xs"
						>LoRA weights to try
						<input class={ui.smallInput} placeholder="e.g. 0, 0.5, 1.5" bind:value={loraWeights} />
					</label>
				{/if}
			</div>
			<div class="flex flex-wrap items-center justify-between gap-2">
				<span class={ui.help}
					>{specs.length} branch{specs.length === 1 ? '' : 'es'} · {regionText} · each renders steps {explore.step}–{steps -
						1}</span
				>
				<div class="flex gap-2">
					{#if compact}
						<button type="button" class={ui.secondaryButton} onclick={() => goTo('lab')}
							>Open in Lab</button
						>
					{/if}
					<button
						type="button"
						class={ui.primaryButton}
						disabled={busy || !specs.length}
						onclick={branch}>Render branches</button
					>
				</div>
			</div>
		</div>

		{#if explore.branches}
			<div class="{ui.card} space-y-2 {compact ? '!p-3' : ''}">
				<p class={ui.help}>
					From step {explore.branches.step} · change measured in the region ({explore.branches
						.metric === 'dino'
						? 'DINO features'
						: 'pixels'}) · saved to the gallery · {(explore.branches.performance_ms / 1000).toFixed(
						1
					)} s
				</p>
				{#if explore.branches.feature_note}<p class={ui.help}>
						{explore.branches.feature_note}
					</p>{/if}
				{#if viewing !== null}
					<BranchViewer
						items={viewerItems}
						bind:index={viewing}
						mask={explore.mask}
						onclose={() => (viewing = null)}
						oncontinue={(i) => {
							const b = branchAt(i);
							if (b && !busy) {
								viewing = null;
								void continueFrom(b);
							}
						}}
						onuse={(i) => {
							const b = branchAt(i);
							if (b) void useAsInput(b);
						}}
					/>
				{:else}
					<button
						type="button"
						class="text-xs text-indigo-600 hover:underline dark:text-indigo-400"
						onclick={() => (viewing = 1)}
						>Compare in the viewer (← → to step through branches)</button
					>
					<div
						class="grid gap-2 {compact
							? 'grid-cols-2 sm:grid-cols-4'
							: 'grid-cols-2 md:grid-cols-3'}"
					>
						{#each explore.branches.branches as b, i (i)}
							<div class="rounded border border-gray-200 p-1.5 dark:border-gray-700">
								<button
									type="button"
									class="block w-full"
									onclick={() => (viewing = i + 1)}
									title="Open in the viewer (← → to step through, Space to compare)"
								>
									<img src={ui.png(b.image_base64)} alt={b.label} class="w-full rounded" />
								</button>
								<div class="mt-1 flex items-center gap-1 text-xs">
									<span class="flex-1 truncate font-medium" title={b.label}>{b.label}</span>
									{#if b.change !== null}
										<span
											class="h-1.5 w-10 rounded bg-gray-100 dark:bg-gray-700"
											title="Change in the region: {changeText(b.change)}"
										>
											<span
												class="block h-1.5 rounded bg-indigo-500"
												style:width="{Math.round(((b.change ?? 0) / maxChange) * 100)}%"
											></span>
										</span>
									{/if}
								</div>
								<div class="mt-1 flex flex-wrap gap-x-2 text-xs">
									<button
										type="button"
										class="text-indigo-600 hover:underline disabled:opacity-50 dark:text-indigo-400"
										disabled={busy}
										onclick={() => continueFrom(b)}>Continue from here</button
									>
									<button
										type="button"
										class="text-indigo-600 hover:underline dark:text-indigo-400"
										onclick={() => useAsInput(b)}>Use as input</button
									>
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
	{/if}
</div>
