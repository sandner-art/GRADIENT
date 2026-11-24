<!-- frontend/src/lib/components/lab/BranchViewer.svelte -->
<!-- Step through the explored image and its branches with the arrow keys and compare them:
     hold Space to flip to the reference (the original, or a pinned branch), W for a wipe,
     D for the difference, M for the region outline. The reference is "This image" until
     you pin another one (R). -->
<script module lang="ts">
	export interface ViewerItem {
		label: string;
		src: string;
		change: number | null;
	}
</script>

<script lang="ts">
	import { ui } from '#lib/stores/ui.svelte.js';

	let {
		items,
		index = $bindable(0),
		mask = null,
		onclose,
		oncontinue,
		onuse
	}: {
		items: ViewerItem[];
		index: number;
		mask?: string | null;
		onclose: () => void;
		oncontinue?: (i: number) => void;
		onuse?: (i: number) => void;
	} = $props();

	type Mode = 'view' | 'wipe' | 'diff';
	let mode = $state<Mode>('view');
	let reference = $state(0);
	let flipped = $state(false);
	let outline = $state(true);
	let wipe = $state(50);
	let diffSrc = $state<string | null>(null);
	let outlineSrc = $state<string | null>(null);
	let root = $state<HTMLElement | null>(null);
	let natural = $state({ w: 0, h: 0 });

	function measure(e: Event) {
		const img = e.currentTarget as HTMLImageElement;
		if (img.naturalWidth !== natural.w || img.naturalHeight !== natural.h)
			natural = { w: img.naturalWidth, h: img.naturalHeight };
	}
	/** as wide as fits, no taller than the viewport allows; all layers fill this box */
	const boxWidth = $derived(
		natural.w
			? `min(100%, ${(72 * natural.w) / natural.h}vh, ${Math.max(natural.w, 768)}px)`
			: 'fit-content'
	);
	const pixelated = $derived(natural.w > 0 && natural.w < 256 ? 'pixelated' : 'auto');

	const current = $derived(items[Math.min(index, items.length - 1)]);
	const ref = $derived(items[Math.min(reference, items.length - 1)]);
	const shown = $derived(flipped ? ref : current);
	const maxChange = $derived(Math.max(1e-9, ...items.map((i) => i.change ?? 0)));

	const go = (delta: number) => {
		flipped = false;
		index = (index + delta + items.length) % items.length;
	};

	function load(src: string): Promise<HTMLImageElement> {
		return new Promise((resolve, reject) => {
			const img = new Image();
			img.onload = () => resolve(img);
			img.onerror = reject;
			img.src = src;
		});
	}

	// difference image: |current - reference|, amplified, as a heat colour
	$effect(() => {
		if (mode !== 'diff' || !current || !ref) return;
		const a = current.src;
		const b = ref.src;
		let cancelled = false;
		Promise.all([load(a), load(b)]).then(([ia, ib]) => {
			if (cancelled) return;
			const w = ia.naturalWidth;
			const h = ia.naturalHeight;
			const c = document.createElement('canvas');
			c.width = w;
			c.height = h;
			const ctx = c.getContext('2d')!;
			ctx.drawImage(ia, 0, 0, w, h);
			const da = ctx.getImageData(0, 0, w, h);
			ctx.drawImage(ib, 0, 0, w, h);
			const db = ctx.getImageData(0, 0, w, h);
			const out = ctx.createImageData(w, h);
			for (let i = 0; i < da.data.length; i += 4) {
				const d =
					(Math.abs(da.data[i] - db.data[i]) +
						Math.abs(da.data[i + 1] - db.data[i + 1]) +
						Math.abs(da.data[i + 2] - db.data[i + 2])) /
					3;
				const v = Math.min(255, d * 4);
				out.data[i] = v; // black (same) → orange → yellow (different)
				out.data[i + 1] = v * 0.6;
				out.data[i + 2] = v * 0.15;
				out.data[i + 3] = 255;
			}
			ctx.putImageData(out, 0, 0);
			diffSrc = c.toDataURL('image/png');
		});
		return () => {
			cancelled = true;
		};
	});

	// region outline from the mask (white = region)
	$effect(() => {
		const m = mask;
		const base = items[0]?.src;
		if (!m || !base) {
			outlineSrc = null;
			return;
		}
		let cancelled = false;
		Promise.all([load(m), load(base)]).then(([im, ib]) => {
			if (cancelled) return;
			const w = ib.naturalWidth;
			const h = ib.naturalHeight;
			const c = document.createElement('canvas');
			c.width = w;
			c.height = h;
			const ctx = c.getContext('2d')!;
			ctx.drawImage(im, 0, 0, w, h);
			const src = ctx.getImageData(0, 0, w, h).data;
			const out = ctx.createImageData(w, h);
			const inside = (x: number, y: number) =>
				x >= 0 && y >= 0 && x < w && y < h && src[(y * w + x) * 4] > 127;
			const r = Math.max(1, Math.round(Math.max(w, h) / 256));
			for (let y = 0; y < h; y++)
				for (let x = 0; x < w; x++) {
					if (!inside(x, y)) continue;
					if (inside(x - r, y) && inside(x + r, y) && inside(x, y - r) && inside(x, y + r))
						continue;
					const i = (y * w + x) * 4;
					out.data[i] = 255;
					out.data[i + 1] = 214;
					out.data[i + 2] = 10;
					out.data[i + 3] = 255;
				}
			ctx.putImageData(out, 0, 0);
			outlineSrc = c.toDataURL('image/png');
		});
		return () => {
			cancelled = true;
		};
	});

	function typing(target: EventTarget | null) {
		const el = target as HTMLElement | null;
		return !!el && (el.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName));
	}

	/** only the visible viewer reacts (the Lab keeps hidden tools mounted) */
	const active = () => !!root && root.offsetParent !== null && ui.viewer === null;

	function keydown(e: KeyboardEvent) {
		if (!active() || typing(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
		const key = e.key.toLowerCase();
		const actions: Record<string, () => void> = {
			arrowright: () => go(1),
			arrowleft: () => go(-1),
			arrowdown: () => go(1),
			arrowup: () => go(-1),
			home: () => (index = 0),
			' ': () => (flipped = true),
			w: () => (mode = mode === 'wipe' ? 'view' : 'wipe'),
			d: () => (mode = mode === 'diff' ? 'view' : 'diff'),
			m: () => (outline = !outline),
			r: () => (reference = index),
			o: () => (reference = 0),
			escape: () => onclose(),
			enter: () => index > 0 && oncontinue?.(index)
		};
		const act = actions[key];
		if (!act) return;
		e.preventDefault();
		act();
	}

	function keyup(e: KeyboardEvent) {
		if (e.key === ' ') flipped = false;
	}

	const tab = (on: boolean) =>
		`rounded border px-2 py-0.5 text-xs ${on ? 'border-indigo-500 bg-indigo-600 text-white' : 'border-gray-300 text-gray-700 dark:border-gray-600 dark:text-gray-200'}`;
</script>

<svelte:window onkeydown={keydown} onkeyup={keyup} />

<div bind:this={root} class="space-y-2" role="region" aria-label="Branch viewer">
	<div class="flex flex-wrap items-center gap-2 text-xs">
		<button type="button" class={tab(false)} onclick={() => go(-1)} aria-label="Previous (←)"
			>‹</button
		>
		<span class="text-gray-600 tabular-nums dark:text-gray-300">{index + 1} / {items.length}</span>
		<button type="button" class={tab(false)} onclick={() => go(1)} aria-label="Next (→)">›</button>
		<span class="max-w-[16rem] truncate font-medium" title={current?.label}>{current?.label}</span>
		{#if current?.change !== null && current?.change !== undefined}
			<span class="text-gray-500">change {current.change.toFixed(3)}</span>
		{/if}
		<span class="mx-1 h-4 w-px bg-gray-300 dark:bg-gray-600"></span>
		<button type="button" class={tab(mode === 'view')} onclick={() => (mode = 'view')}>View</button>
		<button
			type="button"
			class={tab(mode === 'wipe')}
			onclick={() => (mode = mode === 'wipe' ? 'view' : 'wipe')}
			title="W">Wipe</button
		>
		<button
			type="button"
			class={tab(mode === 'diff')}
			onclick={() => (mode = mode === 'diff' ? 'view' : 'diff')}
			title="D">Difference</button
		>
		<button
			type="button"
			class={tab(flipped)}
			onpointerdown={() => (flipped = true)}
			onpointerup={() => (flipped = false)}
			onpointerleave={() => (flipped = false)}
			title="Hold (or hold Space) to see the reference">Hold: reference</button
		>
		{#if mask}
			<button type="button" class={tab(outline)} onclick={() => (outline = !outline)} title="M"
				>Region</button
			>
		{/if}
		<span class="ml-auto flex items-center gap-2">
			<span class="text-gray-500">reference: <b>{ref?.label}</b></span>
			{#if index !== reference}
				<button
					type="button"
					class="text-indigo-600 hover:underline dark:text-indigo-400"
					onclick={() => (reference = index)}
					title="R">use this</button
				>
			{/if}
			<button type="button" class="text-gray-500 hover:underline" onclick={onclose} title="Esc"
				>Close</button
			>
		</span>
	</div>

	<div class="relative mx-auto max-w-full select-none" style:width={boxWidth}>
		{#if mode === 'diff' && diffSrc && !flipped}
			<img
				src={diffSrc}
				alt="Difference from the reference"
				class="block h-auto w-full rounded"
				style:image-rendering={pixelated}
				onload={measure}
			/>
		{:else}
			<img
				src={shown?.src}
				alt={shown?.label}
				class="block h-auto w-full rounded"
				style:image-rendering={pixelated}
				onload={measure}
			/>
			{#if mode === 'wipe' && !flipped}
				<img
					src={ref?.src}
					alt="Reference"
					class="pointer-events-none absolute inset-0 h-full w-full rounded"
					style:image-rendering={pixelated}
					style:clip-path="inset(0 {100 - wipe}% 0 0)"
				/>
				<div
					class="pointer-events-none absolute inset-y-0 w-0.5 bg-white/90 shadow"
					style:left="{wipe}%"
				></div>
			{/if}
		{/if}
		{#if outline && outlineSrc}
			<img
				src={outlineSrc}
				alt=""
				class="pointer-events-none absolute inset-0 h-full w-full rounded opacity-80"
			/>
		{/if}
		{#if flipped}
			<span class="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 text-xs text-white"
				>reference: {ref?.label}</span
			>
		{/if}
	</div>
	{#if mode === 'wipe'}
		<input
			type="range"
			min="0"
			max="100"
			bind:value={wipe}
			class="w-full"
			aria-label="Wipe: reference on the left"
		/>
	{/if}

	<div class="flex gap-1 overflow-x-auto pb-1" role="listbox" aria-label="Branches">
		{#each items as item, i (i)}
			<button
				type="button"
				role="option"
				aria-selected={i === index}
				class="relative shrink-0 rounded border-2"
				class:border-indigo-500={i === index}
				class:border-transparent={i !== index}
				title={item.label}
				onclick={() => {
					flipped = false;
					index = i;
				}}
			>
				<img src={item.src} alt={item.label} class="h-16 w-16 rounded object-cover" />
				{#if i === reference}
					<span class="absolute top-0.5 left-0.5 rounded bg-black/60 px-1 text-[9px] text-white"
						>ref</span
					>
				{/if}
				{#if item.change !== null && i > 0}
					<span class="absolute inset-x-1 bottom-0.5 h-1 rounded bg-black/30">
						<span
							class="block h-1 rounded bg-indigo-400"
							style:width="{Math.round(((item.change ?? 0) / maxChange) * 100)}%"
						></span>
					</span>
				{/if}
			</button>
		{/each}
	</div>
	<div class="flex flex-wrap items-center justify-between gap-2 text-xs text-gray-500">
		<span
			>← → step · hold Space: reference · W wipe · D difference · M region · R pin as reference · O
			original · Enter continue · Esc close</span
		>
		{#if index > 0}
			<span class="flex gap-2">
				{#if oncontinue}
					<button
						type="button"
						class="text-indigo-600 hover:underline dark:text-indigo-400"
						onclick={() => oncontinue?.(index)}>Continue from here</button
					>
				{/if}
				{#if onuse}
					<button
						type="button"
						class="text-indigo-600 hover:underline dark:text-indigo-400"
						onclick={() => onuse?.(index)}>Use as input</button
					>
				{/if}
			</span>
		{/if}
	</div>
</div>
