<!-- frontend/src/lib/components/lab/SettlingChart.svelte -->
<!-- How much the region can still change when branching at each step (measured with
     branches) next to the free estimate from the render itself. Click a step to branch
     from it. Colours: measured = series 1 (blue), estimate = series 2 (orange, dashed). -->
<script lang="ts">
	import type { ExploreMapPoint } from '#lib/types.js';

	let {
		points,
		steps,
		window = null,
		selected,
		onpick
	}: {
		points: ExploreMapPoint[];
		steps: number;
		window?: number[] | null;
		selected: number;
		onpick: (step: number) => void;
	} = $props();

	const W = 560;
	const H = 220;
	const pad = { l: 40, r: 76, t: 12, b: 34 };
	const yMax = $derived(
		Math.max(
			1.1,
			...points.map((p) => p.change + p.spread / Math.max(points[0]?.distance || 1, 1e-9))
		)
	);
	const x = (step: number) => pad.l + (step / Math.max(1, steps - 1)) * (W - pad.l - pad.r);
	const y = (v: number) => pad.t + (1 - v / yMax) * (H - pad.t - pad.b);
	const line = (values: [number, number][]) =>
		values.map(([s, v], i) => `${i ? 'L' : 'M'}${x(s).toFixed(1)},${y(v).toFixed(1)}`).join(' ');

	const measured = $derived(line(points.map((p) => [p.step, p.change])));
	const scale = $derived(points[0]?.distance || 1);
	const band = $derived.by(() => {
		if (points.length < 2) return '';
		const up = points.map((p) => [p.step, p.change + p.spread / scale] as [number, number]);
		const down = points
			.map((p) => [p.step, Math.max(0, p.change - p.spread / scale)] as [number, number])
			.reverse();
		return `${line(up)} ${line(down).replace('M', 'L')} Z`;
	});
	const hasEstimate = $derived(points.some((p) => p.estimate !== null));
	const estimate = $derived(
		line(points.filter((p) => p.estimate !== null).map((p) => [p.step, p.estimate as number]))
	);
	const last = $derived(points[points.length - 1]);
	const ticksY = [0, 0.25, 0.5, 0.75, 1];
	const ticksX = $derived.by(() => {
		const n = Math.min(8, steps);
		return Array.from(
			new Set(
				Array.from({ length: n }, (_, i) => Math.round((i * (steps - 1)) / Math.max(1, n - 1)))
			)
		);
	});
	let hover = $state<ExploreMapPoint | null>(null);
</script>

<figure class="space-y-1">
	<div
		class="flex flex-wrap items-center gap-4 text-xs text-gray-600 dark:text-gray-300"
		aria-hidden="true"
	>
		<span class="flex items-center gap-1.5"
			><svg width="18" height="6"
				><line
					x1="0"
					y1="3"
					x2="18"
					y2="3"
					class="stroke-[#2a78d6] dark:stroke-[#3987e5]"
					stroke-width="2"
				/></svg
			>measured with branches (± spread)</span
		>
		{#if hasEstimate}
			<span class="flex items-center gap-1.5"
				><svg width="18" height="6"
					><line
						x1="0"
						y1="3"
						x2="18"
						y2="3"
						class="stroke-[#eb6834] dark:stroke-[#d95926]"
						stroke-width="2"
						stroke-dasharray="4 3"
					/></svg
				>free estimate (no branches)</span
			>
		{/if}
	</div>
	<div class="relative">
		<svg
			viewBox="0 0 {W} {H}"
			class="w-full"
			role="img"
			aria-label="Change of the region when branching at each step"
		>
			{#if window}
				<rect
					x={x(window[0])}
					y={pad.t}
					width={Math.max(0, x(window[1]) - x(window[0]))}
					height={H - pad.t - pad.b}
					class="fill-gray-200/60 dark:fill-gray-700/40"
				/>
				<text
					x={x(window[0]) + 4}
					y={pad.t + 12}
					class="fill-gray-500 text-[10px] dark:fill-gray-400">settles here</text
				>
			{/if}
			{#each ticksY as t (t)}
				<line
					x1={pad.l}
					x2={W - pad.r}
					y1={y(t)}
					y2={y(t)}
					class="stroke-gray-200 dark:stroke-gray-700"
					stroke-width="1"
				/>
				<text
					x={pad.l - 6}
					y={y(t) + 3}
					text-anchor="end"
					class="fill-gray-500 text-[10px] dark:fill-gray-400">{t}</text
				>
			{/each}
			{#each ticksX as s (s)}
				<text
					x={x(s)}
					y={H - pad.b + 14}
					text-anchor="middle"
					class="fill-gray-500 text-[10px] dark:fill-gray-400">{s}</text
				>
			{/each}
			<text
				x={(pad.l + W - pad.r) / 2}
				y={H - 4}
				text-anchor="middle"
				class="fill-gray-500 text-[10px] dark:fill-gray-400">branch at step (of {steps})</text
			>
			<line
				x1={x(selected)}
				x2={x(selected)}
				y1={pad.t}
				y2={H - pad.b}
				class="stroke-gray-500 dark:stroke-gray-300"
				stroke-width="1"
				stroke-dasharray="2 3"
			/>
			{#if band}<path d={band} class="fill-[#2a78d6]/15 dark:fill-[#3987e5]/20" />{/if}
			{#if hasEstimate}
				<path
					d={estimate}
					fill="none"
					class="stroke-[#eb6834] dark:stroke-[#d95926]"
					stroke-width="2"
					stroke-dasharray="5 4"
					stroke-linecap="round"
				/>
			{/if}
			<path
				d={measured}
				fill="none"
				class="stroke-[#2a78d6] dark:stroke-[#3987e5]"
				stroke-width="2"
				stroke-linejoin="round"
			/>
			{#each points as p (p.step)}
				<circle
					cx={x(p.step)}
					cy={y(p.change)}
					r="4"
					class="fill-[#2a78d6] stroke-white dark:fill-[#3987e5] dark:stroke-gray-800"
					stroke-width="2"
				/>
			{/each}
			{#if last}
				<text
					x={x(last.step) + 8}
					y={y(last.change) + 3}
					class="fill-gray-700 text-[10px] dark:fill-gray-200">measured</text
				>
				{#if last.estimate !== null}
					<text
						x={x(last.step) + 8}
						y={y(last.estimate) + 14}
						class="fill-gray-700 text-[10px] dark:fill-gray-200">estimate</text
					>
				{/if}
			{/if}
			<!-- hit targets bigger than the marks -->
			{#each points as p (p.step)}
				<rect
					x={x(p.step) - 14}
					y={pad.t}
					width="28"
					height={H - pad.t - pad.b}
					fill="transparent"
					class="cursor-pointer"
					role="button"
					tabindex="0"
					aria-label="Branch at step {p.step}: change {p.change.toFixed(2)}"
					onmouseenter={() => (hover = p)}
					onmouseleave={() => (hover = null)}
					onfocus={() => (hover = p)}
					onblur={() => (hover = null)}
					onclick={() => onpick(p.step)}
					onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && onpick(p.step)}
				/>
			{/each}
		</svg>
		{#if hover}
			<div
				class="pointer-events-none absolute top-1 rounded border border-gray-200 bg-white px-2 py-1 text-xs shadow dark:border-gray-600 dark:bg-gray-800"
				style:left="{(x(hover.step) / W) * 100}%"
				style:transform="translateX({hover.step > steps / 2 ? '-105%' : '5%'})"
			>
				<div class="font-medium text-gray-900 dark:text-gray-100">
					Step {hover.step} · σ {hover.sigma.toFixed(2)}
				</div>
				<div class="text-gray-600 dark:text-gray-300">
					measured {hover.change.toFixed(2)} ± {(hover.spread / scale).toFixed(2)}
				</div>
				{#if hover.estimate !== null}
					<div class="text-gray-600 dark:text-gray-300">estimate {hover.estimate.toFixed(2)}</div>
				{/if}
				<div class="text-gray-500">click to branch from here</div>
			</div>
		{/if}
	</div>
	<figcaption class="text-xs text-gray-500 dark:text-gray-400">
		1 = as much change as a completely new region (step 0); 0 = settled.
	</figcaption>
</figure>
<details class="text-xs text-gray-600 dark:text-gray-300">
	<summary class="cursor-pointer text-gray-500">Table</summary>
	<table class="mt-1 tabular-nums">
		<thead
			><tr class="text-gray-500"
				><th class="pr-3 text-left font-medium">Step</th><th class="pr-3 text-left font-medium"
					>σ</th
				><th class="pr-3 text-left font-medium">Measured</th><th class="pr-3 text-left font-medium"
					>± spread</th
				><th class="text-left font-medium">Estimate</th></tr
			></thead
		>
		<tbody>
			{#each points as p (p.step)}
				<tr
					><td class="pr-3">{p.step}</td><td class="pr-3">{p.sigma.toFixed(2)}</td><td class="pr-3"
						>{p.change.toFixed(2)}</td
					><td class="pr-3">{(p.spread / scale).toFixed(2)}</td><td
						>{p.estimate === null ? '—' : p.estimate.toFixed(2)}</td
					></tr
				>
			{/each}
		</tbody>
	</table>
</details>
