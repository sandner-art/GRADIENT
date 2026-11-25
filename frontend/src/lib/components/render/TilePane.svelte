<!-- frontend/src/lib/components/render/TilePane.svelte -->
<!-- Tile view of an image: repeated 3 × 3 (or shifted by half, so the wrap edges meet in the
     middle) with seam scores, to check a seamless texture before using it. -->
<script lang="ts">
	import { imageUrl } from '#lib/gallery/api.js';
	import { seamScores, seamVerdict } from '#lib/tiling.js';

	let { id }: { id: number | null } = $props();

	let view = $state<'repeat' | 'offset'>('repeat');
	let scores = $state<{ x: number; y: number } | null>(null);
	let offsetUrl = $state<string | null>(null);

	$effect(() => {
		scores = null;
		offsetUrl = null;
		if (id === null) return;
		const img = new Image();
		img.crossOrigin = 'anonymous';
		img.onload = () => {
			const canvas = document.createElement('canvas');
			canvas.width = img.naturalWidth;
			canvas.height = img.naturalHeight;
			const ctx = canvas.getContext('2d');
			if (!ctx) return;
			ctx.drawImage(img, 0, 0);
			const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
			scores = seamScores(data, canvas.width, canvas.height);
			// shifted by half: the four quarters swap, so the wrap edges meet in the middle
			const w = canvas.width,
				h = canvas.height;
			const shifted = document.createElement('canvas');
			shifted.width = w;
			shifted.height = h;
			const sctx = shifted.getContext('2d');
			if (!sctx) return;
			for (const [dx, dy] of [
				[-w / 2, -h / 2],
				[w / 2, -h / 2],
				[-w / 2, h / 2],
				[w / 2, h / 2]
			])
				sctx.drawImage(img, Math.round(dx), Math.round(dy));
			offsetUrl = shifted.toDataURL('image/png');
		};
		img.src = imageUrl(id);
	});
	const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '∞');
</script>

{#if id === null}
	<p class="p-6 text-center text-sm text-gray-500">Render or pick an image to see it tiled.</p>
{:else}
	<div class="flex h-full w-full flex-col">
		<div class="flex items-center justify-between gap-2 p-1 text-xs">
			<div class="inline-flex rounded-md shadow-sm" role="group" aria-label="Tile view">
				{#each [['repeat', '3 × 3'], ['offset', 'Offset ½']] as [v, label], i (v)}
					<button
						type="button"
						class="border border-gray-300 px-2 py-0.5 dark:border-gray-600 {i === 0
							? 'rounded-l-md'
							: '-ml-px rounded-r-md'}"
						class:bg-indigo-600={view === v}
						class:text-white={view === v}
						aria-pressed={view === v}
						onclick={() => (view = v as 'repeat' | 'offset')}>{label}</button
					>
				{/each}
			</div>
			{#if scores}
				<span
					class="text-gray-500"
					title="Colour jump across the wrap edge / an ordinary step (1 = no seam)"
					>Left–right {fmt(scores.x)} ({seamVerdict(scores.x)}) · top–bottom {fmt(scores.y)} ({seamVerdict(
						scores.y
					)})</span
				>
			{/if}
		</div>
		{#if view === 'repeat'}
			<div
				class="min-h-0 flex-1 bg-gray-200 dark:bg-gray-800"
				role="img"
				aria-label="The image repeated three times each way"
				style:background-image="url({imageUrl(id)})"
				style:background-repeat="repeat"
				style:background-size="33.333% auto"
			></div>
		{:else if offsetUrl}
			<img
				src={offsetUrl}
				alt="Shifted by half, so the edges meet in the middle"
				class="min-h-0 flex-1 object-contain"
			/>
		{/if}
	</div>
{/if}
