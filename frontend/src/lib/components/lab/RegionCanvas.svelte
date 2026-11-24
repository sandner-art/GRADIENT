<!-- frontend/src/lib/components/lab/RegionCanvas.svelte -->
<!-- An image you can mark a region on (rectangle, brush, eraser). The region is reported
     as a PNG data URL (white = region, black = the rest) at the image's own size; null
     when nothing is marked (= the whole image). An optional overlay (a heatmap) can be
     shown over the image. -->
<script lang="ts">
	let {
		src,
		mask = null,
		overlay = null,
		onchange,
		compact = false
	}: {
		src: string;
		mask?: string | null;
		overlay?: string | null;
		onchange: (mask: string | null) => void;
		compact?: boolean;
	} = $props();

	type Tool = 'rect' | 'brush' | 'erase';
	let tool = $state<Tool>('rect');
	let brush = $state(48);
	let canvas = $state<HTMLCanvasElement | null>(null);
	let image = $state<HTMLImageElement | null>(null);
	let size = $state({ w: 0, h: 0 });
	let drawing = false;
	let start = { x: 0, y: 0 };
	let snapshot: ImageData | null = null;
	let lastMask: string | null = null;

	const FILL = 'rgb(99, 102, 241)';

	function onLoad() {
		if (!image) return;
		const w = image.naturalWidth;
		const h = image.naturalHeight;
		if (w !== size.w || h !== size.h) size = { w, h };
	}

	// redraw when the mask is set from outside (cleared, or another image's region)
	$effect(() => {
		const m = mask;
		if (!canvas || !size.w || m === lastMask) return;
		lastMask = m;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		if (!m) return;
		const img = new Image();
		img.onload = () => {
			// the mask is white-on-black: draw it, keep only white as the coloured region
			const tmp = document.createElement('canvas');
			tmp.width = canvas!.width;
			tmp.height = canvas!.height;
			const t = tmp.getContext('2d')!;
			t.drawImage(img, 0, 0, tmp.width, tmp.height);
			const data = t.getImageData(0, 0, tmp.width, tmp.height);
			for (let i = 0; i < data.data.length; i += 4) {
				const v = data.data[i];
				data.data[i] = 99;
				data.data[i + 1] = 102;
				data.data[i + 2] = 241;
				data.data[i + 3] = v;
			}
			ctx.putImageData(data, 0, 0);
		};
		img.src = m;
	});

	function point(e: PointerEvent) {
		const r = canvas!.getBoundingClientRect();
		return {
			x: ((e.clientX - r.left) / r.width) * canvas!.width,
			y: ((e.clientY - r.top) / r.height) * canvas!.height
		};
	}

	function dab(ctx: CanvasRenderingContext2D, x: number, y: number) {
		const radius = (brush / 512) * Math.max(canvas!.width, canvas!.height) * 0.5;
		ctx.globalCompositeOperation = tool === 'erase' ? 'destination-out' : 'source-over';
		ctx.fillStyle = FILL;
		ctx.beginPath();
		ctx.arc(x, y, radius, 0, Math.PI * 2);
		ctx.fill();
	}

	function down(e: PointerEvent) {
		if (!canvas || e.button !== 0) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		canvas.setPointerCapture(e.pointerId);
		drawing = true;
		start = point(e);
		if (tool === 'rect') snapshot = ctx.getImageData(0, 0, canvas.width, canvas.height);
		else dab(ctx, start.x, start.y);
	}

	function move(e: PointerEvent) {
		if (!drawing || !canvas) return;
		const ctx = canvas.getContext('2d')!;
		const p = point(e);
		if (tool === 'rect') {
			if (snapshot) ctx.putImageData(snapshot, 0, 0);
			ctx.globalCompositeOperation = 'source-over';
			ctx.fillStyle = FILL;
			ctx.fillRect(
				Math.min(start.x, p.x),
				Math.min(start.y, p.y),
				Math.abs(p.x - start.x),
				Math.abs(p.y - start.y)
			);
		} else {
			// fill the gap between pointer events
			const steps = Math.ceil(Math.hypot(p.x - start.x, p.y - start.y) / 4);
			for (let i = 1; i <= steps; i++)
				dab(ctx, start.x + ((p.x - start.x) * i) / steps, start.y + ((p.y - start.y) * i) / steps);
			start = p;
		}
	}

	function up() {
		if (!drawing) return;
		drawing = false;
		snapshot = null;
		emit();
	}

	/** The coloured region → white on black PNG (null if empty). */
	function emit() {
		if (!canvas) return;
		const ctx = canvas.getContext('2d')!;
		const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
		let any = false;
		for (let i = 3; i < data.length; i += 4)
			if (data[i] > 127) {
				any = true;
				break;
			}
		if (!any) {
			lastMask = null;
			onchange(null);
			return;
		}
		const out = document.createElement('canvas');
		out.width = canvas.width;
		out.height = canvas.height;
		const o = out.getContext('2d')!;
		o.drawImage(canvas, 0, 0);
		o.globalCompositeOperation = 'source-in';
		o.fillStyle = '#fff';
		o.fillRect(0, 0, out.width, out.height);
		o.globalCompositeOperation = 'destination-over';
		o.fillStyle = '#000';
		o.fillRect(0, 0, out.width, out.height);
		lastMask = out.toDataURL('image/png');
		onchange(lastMask);
	}

	function clear() {
		const ctx = canvas?.getContext('2d');
		if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
		lastMask = null;
		onchange(null);
	}

	const toolButton = (active: boolean) =>
		`rounded border px-2 py-0.5 text-xs ${active ? 'border-indigo-500 bg-indigo-600 text-white' : 'border-gray-300 text-gray-700 dark:border-gray-600 dark:text-gray-200'}`;
</script>

<div class="space-y-2">
	<div class="flex flex-wrap items-center gap-2 text-xs" role="toolbar" aria-label="Region tools">
		<span class="text-gray-500">Region:</span>
		<button type="button" class={toolButton(tool === 'rect')} onclick={() => (tool = 'rect')}
			>Rectangle</button
		>
		<button type="button" class={toolButton(tool === 'brush')} onclick={() => (tool = 'brush')}
			>Brush</button
		>
		<button type="button" class={toolButton(tool === 'erase')} onclick={() => (tool = 'erase')}
			>Eraser</button
		>
		{#if tool !== 'rect'}
			<label class="flex items-center gap-1 text-gray-500"
				>size <input type="range" min="8" max="160" bind:value={brush} class="w-20" /></label
			>
		{/if}
		<button
			type="button"
			class="text-indigo-600 hover:underline dark:text-indigo-400"
			onclick={clear}>Clear</button
		>
		<span class="text-gray-500"
			>{mask ? 'Only the marked region changes.' : 'Nothing marked: the whole image changes.'}</span
		>
	</div>
	<!-- the box is as wide as fits but no taller than the viewport allows; the image, the
	     overlay and the drawing layer all fill it, so their pixels line up -->
	<div
		class="relative mx-auto max-w-full select-none"
		style:width={size.w
			? `min(100%, ${((compact ? 60 : 70) * size.w) / size.h}vh, ${Math.max(size.w, 768)}px)`
			: 'fit-content'}
	>
		<img
			bind:this={image}
			{src}
			alt="Being explored"
			class="block h-auto w-full rounded"
			style:image-rendering={size.w && size.w < 256 ? 'pixelated' : 'auto'}
			onload={onLoad}
			draggable="false"
		/>
		{#if overlay}
			<img
				src={overlay}
				alt="Map overlay"
				class="pointer-events-none absolute inset-0 h-full w-full rounded opacity-80"
			/>
		{/if}
		{#if size.w}
			<canvas
				bind:this={canvas}
				width={size.w}
				height={size.h}
				class="absolute inset-0 h-full w-full cursor-crosshair rounded opacity-45"
				style:touch-action="none"
				aria-label="Mark the region to explore"
				onpointerdown={down}
				onpointermove={move}
				onpointerup={up}
				onpointercancel={up}
			></canvas>
		{/if}
	</div>
</div>
