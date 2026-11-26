<!-- frontend/src/lib/components/render/MapsPane.svelte -->
<!-- PBR maps of an image: normal, height, AO, albedo, curvature, glow and, with a property
     network, roughness, metallic and the engine packs. A lit preview moves the light with the
     pointer; the set downloads as a ZIP that Unity, HDRP, glTF and Unreal can use. -->
<script lang="ts">
	import { imageUrl } from '#lib/gallery/api.js';
	import {
		lightFromPointer,
		makeMaps,
		mapFileUrl,
		mapOf,
		mapZipUrl,
		pbrModels,
		seamlessLabel,
		shade,
		type MapSet,
		type PbrModels,
		type Seamless
	} from '#lib/pbr.js';
	import { notify } from '#lib/stores/ui.svelte.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let { id }: { id: number | null } = $props();

	const PREVIEW_SIDE = 512;
	let models = $state<PbrModels | null>(null);
	let network = $state('auto'); // a model name, 'auto' or 'classical'
	let networkOnly = $state(false);
	let seamless = $state<Seamless>('auto');
	let detail = $state(0.35);
	let strength = $state(1);
	let busy = $state(false);
	let sets = $state<Record<number, MapSet>>({});
	let view = $state('lit'); // 'lit' or a map kind
	let litBase = $state<'albedo' | 'picture'>('albedo');
	let elevation = $state(0.5);

	const set = $derived(id === null ? null : (sets[id] ?? null));
	const shownMap = $derived(set && view !== 'lit' ? mapOf(set, view) : undefined);

	$effect(() => {
		pbrModels()
			.then((m) => (models = m))
			.catch(() => (models = { folder: '', models: [], default: null }));
	});

	async function run() {
		if (id === null) return;
		const target = id;
		busy = true;
		try {
			const classical = network === 'classical';
			const result = await makeMaps({
				output_id: target,
				model: classical ? 'auto' : network,
				method: classical ? 'classical' : networkOnly ? 'network' : 'hybrid',
				seamless,
				detail,
				strength
			});
			sets = { ...sets, [target]: result };
			view = 'lit';
		} catch (e) {
			notify(`Maps failed: ${apiErrorMessage(e)}`, 'error');
		} finally {
			busy = false;
		}
	}

	// ---- lit preview: pixels of the base colour, normal and AO at preview size
	let canvas = $state<HTMLCanvasElement | null>(null);
	let pixels = $state<{
		w: number;
		h: number;
		color: Uint8ClampedArray;
		normal: Uint8ClampedArray;
		ao: Uint8ClampedArray | null;
	} | null>(null);
	let light: [number, number, number] = [-0.4, 0.4, 0.82];

	function load(src: string): Promise<HTMLImageElement> {
		return new Promise((resolve, reject) => {
			const img = new Image();
			img.onload = () => resolve(img);
			img.onerror = () => reject(new Error(`Could not load ${src}`));
			img.src = src;
		});
	}

	function data(img: HTMLImageElement, w: number, h: number): Uint8ClampedArray {
		const c = document.createElement('canvas');
		c.width = w;
		c.height = h;
		const ctx = c.getContext('2d');
		if (!ctx) throw new Error('No canvas');
		ctx.drawImage(img, 0, 0, w, h);
		return ctx.getImageData(0, 0, w, h).data;
	}

	$effect(() => {
		pixels = null;
		const s = set;
		const base = litBase;
		if (!s || id === null) return;
		const normal = mapOf(s, 'normal');
		const albedo = mapOf(s, 'albedo');
		if (!normal || !albedo) return;
		const ao = mapOf(s, 'ao');
		const scale = Math.min(1, PREVIEW_SIDE / Math.max(s.width, s.height));
		const w = Math.max(1, Math.round(s.width * scale));
		const h = Math.max(1, Math.round(s.height * scale));
		let cancelled = false;
		Promise.all([
			load(base === 'albedo' ? mapFileUrl(s.set_id, albedo.file) : imageUrl(id)),
			load(mapFileUrl(s.set_id, normal.file)),
			ao ? load(mapFileUrl(s.set_id, ao.file)) : Promise.resolve(null)
		])
			.then(([c, n, a]) => {
				if (cancelled) return;
				pixels = {
					w,
					h,
					color: data(c, w, h),
					normal: data(n, w, h),
					ao: a ? data(a, w, h) : null
				};
			})
			.catch((e) => notify(apiErrorMessage(e), 'error'));
		return () => {
			cancelled = true;
		};
	});

	function draw() {
		const p = pixels;
		if (!canvas || !p) return;
		canvas.width = p.w;
		canvas.height = p.h;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const out = ctx.createImageData(p.w, p.h);
		// a picture already holds its light, so it gets no AO and less gain
		const fromPicture = litBase === 'picture';
		shade(
			p.color,
			p.normal,
			fromPicture ? null : p.ao,
			out.data,
			light,
			0.25,
			fromPicture ? 1 : 1.2
		);
		ctx.putImageData(out, 0, 0);
	}

	$effect(() => {
		void pixels;
		void canvas;
		draw();
	});

	function onPointer(e: PointerEvent) {
		if (!canvas || !pixels) return;
		if (e.type === 'pointermove' && e.buttons === 0 && e.pointerType !== 'mouse') return;
		const r = canvas.getBoundingClientRect();
		light = lightFromPointer(e.clientX - r.left, e.clientY - r.top, r.width, r.height, elevation);
		draw();
	}

	function downloadZip(setId: string) {
		const a = document.createElement('a');
		a.href = mapZipUrl(setId);
		a.download = '';
		a.click();
	}

	const fmtDeg = (v: number) => `${Math.round(v)}°`;
</script>

{#if id === null}
	<p class="p-6 text-center text-sm text-gray-500">Render or pick an image to make its maps.</p>
{:else}
	<div class="flex h-full w-full flex-col gap-2 text-xs">
		<div class="flex flex-wrap items-end gap-x-3 gap-y-2">
			<label class="flex flex-col gap-0.5"
				><span class="text-gray-500">Network</span>
				<select
					bind:value={network}
					class="rounded border-gray-300 py-0.5 text-xs dark:bg-gray-800"
				>
					<option value="auto">{models?.default ? `Auto (${models.default})` : 'Auto'}</option>
					{#each models?.models ?? [] as m (m.name)}
						<option value={m.name}>{m.name}</option>
					{/each}
					<option value="classical">Classical (no network)</option>
				</select></label
			>
			{#if network !== 'classical'}
				<label
					class="flex items-center gap-1"
					title="The network's normals without the picture's fine detail"
					><input type="checkbox" bind:checked={networkOnly} /> Network normals only</label
				>
			{/if}
			<label class="flex flex-col gap-0.5"
				><span class="text-gray-500">Seamless</span>
				<select
					bind:value={seamless}
					class="rounded border-gray-300 py-0.5 text-xs dark:bg-gray-800"
				>
					<option value="auto">As rendered</option>
					<option value="none">Off</option>
					<option value="x">Left–right</option>
					<option value="y">Top–bottom</option>
					<option value="xy">Both ways</option>
				</select></label
			>
			{#if network !== 'classical' && !networkOnly}
				<label
					class="flex flex-col gap-0.5"
					title="How much of the picture's fine detail goes into the normals"
					><span class="text-gray-500">Detail {detail.toFixed(2)}</span>
					<input type="range" min="0" max="1" step="0.05" bind:value={detail} class="w-24" /></label
				>
			{/if}
			<label class="flex flex-col gap-0.5"
				><span class="text-gray-500">Strength {strength.toFixed(2)}</span>
				<input
					type="range"
					min="0.25"
					max="4"
					step="0.25"
					bind:value={strength}
					class="w-24"
				/></label
			>
			<button
				type="button"
				class="rounded-md bg-indigo-600 px-3 py-1 text-sm text-white hover:bg-indigo-700 disabled:opacity-50"
				disabled={busy}
				onclick={run}>{busy ? 'Making maps…' : set ? 'Make again' : 'Make maps'}</button
			>
		</div>
		{#if models && !models.models.length && network !== 'classical'}
			<p class="text-gray-500">
				No PBR network installed, so maps use the classical method. For learned normals, albedo, AO,
				roughness and metallic, put <code>pbr_net_v3.pt</code> in <code>{models.folder}</code>.
			</p>
		{/if}

		{#if set}
			<div class="flex min-h-0 flex-1 gap-2">
				<div
					class="relative flex min-w-0 flex-1 items-center justify-center rounded bg-gray-100 dark:bg-gray-900"
				>
					{#if view === 'lit'}
						<canvas
							bind:this={canvas}
							class="h-full w-full cursor-crosshair touch-none object-contain"
							aria-label="The {litBase === 'albedo'
								? 'albedo'
								: 'picture'} lit by the normal map; move the pointer to move the light"
							onpointermove={onPointer}
							onpointerdown={onPointer}
						></canvas>
						<div
							class="absolute top-1 left-1 flex items-center gap-2 rounded bg-white/80 px-2 py-1 dark:bg-gray-800/80"
						>
							<div class="inline-flex rounded-md" role="group" aria-label="Lit preview base">
								{#each [['albedo', 'Albedo'], ['picture', 'Picture']] as [v, label], i (v)}
									<button
										type="button"
										class="border border-gray-300 px-2 py-0.5 dark:border-gray-600 {i === 0
											? 'rounded-l-md'
											: '-ml-px rounded-r-md'}"
										class:bg-indigo-600={litBase === v}
										class:text-white={litBase === v}
										aria-pressed={litBase === v}
										onclick={() => (litBase = v as 'albedo' | 'picture')}>{label}</button
									>
								{/each}
							</div>
							<label class="flex items-center gap-1" title="Height of the light above the surface"
								>Light <input
									type="range"
									min="0.1"
									max="1.5"
									step="0.05"
									bind:value={elevation}
									class="w-16"
								/></label
							>
						</div>
					{:else if shownMap}
						<img
							src={mapFileUrl(set.set_id, shownMap.file)}
							alt={shownMap.label}
							class="max-h-full max-w-full object-contain"
						/>
					{/if}
				</div>
				<div class="flex w-40 shrink-0 flex-col gap-1 overflow-y-auto">
					<button
						type="button"
						class="rounded border px-2 py-1 text-left {view === 'lit'
							? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-950'
							: 'border-gray-300 dark:border-gray-600'}"
						aria-pressed={view === 'lit'}
						onclick={() => (view = 'lit')}>Lit preview</button
					>
					{#each set.maps as m (m.kind)}
						<button
							type="button"
							class="flex items-center gap-2 rounded border p-1 text-left {view === m.kind
								? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-950'
								: 'border-gray-300 dark:border-gray-600'}"
							aria-pressed={view === m.kind}
							title={m.file}
							onclick={() => (view = m.kind)}
						>
							<img
								src={mapFileUrl(set.set_id, m.file)}
								alt=""
								class="h-8 w-8 shrink-0 rounded object-cover"
							/>
							<span class="leading-tight">{m.label}</span>
						</button>
					{/each}
				</div>
			</div>
			<div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-gray-500">
				<button
					type="button"
					class="rounded-md border border-gray-300 px-2 py-0.5 text-gray-700 hover:bg-gray-50 dark:border-gray-600 dark:text-gray-200 dark:hover:bg-gray-800"
					onclick={() => set && downloadZip(set.set_id)}>Download ZIP</button
				>
				<span>{set.model ? `${set.model}, ${set.method}` : 'classical'}</span>
				<span>{seamlessLabel(set.seamless)}</span>
				{#if set.light}
					<span title="The key light painted into the picture, as the network estimates it"
						>Painted light {fmtDeg(set.light.azimuth_deg)} azimuth, {fmtDeg(
							set.light.elevation_deg
						)} up</span
					>
				{/if}
				<span>{set.seconds.toFixed(1)} s</span>
			</div>
			{#each set.notes as note (note)}
				<p class="text-amber-700 dark:text-amber-400">{note}</p>
			{/each}
		{:else if !busy}
			<p class="p-4 text-center text-gray-500">
				Normal, height, AO, albedo, curvature and glow from this image; roughness, metallic and the
				Unity, HDRP and glTF packs with a property network. File names follow the Texture Map
				Generator, so the set imports into Unity as its own.
			</p>
		{/if}
	</div>
{/if}
