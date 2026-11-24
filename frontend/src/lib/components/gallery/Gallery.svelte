<!-- frontend/src/lib/components/gallery/Gallery.svelte -->
<!-- Everything GRADIENT has rendered: filter, browse, select, rate, tag, delete; or browse by run. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import ExportDialog from './ExportDialog.svelte';
	import ProjectsPanel from './ProjectsPanel.svelte';
	import RatingStars from './RatingStars.svelte';
	import {
		bulkAction,
		deleteRun,
		exportFilter,
		getFacets,
		listOutputs,
		listRuns,
		runTypeLabel,
		thumbUrl,
		updateOutput
	} from '#lib/gallery/api.js';
	import { gallery, patchItem, removeItems } from '#lib/gallery/state.svelte.js';
	import * as ui_ from '#lib/lab/ui.js';
	import { onJobChange } from '#lib/stores/queue.svelte.js';
	import { imageMenu } from '#lib/gallery/menus.js';
	import {
		goTo,
		notify,
		openMenu,
		openViewer,
		ui,
		type GalleryView
	} from '#lib/stores/ui.svelte.js';
	import type { GalleryFilter, RunItem } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	const PAGE = 60;
	let exporting = $state<{
		target: { ids: number[] } | { filter: Record<string, unknown> };
		count: number;
	} | null>(null);
	let size = $state(180);
	let search = $state(ui.galleryFilter.q ?? '');
	const selected = new SvelteSet<number>();
	let lastClicked = $state<number | null>(null);
	let bulkTags = $state('');
	let confirmBulkDelete = $state(false);
	let error = $state<string | null>(null);
	let runs = $state<RunItem[]>([]);
	let runsTotal = $state(0);
	let confirmRun = $state<string | null>(null);
	let sentinel = $state<HTMLElement | null>(null);

	const f = $derived(ui.galleryFilter);
	const ids = $derived(gallery.items.map((i) => i.id));
	const activeFilters = $derived(
		!!(f.run || f.model || f.run_type || f.q || f.min_rating || f.favorite || f.tag || f.project)
	);

	function setFilter(changes: Partial<GalleryFilter>) {
		ui.galleryFilter = { ...ui.galleryFilter, ...changes };
	}

	let loadToken = 0;
	async function load(reset: boolean, keepSelection = false) {
		const token = ++loadToken;
		const offset = reset ? 0 : (gallery.nextOffset ?? 0);
		if (!reset && gallery.nextOffset === null) return;
		gallery.loading = true;
		error = null;
		try {
			const page = await listOutputs(ui.galleryFilter, offset, PAGE);
			if (token !== loadToken) return;
			gallery.items = reset ? page.items : [...gallery.items, ...page.items];
			gallery.total = page.total;
			gallery.nextOffset = page.next_offset;
			if (reset && !keepSelection) selected.clear();
			else if (reset) {
				// keep the selection across a refresh, minus images that left the view
				const shown = new Set(gallery.items.map((it) => it.id));
				for (const id of [...selected]) if (!shown.has(id)) selected.delete(id);
			}
		} catch (e) {
			if (token === loadToken) error = apiErrorMessage(e);
		} finally {
			if (token === loadToken) gallery.loading = false;
		}
	}

	async function loadRuns(reset: boolean) {
		try {
			const page = await listRuns(reset ? 0 : runs.length, 30);
			runs = reset ? page.items : [...runs, ...page.items];
			runsTotal = page.total;
		} catch (e) {
			error = apiErrorMessage(e);
		}
	}

	async function refreshFacets() {
		gallery.facets = await getFacets().catch(() => gallery.facets);
	}

	// reload whenever the filter changes (or someone bumps galleryVersion)
	$effect(() => {
		JSON.stringify(ui.galleryFilter);
		void ui.galleryVersion;
		load(true);
	});

	// debounce the search box
	$effect(() => {
		const q = search.trim();
		const timer = setTimeout(() => {
			if ((ui.galleryFilter.q ?? '') !== q) setFilter({ q });
		}, 300);
		return () => clearTimeout(timer);
	});

	$effect(() => {
		if (ui.galleryView === 'runs') loadRuns(true);
	});

	// infinite scroll
	$effect(() => {
		if (!sentinel) return;
		const observer = new IntersectionObserver(
			(entries) => {
				if (
					entries.some((e) => e.isIntersecting) &&
					!gallery.loading &&
					gallery.nextOffset !== null
				)
					load(false);
			},
			{ rootMargin: '600px' }
		);
		observer.observe(sentinel);
		return () => observer.disconnect();
	});

	onMount(() => {
		refreshFacets();
		let timer: ReturnType<typeof setTimeout> | null = null;
		// new renders show up on their own (newest-first view)
		const off = onJobChange((job, previous) => {
			if (job.status !== 'completed' || previous?.status === 'completed') return;
			if (!job.result?.outputs?.length) return;
			if (timer) clearTimeout(timer);
			timer = setTimeout(() => {
				if ((ui.galleryFilter.sort ?? 'newest') === 'newest') load(true);
				if (ui.galleryView === 'runs') loadRuns(true);
				refreshFacets();
			}, 600);
		});
		return () => {
			off();
			if (timer) clearTimeout(timer);
		};
	});

	// ---- selection
	function tileClick(e: MouseEvent, id: number, index: number) {
		if (e.shiftKey && lastClicked !== null) {
			const from = ids.indexOf(lastClicked);
			const [a, b] = from < index ? [from, index] : [index, from];
			for (const x of ids.slice(a, b + 1)) selected.add(x);
		} else if (e.ctrlKey || e.metaKey || selected.size) {
			toggle(id);
		} else {
			openViewer(ids, index);
		}
		lastClicked = id;
	}
	function toggle(id: number) {
		if (selected.has(id)) selected.delete(id);
		else selected.add(id);
	}

	/** Right-click on a tile: the selection's menu if it is part of one, else the image's. */
	function tileMenu(e: MouseEvent, id: number, index: number) {
		const item = gallery.items[index];
		if (selected.size > 1 && selected.has(id)) {
			const n = selected.size;
			openMenu(
				e,
				[
					{ label: `Open ${n} images`, run: () => openViewer([...selected], 0) },
					{ separator: true },
					{ rating: null, rate: (stars) => bulk('rate', stars) },
					{ label: 'Add to favourites', run: () => bulk('favorite', true) },
					{ label: 'Remove from favourites', run: () => bulk('favorite', false) },
					{
						label: `Export ${n} to a project…`,
						run: () => (exporting = { target: { ids: [...selected] }, count: n })
					},
					{ label: 'Clear the selection', run: () => selected.clear() },
					{ separator: true },
					{
						label: `Delete ${n} images…`,
						danger: true,
						keepOpen: true,
						run: () => {
							if (ui.menu)
								ui.menu.items = ui.menu.items.map((it) =>
									'label' in it && it.label === `Delete ${n} images…`
										? {
												label: `Delete ${n}: click to confirm (moves to the trash folder)`,
												danger: true,
												run: () => bulk('delete')
											}
										: it
								);
						}
					}
				],
				`${n} selected`
			);
			return;
		}
		openMenu(
			e,
			imageMenu(id, {
				ids: gallery.items.map((it) => it.id),
				rating: item?.rating ?? null,
				favorite: item?.favorite ?? false,
				extra: [
					{
						label: selected.has(id) ? 'Deselect' : 'Select',
						run: () => {
							toggle(id);
							lastClicked = id;
						}
					}
				],
				more: [
					{
						label: 'Export to a project…',
						run: () => (exporting = { target: { ids: [id] }, count: 1 })
					}
				]
			}),
			item?.prompt ?? undefined
		);
	}

	async function bulk(action: Parameters<typeof bulkAction>[1], value?: unknown) {
		const list = [...selected];
		try {
			const r = await bulkAction(list, action, value);
			if (action === 'delete') {
				removeItems(list);
				selected.clear();
				notify(`${r.changed} image${r.changed === 1 ? '' : 's'} moved to the trash folder`);
			} else {
				await load(true, true);
				notify(
					`Updated ${r.changed} image${r.changed === 1 ? '' : 's'}`,
					'success',
					undefined,
					2000
				);
			}
			refreshFacets();
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		}
		confirmBulkDelete = false;
	}

	async function quickFavorite(id: number, favorite: boolean) {
		try {
			await updateOutput(id, { favorite });
			patchItem(id, { favorite });
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		}
	}

	async function quickRate(id: number, rating: number) {
		try {
			await updateOutput(id, { rating });
			patchItem(id, { rating: rating || null });
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		}
	}

	async function removeRun(run: RunItem) {
		if (confirmRun !== run.run_uuid) {
			confirmRun = run.run_uuid;
			return;
		}
		try {
			await deleteRun(run.run_uuid);
			runs = runs.filter((r) => r.run_uuid !== run.run_uuid);
			runsTotal -= 1;
			notify('Run moved to the trash folder');
			refreshFacets();
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		}
		confirmRun = null;
	}

	const fmtDate = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : '');
</script>

<div class="space-y-4">
	<!-- view switch and filters -->
	<div class="{ui_.card} space-y-3">
		<div class="flex flex-wrap items-center justify-between gap-2">
			<div class="inline-flex rounded-md shadow-sm" role="group" aria-label="Gallery view">
				{#each [['images', 'Images'], ['runs', 'Runs'], ['projects', 'Projects']] as [id, label], i (id)}
					<button
						type="button"
						class="border border-gray-300 px-3 py-1.5 text-sm font-medium dark:border-gray-600"
						class:rounded-l-md={i === 0}
						class:rounded-r-md={i === 2}
						class:bg-indigo-600={ui.galleryView === id}
						class:text-white={ui.galleryView === id}
						aria-pressed={ui.galleryView === id}
						onclick={() => (ui.galleryView = id as GalleryView)}>{label}</button
					>
				{/each}
			</div>
			<span class={ui_.muted}>
				{#if ui.galleryView === 'images'}{gallery.total} image{gallery.total === 1
						? ''
						: 's'}{#if gallery.facets && activeFilters}
						of {gallery.facets.total}{/if}{:else if ui.galleryView === 'runs'}{runsTotal} runs{/if}
			</span>
		</div>

		{#if ui.galleryView === 'images'}
			<div class="flex flex-wrap items-center gap-2">
				<input
					class={ui_.sized(ui_.smallInput, 'w-56')}
					type="search"
					placeholder="Search prompts and notes"
					aria-label="Search"
					bind:value={search}
				/>
				<select
					class={ui_.sized(ui_.smallInput, 'w-auto pr-8')}
					aria-label="Model"
					value={f.model ?? ''}
					onchange={(e) => setFilter({ model: e.currentTarget.value || null })}
				>
					<option value="">All models</option>
					{#each gallery.facets?.models ?? [] as m (m.model_id)}<option value={m.model_id}
							>{m.model_id || '(unknown)'} ({m.count})</option
						>{/each}
				</select>
				<select
					class={ui_.sized(ui_.smallInput, 'w-auto pr-8')}
					aria-label="Type"
					value={f.run_type ?? ''}
					onchange={(e) => setFilter({ run_type: e.currentTarget.value || null })}
				>
					<option value="">All types</option>
					<option value="text_to_image">Text-to-image</option>
					<option value="image_to_image">Image-to-image</option>
					<option value="xyz_grid">XYZ grids</option>
					<option value="lab_*">Lab results</option>
				</select>
				<select
					class={ui_.sized(ui_.smallInput, 'w-auto pr-8')}
					aria-label="Minimum rating"
					value={String(f.min_rating ?? '')}
					onchange={(e) =>
						setFilter({ min_rating: e.currentTarget.value ? Number(e.currentTarget.value) : null })}
				>
					<option value="">Any rating</option>
					{#each [1, 2, 3, 4, 5] as n (n)}<option value={String(n)}
							>{'★'.repeat(n)}{n < 5 ? '+' : ''}</option
						>{/each}
				</select>
				<select
					class={ui_.sized(ui_.smallInput, 'w-auto pr-8')}
					aria-label="Tag"
					value={f.tag ?? ''}
					onchange={(e) => setFilter({ tag: e.currentTarget.value || null })}
				>
					<option value="">Any tag</option>
					{#each gallery.facets?.tags ?? [] as t (t.tag)}<option value={t.tag}
							>{t.tag} ({t.count})</option
						>{/each}
				</select>
				{#if gallery.facets?.projects?.length}
					<select
						class={ui_.sized(ui_.smallInput, 'w-auto pr-8')}
						aria-label="Project"
						value={f.project ?? ''}
						onchange={(e) => setFilter({ project: e.currentTarget.value || null })}
					>
						<option value="">Any project</option>
						{#each gallery.facets.projects as pr (pr.uuid)}<option value={pr.uuid}
								>{pr.name} ({pr.count})</option
							>{/each}
					</select>
				{/if}
				<button
					type="button"
					class="rounded border px-2 py-1 text-sm"
					class:border-rose-400={f.favorite}
					class:text-rose-500={f.favorite}
					class:border-gray-300={!f.favorite}
					aria-pressed={!!f.favorite}
					onclick={() => setFilter({ favorite: f.favorite ? null : true })}>♥ Favourites</button
				>
				<select
					class={ui_.sized(ui_.smallInput, 'w-auto pr-8')}
					aria-label="Sort"
					value={f.sort ?? 'newest'}
					onchange={(e) => setFilter({ sort: e.currentTarget.value as GalleryFilter['sort'] })}
				>
					<option value="newest">Newest first</option>
					<option value="oldest">Oldest first</option>
					<option value="rating">Best rated</option>
				</select>
				<button
					type="button"
					class="{ui_.secondaryButton} ml-auto"
					disabled={!gallery.total}
					title="Copy the images this view shows into a project folder for Kurator, ArtAgents …"
					onclick={() =>
						(exporting = {
							target: { filter: exportFilter(ui.galleryFilter) },
							count: gallery.total
						})}>Export {gallery.total}…</button
				>
				<label class="flex items-center gap-2 text-xs text-gray-500">
					Size <input
						type="range"
						min="110"
						max="340"
						step="10"
						bind:value={size}
						aria-label="Thumbnail size"
					/>
				</label>
			</div>
			{#if activeFilters}
				<div class="flex flex-wrap items-center gap-2 text-xs">
					{#if f.run}
						<span
							class="rounded bg-indigo-100 px-2 py-0.5 text-indigo-900 dark:bg-indigo-900 dark:text-indigo-100"
						>
							run {f.run.slice(0, 8)}…
							<button
								type="button"
								class="ml-1"
								aria-label="Remove run filter"
								onclick={() => setFilter({ run: null })}>✕</button
							>
						</span>
					{/if}
					{#if f.project}
						<span
							class="rounded bg-emerald-100 px-2 py-0.5 text-emerald-900 dark:bg-emerald-900 dark:text-emerald-100"
						>
							project {gallery.facets?.projects?.find((p) => p.uuid === f.project)?.name ?? '…'}
							<button
								type="button"
								class="ml-1"
								aria-label="Remove project filter"
								onclick={() => setFilter({ project: null })}>✕</button
							>
						</span>
					{/if}
					<button
						type="button"
						class="text-indigo-600 hover:underline dark:text-indigo-400"
						onclick={() => {
							search = '';
							ui.galleryFilter = { sort: f.sort ?? 'newest' };
						}}>Clear filters</button
					>
				</div>
			{/if}
		{/if}
	</div>

	{#if error}<div class={ui_.errorBox} role="alert">{error}</div>{/if}

	{#if ui.galleryView === 'images'}
		{#if !gallery.items.length && !gallery.loading}
			<div class="{ui_.card} text-center">
				<p class={ui_.muted}>
					{activeFilters ? 'No images match these filters.' : 'Nothing rendered yet.'}
				</p>
				{#if !activeFilters}
					<button type="button" class="{ui_.primaryButton} mt-3" onclick={() => goTo('t2i')}
						>Render something</button
					>
				{/if}
			</div>
		{/if}
		<div class="grid gap-2" style:grid-template-columns="repeat(auto-fill, minmax({size}px, 1fr))">
			{#each gallery.items as item, i (item.id)}
				<div
					class="group relative overflow-hidden rounded-md bg-gray-100 dark:bg-gray-800"
					class:ring-4={selected.has(item.id)}
					class:ring-indigo-500={selected.has(item.id)}
					role="group"
					oncontextmenu={(e) => tileMenu(e, item.id, i)}
				>
					<button
						type="button"
						class="block w-full"
						title={item.prompt ?? ''}
						onclick={(e) => tileClick(e, item.id, i)}
					>
						<img
							src={thumbUrl(item.id, size > 250 ? 384 : 256)}
							alt={item.prompt ?? `Image ${item.id}`}
							loading="lazy"
							class="aspect-square w-full object-cover"
						/>
					</button>
					<label
						class="absolute top-1.5 left-1.5 rounded bg-black/40 p-1 opacity-0 transition group-hover:opacity-100"
						class:opacity-100={selected.size > 0}
					>
						<input
							type="checkbox"
							aria-label="Select image {item.id}"
							checked={selected.has(item.id)}
							onchange={() => {
								toggle(item.id);
								lastClicked = item.id;
							}}
						/>
					</label>
					<button
						type="button"
						class="absolute top-1 right-1.5 text-xl drop-shadow transition"
						class:text-rose-500={item.favorite}
						class:text-white={!item.favorite}
						class:opacity-0={!item.favorite}
						class:group-hover:opacity-100={!item.favorite}
						aria-label={item.favorite ? 'Remove from favourites' : 'Add to favourites'}
						onclick={() => quickFavorite(item.id, !item.favorite)}>♥</button
					>
					<div
						class="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-1.5 opacity-0 transition group-hover:opacity-100"
					>
						<div class="pointer-events-auto flex items-center justify-between text-white">
							<RatingStars size="sm" value={item.rating} onchange={(n) => quickRate(item.id, n)} />
							<span class="truncate pl-1 text-[10px] opacity-80">{item.width}×{item.height}</span>
						</div>
					</div>
					{#if item.rating && !selected.size}
						<span
							class="absolute bottom-1 left-1.5 text-xs text-amber-400 drop-shadow group-hover:opacity-0"
							>{'★'.repeat(item.rating)}</span
						>
					{/if}
				</div>
			{/each}
		</div>
		<div bind:this={sentinel} class="h-8 text-center text-sm text-gray-500">
			{#if gallery.loading}Loading…{:else if gallery.nextOffset !== null}<button
					type="button"
					class="hover:underline"
					onclick={() => load(false)}>Load more</button
				>{/if}
		</div>

		{#if selected.size}
			<div
				class="sticky bottom-3 z-20 mx-auto flex max-w-4xl flex-wrap items-center gap-2 rounded-lg border border-gray-200 bg-white p-3 text-sm shadow-lg dark:border-gray-700 dark:bg-gray-800"
			>
				<strong>{selected.size} selected</strong>
				<RatingStars value={null} onchange={(n) => bulk('rate', n)} />
				<button type="button" class={ui_.secondaryButton} onclick={() => bulk('favorite', true)}
					>♥ Favourite</button
				>
				<button type="button" class={ui_.secondaryButton} onclick={() => bulk('favorite', false)}
					>Unfavourite</button
				>
				<input
					class={ui_.sized(ui_.smallInput, 'w-40')}
					placeholder="tags, comma-separated"
					aria-label="Tags for the selected images"
					list="bulk-tag-suggestions"
					bind:value={bulkTags}
				/>
				<datalist id="bulk-tag-suggestions">
					{#each gallery.facets?.tags ?? [] as t (t.tag)}<option value={t.tag}></option>{/each}
				</datalist>
				<button
					type="button"
					class={ui_.secondaryButton}
					disabled={!bulkTags.trim()}
					onclick={() => {
						bulk('add_tags', bulkTags.split(','));
						bulkTags = '';
					}}>Add tags</button
				>
				<button
					type="button"
					class={ui_.secondaryButton}
					disabled={!bulkTags.trim()}
					onclick={() => {
						bulk('remove_tags', bulkTags.split(','));
						bulkTags = '';
					}}>Remove tags</button
				>
				<button
					type="button"
					class={ui_.secondaryButton}
					onclick={() => openViewer([...selected], 0)}>View</button
				>
				<button
					type="button"
					class={ui_.secondaryButton}
					onclick={() => (exporting = { target: { ids: [...selected] }, count: selected.size })}
					>Export…</button
				>
				<button
					type="button"
					class="rounded px-2 py-1 text-red-600 hover:bg-red-50 dark:hover:bg-red-950"
					class:bg-red-100={confirmBulkDelete}
					onclick={() => (confirmBulkDelete ? bulk('delete') : (confirmBulkDelete = true))}
					>{confirmBulkDelete ? `Delete ${selected.size}? Click again` : 'Delete'}</button
				>
				<span class="ml-auto flex gap-2">
					<button
						type="button"
						class="text-gray-500 hover:underline"
						onclick={() => ids.forEach((x) => selected.add(x))}>Select all loaded</button
					>
					<button
						type="button"
						class="text-gray-500 hover:underline"
						onclick={() => {
							selected.clear();
							confirmBulkDelete = false;
						}}>Clear</button
					>
				</span>
			</div>
		{/if}
	{:else if ui.galleryView === 'projects'}
		<ProjectsPanel
			onshow={(uuid) => {
				search = '';
				ui.galleryFilter = { sort: f.sort ?? 'newest', project: uuid };
				ui.galleryView = 'images';
			}}
			onchange={() => {
				refreshFacets();
				load(true, true);
			}}
		/>
	{:else}
		<div class="space-y-2">
			{#if !runs.length}<p class="{ui_.card} {ui_.muted}">No runs yet.</p>{/if}
			{#each runs as run (run.run_uuid)}
				<div class="{ui_.card} flex gap-4 p-3">
					<button
						type="button"
						class="grid h-24 w-24 flex-none grid-cols-2 gap-0.5 overflow-hidden rounded bg-gray-100 dark:bg-gray-700"
						aria-label="Show the images of this run"
						onclick={() => {
							ui.galleryView = 'images';
							ui.galleryFilter = { sort: 'oldest', run: run.run_uuid };
						}}
					>
						{#each run.cover_ids as id (id)}
							<img
								src={thumbUrl(id, 128)}
								alt=""
								class="h-full w-full object-cover"
								class:col-span-2={run.cover_ids.length === 1}
								class:row-span-2={run.cover_ids.length === 1}
								loading="lazy"
							/>
						{/each}
					</button>
					<div class="min-w-0 flex-1">
						<div class="flex flex-wrap items-center gap-2 text-xs">
							<span class="rounded bg-gray-100 px-1.5 py-0.5 dark:bg-gray-700"
								>{runTypeLabel(run.run_type)}</span
							>
							<span
								class:text-green-600={run.status === 'completed'}
								class:text-red-600={run.status === 'failed'}
								class:text-gray-500={run.status !== 'completed' && run.status !== 'failed'}
								>{run.status}</span
							>
							<span class="text-gray-500">{fmtDate(run.started)}</span>
							<span class="text-gray-500">{run.count} image{run.count === 1 ? '' : 's'}</span>
						</div>
						<p class="mt-1 truncate text-sm font-medium" title={run.title ?? run.prompt ?? ''}>
							{run.title ?? run.prompt ?? '—'}
						</p>
						<p class="truncate text-xs text-gray-500">{run.model_id ?? ''}</p>
						{#if run.error}<p class="mt-1 truncate text-xs text-red-600" title={run.error}>
								{run.error}
							</p>{/if}
					</div>
					<div class="flex flex-none flex-col items-end gap-1 text-sm">
						{#if run.count}
							<button
								type="button"
								class="text-indigo-600 hover:underline dark:text-indigo-400"
								onclick={() => {
									ui.galleryView = 'images';
									ui.galleryFilter = { sort: 'oldest', run: run.run_uuid };
								}}>Open</button
							>
						{/if}
						<button
							type="button"
							class="text-red-600 hover:underline"
							onclick={() => removeRun(run)}
							>{confirmRun === run.run_uuid ? 'Click again' : 'Delete'}</button
						>
					</div>
				</div>
			{/each}
			{#if runs.length < runsTotal}
				<button type="button" class="{ui_.secondaryButton} w-full" onclick={() => loadRuns(false)}
					>Load more runs</button
				>
			{/if}
		</div>
	{/if}
</div>

{#if exporting}
	<ExportDialog
		target={exporting.target}
		count={exporting.count}
		onclose={() => (exporting = null)}
		ondone={() => refreshFacets()}
	/>
{/if}
