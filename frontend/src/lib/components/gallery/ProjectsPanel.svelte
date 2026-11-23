<!-- frontend/src/lib/components/gallery/ProjectsPanel.svelte -->
<!-- Project folders shared with other tools: show their images, sync their work back, forget. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { forgetProject, listProjects, openProjectFolder, syncProject } from '#lib/gallery/api.js';
	import * as ui from '#lib/lab/ui.js';
	import { notify } from '#lib/stores/ui.svelte.js';
	import type { ProjectInfo, SyncReport } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	interface Props {
		/** show a project's images in the gallery */
		onshow: (uuid: string) => void;
		/** something changed (tags, ratings, captions) */
		onchange?: () => void;
	}
	let { onshow, onchange }: Props = $props();

	const LABEL_KEY = 'gradient.captionLabel';
	let projects = $state<ProjectInfo[]>([]);
	let loading = $state(true);
	let error = $state<string | null>(null);
	let busy = $state<string | null>(null);
	let reports = $state<Record<string, SyncReport>>({});
	let confirmForget = $state<string | null>(null);
	let openPath = $state('');
	let captionLabel = $state(readLabel());
	let captionsAsTags = $state(false);

	function readLabel(): string {
		try {
			return localStorage.getItem(LABEL_KEY) || 'artagents';
		} catch {
			return 'artagents';
		}
	}
	$effect(() => {
		try {
			localStorage.setItem(LABEL_KEY, captionLabel);
		} catch {
			/* private mode */
		}
	});

	async function load() {
		loading = true;
		try {
			projects = (await listProjects()).items;
			error = null;
		} catch (e) {
			error = apiErrorMessage(e);
		} finally {
			loading = false;
		}
	}
	onMount(load);

	function summary(r: SyncReport): string {
		const parts = [
			r.tags_added && `${r.tags_added} tag${r.tags_added === 1 ? '' : 's'} added`,
			r.tags_removed && `${r.tags_removed} removed`,
			r.ratings_changed && `${r.ratings_changed} rating${r.ratings_changed === 1 ? '' : 's'}`,
			r.captions && `${r.captions} caption${r.captions === 1 ? '' : 's'}`,
			r.renamed.length &&
				`${r.renamed.length} renamed file${r.renamed.length === 1 ? '' : 's'} found`
		].filter(Boolean);
		return parts.length ? parts.join(', ') : 'nothing new';
	}

	async function sync(p: ProjectInfo) {
		busy = p.uuid;
		try {
			const r = await syncProject(p.uuid, captionLabel.trim() || 'artagents', captionsAsTags);
			reports[p.uuid] = r;
			notify(`${p.name}: ${summary(r)}`, 'success');
			onchange?.();
			await load();
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		} finally {
			busy = null;
		}
	}

	async function forget(p: ProjectInfo) {
		try {
			await forgetProject(p.uuid);
			notify(`${p.name} removed from the list (the folder stays)`);
			onchange?.();
			await load();
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		}
		confirmForget = null;
	}

	async function addFolder() {
		busy = 'open';
		try {
			const r = await openProjectFolder(openPath.trim(), captionLabel.trim() || 'artagents');
			reports[r.project.uuid] = r;
			notify(`Added ${r.project.name}: ${summary(r)}`, 'success');
			openPath = '';
			onchange?.();
			await load();
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
		} finally {
			busy = null;
		}
	}

	async function copyPath(path: string) {
		try {
			await navigator.clipboard.writeText(path);
			notify('Folder path copied', 'success', undefined, 1500);
		} catch {
			notify('The browser blocked clipboard access', 'error');
		}
	}

	const fmt = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : '—');
</script>

<div class="space-y-3">
	<div class="{ui.card} space-y-3 text-sm">
		<p class={ui.muted}>
			A project folder holds copies of gallery images with XMP sidecars, ready for Kurator,
			ArtAgents, digiKam or Lightroom. Make one with <strong>Export</strong> in the Images view
			(selected images, or everything the current filters show). <strong>Sync</strong> brings back
			tags added in the sidecars, changed ratings, and caption files (<code>name.txt</code>).
		</p>
		<div class="flex flex-wrap items-end gap-3">
			<div>
				<label class={ui.label} for="caption_label">Caption label</label>
				<input
					id="caption_label"
					class={ui.sized(ui.smallInput, 'w-56')}
					bind:value={captionLabel}
					placeholder="artagents"
				/>
			</div>
			<label class="mb-1 flex items-center gap-2">
				<input type="checkbox" bind:checked={captionsAsTags} /> Also add comma-separated captions as tags
			</label>
		</div>
		<p class={ui.help}>
			Captions are stored under this label, one per label and image. Use one label per captioner
			(e.g. <code>artagents:llava</code>, then <code>artagents:qwen2.5vl</code>) to keep and compare
			several.
		</p>
	</div>

	{#if error}<div class={ui.errorBox} role="alert">{error}</div>{/if}
	{#if !loading && !projects.length}
		<p class="{ui.card} {ui.muted}">No projects yet.</p>
	{/if}

	{#each projects as p (p.uuid)}
		<div class="{ui.card} space-y-2">
			<div class="flex flex-wrap items-start justify-between gap-2">
				<div class="min-w-0">
					<h3 class="font-medium">{p.name}</h3>
					<div class="mt-1 flex items-center gap-2">
						<code class="truncate rounded bg-gray-100 px-1.5 py-0.5 text-xs dark:bg-gray-900"
							>{p.path}</code
						>
						<button
							type="button"
							class="text-xs text-indigo-600 hover:underline"
							onclick={() => copyPath(p.path)}>copy</button
						>
					</div>
					<p class="mt-1 text-xs text-gray-500">
						{p.count} image{p.count === 1 ? '' : 's'} · exported {fmt(p.updated)} · synced {fmt(
							p.last_sync
						)}
					</p>
					{#if !p.exists}
						<p class="mt-1 text-xs text-red-600">
							The folder is gone (moved or deleted). If you moved it, add it again below.
						</p>
					{/if}
				</div>
				<div class="flex flex-none items-center gap-3 text-sm">
					<button
						type="button"
						class="text-indigo-600 hover:underline dark:text-indigo-400"
						onclick={() => onshow(p.uuid)}>Show images</button
					>
					<button
						type="button"
						class={ui.secondaryButton}
						disabled={!p.exists || busy !== null}
						onclick={() => sync(p)}>{busy === p.uuid ? 'Syncing…' : 'Sync'}</button
					>
					<button
						type="button"
						class="text-red-600 hover:underline"
						title="Remove from this list; the folder and its files stay"
						onclick={() => (confirmForget === p.uuid ? forget(p) : (confirmForget = p.uuid))}
						>{confirmForget === p.uuid ? 'Forget? Click again' : 'Forget'}</button
					>
				</div>
			</div>
			{#if reports[p.uuid]}
				{@const r = reports[p.uuid]}
				<div class="rounded bg-gray-50 p-2 text-xs dark:bg-gray-900">
					<p>Last sync: {summary(r)}.</p>
					{#if r.missing.length}<p class="text-amber-700">
							{r.missing.length} file{r.missing.length === 1 ? '' : 's'} missing from the folder.
						</p>{/if}
					{#if r.unknown_count}<p class="text-gray-500">
							{r.unknown_count} other image file{r.unknown_count === 1 ? '' : 's'} in the folder (not
							from GRADIENT): {r.unknown_files.slice(0, 5).join(', ')}{r.unknown_count > 5
								? '…'
								: ''}
						</p>{/if}
					{#if r.unreadable_sidecars.length}<p class="text-red-600">
							Unreadable sidecars: {r.unreadable_sidecars.join(', ')}
						</p>{/if}
				</div>
			{/if}
		</div>
	{/each}

	<div class="{ui.card} space-y-2 text-sm">
		<label class={ui.label} for="open_project">Add an existing project folder</label>
		<div class="flex gap-2">
			<input
				id="open_project"
				class="{ui.input} font-mono text-sm"
				bind:value={openPath}
				placeholder="/path/to/project (has gradient-project.json)"
			/>
			<button
				type="button"
				class={ui.secondaryButton}
				disabled={!openPath.trim() || busy !== null}
				onclick={addFolder}>{busy === 'open' ? 'Adding…' : 'Add'}</button
			>
		</div>
		<p class={ui.help}>For a project folder you moved, or one you removed from the list.</p>
	</div>
</div>
