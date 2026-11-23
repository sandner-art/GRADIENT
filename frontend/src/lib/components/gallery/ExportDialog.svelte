<!-- frontend/src/lib/components/gallery/ExportDialog.svelte -->
<!-- Export images (a selection or the current view) into a project folder for other tools. -->
<script lang="ts">
	import { onMount } from 'svelte';
	import { exportProject, listProjects } from '#lib/gallery/api.js';
	import * as ui from '#lib/lab/ui.js';
	import { notify } from '#lib/stores/ui.svelte.js';
	import type { ExportResult, ProjectInfo } from '#lib/types.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	interface Props {
		/** what to export: picked images, or every image of a gallery filter */
		target: { ids: number[] } | { filter: Record<string, unknown> };
		count: number;
		onclose: () => void;
		ondone?: (result: ExportResult) => void;
	}
	let { target, count, onclose, ondone }: Props = $props();

	let projects = $state<ProjectInfo[]>([]);
	let root = $state('');
	let mode = $state<'new' | 'existing'>('new');
	let existing = $state('');
	let name = $state('');
	let folder = $state('');
	let sidecars = $state(true);
	let modelKeywords = $state(true);
	let promptCaptions = $state(false);
	let link = $state(false);
	let busy = $state(false);
	let error = $state<string | null>(null);
	let result = $state<ExportResult | null>(null);

	const slug = (s: string) =>
		s
			.replace(/[^\p{L}\p{N}_\- ]+/gu, '')
			.trim()
			.replace(/ /g, '-')
			.slice(0, 60) || 'project';
	const sep = $derived(root.includes('\\') ? '\\' : '/');
	const defaultFolder = $derived(root ? `${root}${sep}${slug(name)}` : '');
	const canExport = $derived(
		!busy && (mode === 'new' ? name.trim().length > 0 : existing.length > 0)
	);

	onMount(async () => {
		try {
			const r = await listProjects();
			projects = r.items;
			root = r.root;
			if (projects.length) existing = projects[0].uuid;
		} catch (e) {
			error = apiErrorMessage(e);
		}
	});

	async function run() {
		busy = true;
		error = null;
		try {
			const where =
				mode === 'new' ? { name: name.trim(), path: folder.trim() || null } : { project: existing };
			result = await exportProject({
				...target,
				...where,
				sidecars,
				model_keywords: modelKeywords,
				captions: promptCaptions ? 'prompt' : 'none',
				link
			});
			ondone?.(result);
		} catch (e) {
			error = apiErrorMessage(e);
		} finally {
			busy = false;
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
</script>

<svelte:window
	onkeydown={(e) => {
		if (e.key === 'Escape') onclose();
	}}
/>

<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
	<div
		class="{ui.card} max-h-full w-full max-w-xl space-y-4 overflow-y-auto"
		role="dialog"
		aria-modal="true"
		aria-label="Export to a project folder"
	>
		<div class="flex items-start justify-between">
			<div>
				<h2 class={ui.heading}>Export to a project folder</h2>
				<p class={ui.muted}>{count} image{count === 1 ? '' : 's'}</p>
			</div>
			<button
				type="button"
				class="text-2xl leading-none text-gray-500"
				aria-label="Close"
				onclick={onclose}>×</button
			>
		</div>

		{#if result}
			<div class="space-y-2 text-sm">
				<p>
					<strong>{result.project.name}</strong>: {result.added} added{result.updated
						? `, ${result.updated} updated`
						: ''} · {result.project.count} images in the project.
				</p>
				<div class="flex items-center gap-2">
					<code
						class="min-w-0 flex-1 truncate rounded bg-gray-100 px-2 py-1 text-xs dark:bg-gray-900"
						>{result.project.path}</code
					>
					<button
						type="button"
						class={ui.secondaryButton}
						onclick={() => copyPath(result!.project.path)}>Copy path</button
					>
				</div>
				{#if result.sync && (result.sync.tags_added || result.sync.captions || result.sync.ratings_changed)}
					<p class={ui.help}>
						First read back from the folder: {result.sync.tags_added} tags, {result.sync.captions} captions,
						{result.sync.ratings_changed} ratings.
					</p>
				{/if}
				{#if result.kept_captions}
					<p class={ui.help}>
						{result.kept_captions} caption file{result.kept_captions === 1 ? '' : 's'} written by another
						tool were kept.
					</p>
				{/if}
				{#if result.kept_foreign_sidecars.length}
					<p class={ui.help}>
						{result.kept_foreign_sidecars.length} sidecars written by another program were left alone.
					</p>
				{/if}
				{#if result.missing_source.length || result.errors.length}
					<div class={ui.noteBox}>
						{#if result.missing_source.length}{result.missing_source.length} image files no longer exist.
						{/if}
						{#each result.errors as err, i (i)}<div>{err}</div>{/each}
					</div>
				{/if}
				<div class={ui.noteBox}>
					<p class="font-medium">Next</p>
					<ul class="mt-1 list-disc space-y-1 pl-5">
						<li>
							<strong>Kurator:</strong> add the folder (or a folder above it), then Settings → XMP
							sidecars → <em>Import keywords</em>. To send tags back, tick
							<em>Also update sidecars of other programs</em>
							and use <em>Write sidecars now</em>.
						</li>
						<li>
							<strong>ArtAgents:</strong> in the Captions tab, load this folder and generate
							captions (they are saved as <code>name.txt</code>).
						</li>
						<li>
							Then press <strong>Sync</strong> in Gallery → Projects to bring tags, ratings and captions
							back.
						</li>
					</ul>
				</div>
				<div class="flex justify-end">
					<button type="button" class={ui.primaryButton} onclick={onclose}>Done</button>
				</div>
			</div>
		{:else}
			<div class="flex gap-4 text-sm">
				<label class="flex items-center gap-1">
					<input type="radio" bind:group={mode} value="new" /> New project
				</label>
				<label class="flex items-center gap-1" class:opacity-50={!projects.length}>
					<input type="radio" bind:group={mode} value="existing" disabled={!projects.length} /> Add to
					a project
				</label>
			</div>

			{#if mode === 'new'}
				<div>
					<label class={ui.label} for="export_name">Name</label>
					<input
						id="export_name"
						class={ui.input}
						bind:value={name}
						placeholder="e.g. Dragons for review"
					/>
				</div>
				<div>
					<label class={ui.label} for="export_folder">Folder</label>
					<input
						id="export_folder"
						class="{ui.input} font-mono text-sm"
						bind:value={folder}
						placeholder={defaultFolder || 'default: data/projects/<name>'}
					/>
					<p class={ui.help}>
						Leave empty for {defaultFolder || 'data/projects/<name>'}. Use an empty or new folder; a
						folder Kurator already watches works well.
					</p>
				</div>
			{:else}
				<div>
					<label class={ui.label} for="export_existing">Project</label>
					<select id="export_existing" class={ui.input} bind:value={existing}>
						{#each projects as p (p.uuid)}
							<option value={p.uuid}>{p.name} ({p.count} images)</option>
						{/each}
					</select>
					<p class={ui.help}>
						Tags, ratings and captions from the folder are read back first, so nothing is lost.
					</p>
				</div>
			{/if}

			<fieldset class="space-y-1 text-sm">
				<legend class={ui.label}>Write</legend>
				<label class="flex items-start gap-2"
					><input type="checkbox" class="mt-1" bind:checked={sidecars} />
					<span
						>XMP sidecars (<code>name.png.xmp</code>): tags, rating and prompt for Kurator, digiKam,
						Lightroom, darktable</span
					></label
				>
				<label class="flex items-start gap-2" class:opacity-50={!sidecars}
					><input type="checkbox" class="mt-1" bind:checked={modelKeywords} disabled={!sidecars} />
					<span
						>Model and type keywords (<code>GRADIENT/Model/…</code>,
						<code>GRADIENT/Type/…</code>)</span
					></label
				>
				<label class="flex items-start gap-2"
					><input type="checkbox" class="mt-1" bind:checked={promptCaptions} />
					<span
						>The prompt as a caption file (<code>name.txt</code>, dataset style). Leave off when
						ArtAgents will caption the images; its captions are never overwritten.</span
					></label
				>
				<label class="flex items-start gap-2"
					><input type="checkbox" class="mt-1" bind:checked={link} />
					<span>Hard-link instead of copying (saves space; same drive only)</span></label
				>
			</fieldset>

			{#if error}<div class={ui.errorBox} role="alert">{error}</div>{/if}

			<div class="flex justify-end gap-2">
				<button type="button" class={ui.secondaryButton} onclick={onclose}>Cancel</button>
				<button type="button" class={ui.primaryButton} disabled={!canExport} onclick={run}
					>{busy ? 'Exporting…' : `Export ${count} image${count === 1 ? '' : 's'}`}</button
				>
			</div>
		{/if}
	</div>
</div>
