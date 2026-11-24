<!-- frontend/src/routes/+page.svelte -->
<script lang="ts">
	import { afterNavigate } from '$app/navigation';
	import { onDestroy, onMount } from 'svelte';

	import ContextMenu from '#lib/components/ContextMenu.svelte';
	import Toasts from '#lib/components/Toasts.svelte';
	import LLMChat from '#lib/components/LLMChat.svelte';
	import ModelManager from '#lib/components/ModelManager.svelte';
	import XYZGridDisplay from '#lib/components/XYZGridDisplay.svelte';
	import XYZGridForm from '#lib/components/XYZGridForm.svelte';
	import Gallery from '#lib/components/gallery/Gallery.svelte';
	import ImageViewer from '#lib/components/gallery/ImageViewer.svelte';
	import LabWorkbench from '#lib/components/lab/LabWorkbench.svelte';
	import QueuePanel from '#lib/components/queue/QueuePanel.svelte';
	import PreviewPanel from '#lib/components/render/PreviewPanel.svelte';
	import RenderForm from '#lib/components/render/RenderForm.svelte';

	import { get } from '#lib/apiClient.js';
	import { i2iFormStore } from '#lib/stores/i2iFormStore.js';
	import {
		connectQueue,
		disconnectQueue,
		onJobChange,
		outputIds
	} from '#lib/stores/queue.svelte.js';
	import { setupNavigation } from '#lib/stores/nav.svelte.js';
	import { t2iFormStore } from '#lib/stores/t2iFormStore.js';
	import { goTo, notify, openViewer, ui, type Tab } from '#lib/stores/ui.svelte.js';
	import type { XYZGridResponse } from '#lib/types.js';

	const tabs: { id: Tab; label: string }[] = [
		{ id: 't2i', label: 'Text-to-Image' },
		{ id: 'i2i', label: 'Image-to-Image' },
		{ id: 'gallery', label: 'Gallery' },
		{ id: 'grid', label: 'XYZ Grid' },
		{ id: 'lab', label: 'Lab' },
		{ id: 'chat', label: 'LLM Chat' },
		{ id: 'models', label: 'Models' }
	];

	// Back/Forward (mouse back button, Alt+Left) navigate inside the app; set up once the
	// router has started
	let stopNavigation: (() => void) | undefined;
	afterNavigate(() => {
		stopNavigation ??= setupNavigation();
	});
	onDestroy(() => stopNavigation?.());

	// Device / model availability shown in the header
	let deviceLabel = $state('');
	let noModels = $state(false);

	onMount(() => {
		connectQueue();
		(async () => {
			try {
				const [info, models] = await Promise.all([
					get<{ device: Record<string, unknown> }>('/system/info'),
					get<string[]>('/system/models/diffusion')
				]);
				const d = info.device;
				deviceLabel = [
					d.name ?? d.device,
					d.total_vram_gb ? `${d.total_vram_gb} GB VRAM` : null,
					`torch ${d.torch}`
				]
					.filter(Boolean)
					.join(' · ');
				noModels = models.length === 0;
				if (noModels) ui.tab = 'models';
			} catch {
				deviceLabel = 'API not reachable';
			}
		})();

		// tell the user about finished work they're not looking at
		const off = onJobChange((job, previous) => {
			if (!previous || previous.status === job.status) return;
			const origin: Record<string, Tab> = {
				text_to_image: 't2i',
				image_to_image: 'i2i',
				xyz_grid: 'grid'
			};
			if (job.status === 'failed') {
				notify(`Failed: ${job.label} — ${job.error ?? ''}`, 'error', undefined, 10000);
			} else if (
				job.status === 'completed' &&
				origin[job.kind] &&
				ui.tab !== origin[job.kind] &&
				ui.tab !== 'gallery'
			) {
				const ids = outputIds(job);
				if (ids.length)
					notify(`${ids.length} image${ids.length > 1 ? 's' : ''} ready: ${job.label}`, 'success', {
						label: 'View',
						run: () => openViewer(ids, 0)
					});
			}
		});
		return () => {
			off();
			disconnectQueue();
		};
	});

	// Grid state (managed by events from XYZGridForm)
	let latestGridResponse = $state<XYZGridResponse | null>(null);
	let isGridLoading = $state(false);
	let gridError = $state<string | null>(null);

	// Rerun a grid cell in the Text-to-Image tab
	function handleRerun(event: CustomEvent<Record<string, unknown>>) {
		const params = event.detail;
		t2iFormStore.update((current) => {
			const next = { ...current };
			for (const key of Object.keys(params)) {
				if (key in next) (next as Record<string, unknown>)[key] = params[key];
			}
			next.seed = (params.seed as number | null | undefined) ?? null;
			next.scheduler_id = (params.scheduler_id as string | null | undefined) ?? null;
			if (params.num_inference_steps !== undefined) next.custom_sigmas = null;
			return next;
		});
		goTo('t2i');
	}
</script>

<svelte:head>
	<title>GRADIENT Workbench</title>
	<meta name="description" content="GRADIENT AI Workbench" />
</svelte:head>

<section class="mb-6 flex flex-wrap items-end justify-between gap-2">
	<div>
		<h1 class="text-3xl font-bold">GRADIENT</h1>
		<p class="text-lg text-gray-600 dark:text-gray-400">Generative research workbench</p>
	</div>
	<div class="flex items-center gap-3">
		{#if deviceLabel}
			<p class="text-sm text-gray-500 dark:text-gray-400">{deviceLabel}</p>
		{/if}
		<QueuePanel />
	</div>
</section>

{#if noModels && ui.tab !== 'models'}
	<div
		class="mb-4 rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200"
	>
		No diffusion models installed yet.
		<button class="font-medium underline" onclick={() => (ui.tab = 'models')}
			>Open the Models tab</button
		> to download one.
	</div>
{/if}

<!-- Tabs -->
<div class="mb-6 border-b border-gray-200 dark:border-gray-700">
	<ul class="-mb-px flex flex-wrap text-center text-sm font-medium" role="tablist">
		{#each tabs as tab (tab.id)}
			<li class="mr-2" role="presentation">
				<button
					class="inline-block rounded-t-lg border-b-2 p-4"
					class:border-indigo-500={ui.tab === tab.id}
					class:text-indigo-600={ui.tab === tab.id}
					class:dark:border-indigo-400={ui.tab === tab.id}
					class:dark:text-indigo-400={ui.tab === tab.id}
					class:border-transparent={ui.tab !== tab.id}
					class:hover:text-gray-600={ui.tab !== tab.id}
					class:hover:border-gray-300={ui.tab !== tab.id}
					class:dark:hover:text-gray-300={ui.tab !== tab.id}
					type="button"
					role="tab"
					aria-selected={ui.tab === tab.id}
					onclick={() => (ui.tab = tab.id)}>{tab.label}</button
				>
			</li>
		{/each}
	</ul>
</div>

{#if ui.tab === 'models'}
	<ModelManager />
{:else if ui.tab === 'lab'}
	<LabWorkbench />
{:else if ui.tab === 'gallery'}
	<Gallery />
{:else if ui.tab === 't2i'}
	<div class="grid items-start gap-6 xl:grid-cols-[minmax(0,36rem)_minmax(0,1fr)]">
		<RenderForm mode="text" store={t2iFormStore} />
		<PreviewPanel form="text" />
	</div>
{:else if ui.tab === 'i2i'}
	<div class="grid items-start gap-6 xl:grid-cols-[minmax(0,36rem)_minmax(0,1fr)]">
		<RenderForm mode="image" store={i2iFormStore} />
		<PreviewPanel form="image" />
	</div>
{:else}
	<div class="grid grid-cols-1 gap-6 lg:grid-cols-2">
		<section class="space-y-6">
			{#if ui.tab === 'grid'}
				<XYZGridForm
					on:submit={(e: CustomEvent<XYZGridResponse>) => {
						latestGridResponse = e.detail;
						gridError = null;
					}}
					on:error={(e: CustomEvent<string>) => {
						gridError = e.detail;
						latestGridResponse = null;
					}}
					on:loading={(e: CustomEvent<boolean>) => {
						isGridLoading = e.detail;
						if (isGridLoading) {
							gridError = null;
							latestGridResponse = null;
						}
					}}
				/>
			{:else if ui.tab === 'chat'}
				<LLMChat />
			{/if}
		</section>

		<section>
			{#if ui.tab === 'grid'}
				{#if isGridLoading}
					<div
						class="sticky top-4 rounded-lg border border-gray-200 bg-white p-6 text-center shadow-md dark:border-gray-700 dark:bg-gray-800"
					>
						<div
							class="mx-auto h-12 w-12 animate-spin rounded-full border-b-2 border-teal-500"
						></div>
						<p class="mt-3 text-gray-600 dark:text-gray-400">
							Rendering the grid… progress and Stop are in the Queue panel.
						</p>
					</div>
				{:else if latestGridResponse}
					<XYZGridDisplay gridResponse={latestGridResponse} on:rerun={handleRerun} />
				{:else if gridError}
					<div
						class="sticky top-4 mt-4 rounded border border-red-400 bg-red-100 p-4 text-sm text-red-700 dark:border-red-700 dark:bg-red-900 dark:text-red-200"
						role="alert"
					>
						<strong class="font-bold">Grid generation error:</strong>
						{gridError}
					</div>
				{:else}
					<div
						class="sticky top-4 rounded-lg border border-gray-200 bg-white p-6 text-center shadow-md dark:border-gray-700 dark:bg-gray-800"
					>
						<p class="text-gray-500 dark:text-gray-400">
							Configure and generate an XYZ grid using the form on the left.
						</p>
					</div>
				{/if}
			{:else}
				<div
					class="sticky top-4 rounded-lg border border-gray-200 bg-white p-6 text-center shadow-md dark:border-gray-700 dark:bg-gray-800"
				>
					<p class="text-gray-500 dark:text-gray-400">LLM Chat Interface</p>
				</div>
			{/if}
		</section>
	</div>
{/if}

<ImageViewer />
<ContextMenu />
<Toasts />
