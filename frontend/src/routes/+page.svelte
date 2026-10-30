<!-- frontend/src/routes/+page.svelte -->
<script lang="ts">
	import { onMount } from 'svelte';

	import TextToImageForm from '#lib/components/TextToImageForm.svelte';
	import ImageToImageForm from '#lib/components/ImageToImageForm.svelte';
	import LLMChat from '#lib/components/LLMChat.svelte';
	import XYZGridForm from '#lib/components/XYZGridForm.svelte';
	import XYZGridDisplay from '#lib/components/XYZGridDisplay.svelte';
	import ModelManager from '#lib/components/ModelManager.svelte';

	import { get } from '#lib/apiClient.js';
	import type { XYZGridResponse } from '#lib/types.js';
	import { t2iFormStore } from '#lib/stores/t2iFormStore.js';

	type Tab = 't2i' | 'i2i' | 'grid' | 'chat' | 'models';
	const tabs: { id: Tab; label: string }[] = [
		{ id: 't2i', label: 'Text-to-Image' },
		{ id: 'i2i', label: 'Image-to-Image' },
		{ id: 'grid', label: 'XYZ Grid' },
		{ id: 'chat', label: 'LLM Chat' },
		{ id: 'models', label: 'Models' }
	];

	let activeTab: Tab = 't2i';

	// Device / model availability shown in the header
	let deviceLabel = '';
	let noModels = false;
	onMount(async () => {
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
			if (noModels) activeTab = 'models';
		} catch {
			deviceLabel = 'API not reachable';
		}
	});

	// Grid state (managed by events from XYZGridForm)
	let latestGridResponse: XYZGridResponse | null = null;
	let isGridLoading = false;
	let gridError: string | null = null;

	function handleGridSubmit(event: CustomEvent<XYZGridResponse>) {
		latestGridResponse = event.detail;
		gridError = null;
	}

	function handleGridError(event: CustomEvent<string>) {
		gridError = event.detail;
		latestGridResponse = null;
	}

	function handleGridLoading(event: CustomEvent<boolean>) {
		isGridLoading = event.detail;
		if (isGridLoading) {
			gridError = null;
			latestGridResponse = null;
			activeTab = 'grid';
		}
	}

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
		activeTab = 't2i';
		window.scrollTo({ top: 0, behavior: 'smooth' });
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
	{#if deviceLabel}
		<p class="text-sm text-gray-500 dark:text-gray-400">{deviceLabel}</p>
	{/if}
</section>

{#if noModels && activeTab !== 'models'}
	<div
		class="mb-4 rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-200"
	>
		No diffusion models installed yet.
		<button class="font-medium underline" on:click={() => (activeTab = 'models')}
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
					class:border-indigo-500={activeTab === tab.id}
					class:text-indigo-600={activeTab === tab.id}
					class:dark:border-indigo-400={activeTab === tab.id}
					class:dark:text-indigo-400={activeTab === tab.id}
					class:border-transparent={activeTab !== tab.id}
					class:hover:text-gray-600={activeTab !== tab.id}
					class:hover:border-gray-300={activeTab !== tab.id}
					class:dark:hover:text-gray-300={activeTab !== tab.id}
					type="button"
					role="tab"
					aria-selected={activeTab === tab.id}
					on:click={() => (activeTab = tab.id)}>{tab.label}</button
				>
			</li>
		{/each}
	</ul>
</div>

{#if activeTab === 'models'}
	<ModelManager />
{:else}
	<div class="grid grid-cols-1 gap-6 lg:grid-cols-2">
		<!-- Left: configuration -->
		<section class="space-y-6">
			{#if activeTab === 't2i'}
				<TextToImageForm />
			{:else if activeTab === 'i2i'}
				<ImageToImageForm />
			{:else if activeTab === 'grid'}
				<XYZGridForm
					on:submit={handleGridSubmit}
					on:error={handleGridError}
					on:loading={handleGridLoading}
				/>
			{:else if activeTab === 'chat'}
				<LLMChat />
			{/if}
		</section>

		<!-- Right: contextual display -->
		<section>
			{#if activeTab === 'grid'}
				{#if isGridLoading}
					<div
						class="sticky top-4 rounded-lg border border-gray-200 bg-white p-6 text-center shadow-md dark:border-gray-700 dark:bg-gray-800"
					>
						<div
							class="mx-auto h-12 w-12 animate-spin rounded-full border-b-2 border-teal-500"
						></div>
						<p class="mt-3 text-gray-600 dark:text-gray-400">Generating XYZ Grid...</p>
					</div>
				{:else if latestGridResponse}
					<XYZGridDisplay gridResponse={latestGridResponse} on:rerun={handleRerun} />
				{:else if gridError}
					<div
						class="sticky top-4 mt-4 rounded border border-red-400 bg-red-100 p-4 text-sm text-red-700 dark:border-red-700 dark:bg-red-900 dark:text-red-200"
						role="alert"
					>
						<strong class="font-bold">Grid Generation Error:</strong>
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
					{#if activeTab === 't2i'}
						<p class="text-gray-500 dark:text-gray-400">
							Text-to-Image results appear below the form on the left.
						</p>
					{:else if activeTab === 'i2i'}
						<p class="text-gray-500 dark:text-gray-400">
							Image-to-Image results appear below the form on the left.
						</p>
					{:else if activeTab === 'chat'}
						<p class="text-gray-500 dark:text-gray-400">LLM Chat Interface</p>
					{/if}

					{#if latestGridResponse && !isGridLoading}
						<div class="mt-4 border-t pt-4 text-left dark:border-gray-700">
							<p class="mb-2 text-sm font-medium text-gray-600 dark:text-gray-300">
								Last Grid Results Available:
							</p>
							<button
								class="text-sm text-indigo-600 hover:underline dark:text-indigo-400"
								on:click={() => (activeTab = 'grid')}>View Full Grid</button
							>
						</div>
					{/if}
					{#if gridError && !isGridLoading}
						<div class="mt-4 border-t pt-4 text-left dark:border-gray-700">
							<p class="mb-2 text-sm font-medium text-red-600 dark:text-red-400">
								Last Grid Error:
							</p>
							<p class="text-xs break-words text-red-700 dark:text-red-300">{gridError}</p>
							<button
								class="mt-1 text-sm text-indigo-600 hover:underline dark:text-indigo-400"
								on:click={() => (activeTab = 'grid')}
							>
								Go to Grid Form to Retry
							</button>
						</div>
					{/if}
				</div>
			{/if}
		</section>
	</div>
{/if}
