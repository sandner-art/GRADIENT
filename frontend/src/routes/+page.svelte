<!-- frontend/src/routes/+page.svelte -->
<script lang="ts">
	import TextToImageForm from '$lib/components/TextToImageForm.svelte';
	import ImageToImageForm from '$lib/components/ImageToImageForm.svelte'; // <-- Import Img2Img Form
	import LLMChat from '$lib/components/LLMChat.svelte';
	import XYZGridForm from '$lib/components/XYZGridForm.svelte';
	import XYZGridDisplay from '$lib/components/XYZGridDisplay.svelte';
	import type { XYZGridResponse } from '$lib/types';

	// State for grid results and loading/error status (Unchanged)
	let latestGridResponse: XYZGridResponse | null = null;
	let isGridLoading: boolean = false;
	let gridError: string | null = null;

	// Handlers for grid (Unchanged)
	function handleGridSubmit(event: CustomEvent<XYZGridResponse>) { /* ... */ }
	function handleGridError(event: CustomEvent<string>) { /* ... */ }
	function handleGridLoading(event: CustomEvent<boolean>) { /* ... */ }
	function handleRerun(event: CustomEvent<Record<string, any>>) { /* ... */ }

    // --- Tab State ---
    let activeTab: 't2i' | 'i2i' | 'grid' = 't2i'; // Default tab
</script>

<svelte:head>
	<title>GRADIENT Workbench</title>
	<meta name="description" content="GRADIENT AI Workbench" />
</svelte:head>

<section class="text-center mb-8">
	<h1 class="text-3xl font-bold">Welcome to GRADIENT</h1>
	<p class="text-lg text-gray-600 dark:text-gray-400">Experimental AI Workbench</p>
</section>

<!-- Layout components -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
	<section class="space-y-6">

        <!-- Tab Navigation -->
        <div class="mb-4 border-b border-gray-200 dark:border-gray-700">
            <ul class="flex flex-wrap -mb-px text-sm font-medium text-center" role="tablist">
                <li class="mr-2" role="presentation">
                    <button
                        class="inline-block p-4 border-b-2 rounded-t-lg {activeTab === 't2i' ? 'border-indigo-500 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400' : 'border-transparent hover:text-gray-600 hover:border-gray-300 dark:hover:text-gray-300'}"
                        type="button" role="tab" on:click={() => activeTab = 't2i'}>Text-to-Image</button>
                </li>
                <li class="mr-2" role="presentation">
                    <button
                        class="inline-block p-4 border-b-2 rounded-t-lg {activeTab === 'i2i' ? 'border-indigo-500 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400' : 'border-transparent hover:text-gray-600 hover:border-gray-300 dark:hover:text-gray-300'}"
                        type="button" role="tab" on:click={() => activeTab = 'i2i'}>Image-to-Image</button>
                </li>
                <li class="mr-2" role="presentation">
                     <button
                         class="inline-block p-4 border-b-2 rounded-t-lg {activeTab === 'grid' ? 'border-indigo-500 text-indigo-600 dark:border-indigo-400 dark:text-indigo-400' : 'border-transparent hover:text-gray-600 hover:border-gray-300 dark:hover:text-gray-300'}"
                         type="button" role="tab" on:click={() => activeTab = 'grid'}>XYZ Grid</button>
                </li>
            </ul>
        </div>

        <!-- Tab Content -->
        <div>
            {#if activeTab === 't2i'}
                 <TextToImageForm />
            {:else if activeTab === 'i2i'}
                 <ImageToImageForm /> <!-- Display Img2Img form -->
            {:else if activeTab === 'grid'}
                 <XYZGridForm
                    on:submit={handleGridSubmit}
                    on:error={handleGridError}
                    on:loading={handleGridLoading}
                />
            {/if}
        </div>


		{#if activeTab === 'grid' && gridError && !isGridLoading}
			<!-- Grid Error Display -->
			<div
				class="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 dark:bg-red-900 dark:text-red-200 dark:border-red-700 rounded text-sm"
				role="alert"
			>
				<strong class="font-bold">Grid Error:</strong> {gridError}
			</div>
		{/if}
	</section>
	<section>
		<LLMChat />
	</section>
</div>

<!-- Grid Display Section (shows when data is available AND not loading) -->
{#if activeTab === 'grid'}
    {#if isGridLoading}
        <div class="mt-8 text-center">
            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500 mx-auto"></div>
            <p class="mt-3 text-gray-600 dark:text-gray-400">Generating XYZ Grid...</p>
        </div>
    {:else if latestGridResponse}
        <section class="mt-8">
            <XYZGridDisplay gridResponse={latestGridResponse} on:rerun={handleRerun} />
        </section>
    {/if}
{/if}