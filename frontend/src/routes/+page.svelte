<!-- frontend/src/routes/+page.svelte -->
<script lang="ts">
	// Core Svelte / SvelteKit Imports (none needed here currently)

	// Component Imports
	import TextToImageForm from '$lib/components/TextToImageForm.svelte';
	import ImageToImageForm from '$lib/components/ImageToImageForm.svelte';
	import LLMChat from '$lib/components/LLMChat.svelte'; // Assuming this component exists
	import XYZGridForm from '$lib/components/XYZGridForm.svelte';
	import XYZGridDisplay from '$lib/components/XYZGridDisplay.svelte';

	// Type Imports
	import type { XYZGridResponse } from '$lib/types';

	// Store Imports (for Rerun functionality)
	import { t2iFormStore } from '$lib/stores/t2iFormStore';
	// import { i2iFormStore } from '$lib/stores/i2iFormStore'; // Import if needed for I2I rerun later

	// --- State Management ---

	// Tab Control
	let activeTab: 't2i' | 'i2i' | 'grid' | 'chat' = 't2i'; // Default tab

	// Grid Specific State (Managed by events from XYZGridForm)
	let latestGridResponse: XYZGridResponse | null = null;
	let isGridLoading: boolean = false;
	let gridError: string | null = null;

	// --- Event Handlers for XYZ Grid ---
	function handleGridSubmit(event: CustomEvent<XYZGridResponse>) {
		console.log("Page received 'submit' event for grid:", event.detail); // Check if this logs
		latestGridResponse = event.detail; // Ensure this assignment happens
		gridError = null;
        // Don't automatically switch tab
		// console.log("Page updated latestGridResponse:", latestGridResponse);
	}

	function handleGridError(event: CustomEvent<string>) {
		console.error("Page received 'error' event for grid:", event.detail); // Check errors
		gridError = event.detail;
        latestGridResponse = null; // Clear results on error
	}

	function handleGridLoading(event: CustomEvent<boolean>) {
		console.log("Page received 'loading' event for grid:", event.detail); // Check loading state
		isGridLoading = event.detail;
		 if (isGridLoading) {
            gridError = null; // Clear error when loading starts
            latestGridResponse = null; // Clear previous results when new grid starts
             // Switch to grid tab automatically when loading starts
             activeTab = 'grid';
        }
	}

    // Handler for rerunning a cell from the grid display
    function handleRerun(event: CustomEvent<Record<string, any>>) {
        const params = event.detail;
        console.log("Rerun requested from grid display with params:", params);

        // Populate the TextToImageForm Store
        t2iFormStore.update(current => {
            const newState = { ...current };
            for (const key in params) {
                if (Object.prototype.hasOwnProperty.call(params, key)) {
                    if (key in newState) { (newState as any)[key] = params[key]; }
                    else { console.warn(`Param '${key}' from grid not in T2I store.`); }
                }
            }
            newState.seed = params.seed ?? null;
            newState.scheduler_id = params.scheduler_id ?? null;
            if(params.num_inference_steps !== undefined){ newState.custom_sigmas = null; }
            return newState;
        });

        // Switch to the Text-to-Image tab
        activeTab = 't2i';
        alert(`Parameters loaded into Text-to-Image tab.`);
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

</script>

<svelte:head>
	<title>GRADIENT Workbench</title>
	<meta name="description" content="GRADIENT AI Workbench" />
</svelte:head>

<!-- Page Header -->
<section class="text-center mb-8">
	<h1 class="text-3xl font-bold">Welcome to GRADIENT</h1>
	<p class="text-lg text-gray-600 dark:text-gray-400">Experimental AI Workbench</p>
</section>

<!-- Main Layout: Configuration on Left, Contextual Display on Right -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

    <!-- Left Column: Configuration Area -->
	<section class="space-y-6">

        <!-- Tab Navigation -->
        <div class="mb-4 border-b border-gray-200 dark:border-gray-700">
            <ul class="flex flex-wrap -mb-px text-sm font-medium text-center" role="tablist">
                <!-- T2I Tab Button -->
                <li class="mr-2" role="presentation">
                    <button
                        class="inline-block p-4 border-b-2 rounded-t-lg"
                        class:border-indigo-500={activeTab === 't2i'}
                        class:text-indigo-600={activeTab === 't2i'}
                        class:dark:border-indigo-400={activeTab === 't2i'}
                        class:dark:text-indigo-400={activeTab === 't2i'}
                        class:border-transparent={activeTab !== 't2i'}
                        class:hover:text-gray-600={activeTab !== 't2i'}
                        class:hover:border-gray-300={activeTab !== 't2i'}
                        class:dark:hover:text-gray-300={activeTab !== 't2i'}
                        type="button" role="tab" on:click={() => activeTab = 't2i'}>Text-to-Image</button>
                </li>
                <!-- I2I Tab Button -->
                <li class="mr-2" role="presentation">
                    <button
                         class="inline-block p-4 border-b-2 rounded-t-lg"
                         class:border-indigo-500={activeTab === 'i2i'}
                         class:text-indigo-600={activeTab === 'i2i'}
                         class:dark:border-indigo-400={activeTab === 'i2i'}
                         class:dark:text-indigo-400={activeTab === 'i2i'}
                         class:border-transparent={activeTab !== 'i2i'}
                         class:hover:text-gray-600={activeTab !== 'i2i'}
                         class:hover:border-gray-300={activeTab !== 'i2i'}
                         class:dark:hover:text-gray-300={activeTab !== 'i2i'}
                         type="button" role="tab" on:click={() => activeTab = 'i2i'}>Image-to-Image</button>
                </li>
                 <!-- Grid Tab Button -->
                <li class="mr-2" role="presentation">
                     <button
                         class="inline-block p-4 border-b-2 rounded-t-lg"
                         class:border-indigo-500={activeTab === 'grid'}
                         class:text-indigo-600={activeTab === 'grid'}
                         class:dark:border-indigo-400={activeTab === 'grid'}
                         class:dark:text-indigo-400={activeTab === 'grid'}
                         class:border-transparent={activeTab !== 'grid'}
                         class:hover:text-gray-600={activeTab !== 'grid'}
                         class:hover:border-gray-300={activeTab !== 'grid'}
                         class:dark:hover:text-gray-300={activeTab !== 'grid'}
                         type="button" role="tab" on:click={() => activeTab = 'grid'}>XYZ Grid</button>
                </li>
                 <!-- Chat Tab Button -->
                 <li class="mr-2" role="presentation">
                     <button
                         class="inline-block p-4 border-b-2 rounded-t-lg"
                         class:border-indigo-500={activeTab === 'chat'}
                         class:text-indigo-600={activeTab === 'chat'}
                         class:dark:border-indigo-400={activeTab === 'chat'}
                         class:dark:text-indigo-400={activeTab === 'chat'}
                         class:border-transparent={activeTab !== 'chat'}
                         class:hover:text-gray-600={activeTab !== 'chat'}
                         class:hover:border-gray-300={activeTab !== 'chat'}
                         class:dark:hover:text-gray-300={activeTab !== 'chat'}
                         type="button" role="tab" on:click={() => activeTab = 'chat'}>LLM Chat</button>
                </li>
            </ul>
        </div>

        <!-- Tab Content Area -->
        <div>
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
        </div>

	</section> <!-- End Left Column -->

    <!-- Right Column: Contextual Display Area -->
	<section>
        {#if activeTab === 'grid'}
            <!-- Grid Display / Loading / Error Area -->
            {#if isGridLoading}
                <!-- Loading Indicator -->
                <div class="sticky top-4 text-center p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
                    <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-teal-500 mx-auto"></div>
                    <p class="mt-3 text-gray-600 dark:text-gray-400">Generating XYZ Grid...</p>
                </div>
            {:else if latestGridResponse}
                 <!-- Grid Display Component -->
                 <!-- *** CORRECTED PROP PASSING *** -->
                 <XYZGridDisplay gridResponse={latestGridResponse} on:rerun={handleRerun} />
                 <!-- *** END CORRECTION *** -->
            {:else if gridError}
                 <!-- Grid Error Display when Grid tab is active -->
                 <div class="sticky top-4 mt-4 p-4 bg-red-100 border border-red-400 text-red-700 dark:bg-red-900 dark:text-red-200 dark:border-red-700 rounded text-sm" role="alert">
                    <strong class="font-bold">Grid Generation Error:</strong> {gridError}
                 </div>
            {:else}
                 <!-- Placeholder for Grid Tab when no results/loading/error -->
                 <div class="sticky top-4 text-center p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
                    <p class="text-gray-500 dark:text-gray-400">Configure and generate an XYZ grid using the form on the left.</p>
                 </div>
            {/if}

        {:else}
            <!-- Placeholder / Context Area for non-Grid tabs -->
             <div class="sticky top-4 text-center p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
                {#if activeTab === 't2i'}
                    <p class="text-gray-500 dark:text-gray-400">Text-to-Image results appear below the form on the left.</p>
                {:else if activeTab === 'i2i'}
                    <p class="text-gray-500 dark:text-gray-400">Image-to-Image results appear below the form on the left.</p>
                {:else if activeTab === 'chat'}
                     <p class="text-gray-500 dark:text-gray-400">LLM Chat Interface</p>
                {:else}
                    <p class="text-gray-500 dark:text-gray-400">Select a mode.</p>
                {/if}

                 <!-- Link/Info about last grid results/errors if they exist -->
                 {#if latestGridResponse && !isGridLoading} <!-- Removed redundant activeTab !== 'grid' check -->
                    <div class="mt-4 text-left border-t pt-4 dark:border-gray-700">
                        <p class="text-sm font-medium text-gray-600 dark:text-gray-300 mb-2">Last Grid Results Available:</p>
                        <button class="text-sm text-indigo-600 hover:underline dark:text-indigo-400" on:click={() => activeTab = 'grid'}>
                            View Full Grid
                        </button>
                    </div>
                  {/if}
                   {#if gridError && !isGridLoading}
                     <div class="mt-4 text-left border-t pt-4 dark:border-gray-700">
                         <p class="text-sm font-medium text-red-600 dark:text-red-400 mb-2">Last Grid Error:</p>
                         <p class="text-xs text-red-700 dark:text-red-300 break-words">{gridError}</p>
                         <button class="mt-1 text-sm text-indigo-600 hover:underline dark:text-indigo-400" on:click={() => activeTab = 'grid'}>
                             Go to Grid Form to Retry
                         </button>
                     </div>
                  {/if}
             </div>
        {/if}
	</section> <!-- End Right Column -->

</div> <!-- End Main Layout Grid -->