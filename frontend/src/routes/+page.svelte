<!-- frontend/src/routes/+page.svelte -->
<script lang="ts">
	import TextToImageForm from '$lib/components/TextToImageForm.svelte';
	import LLMChat from '$lib/components/LLMChat.svelte';
	import XYZGridForm from '$lib/components/XYZGridForm.svelte'; // <-- Import Grid Form
	import XYZGridDisplay from '$lib/components/XYZGridDisplay.svelte';
    import type { XYZGridResponse } from '$lib/types';

    // State for grid results and loading/error status
    let latestGridResponse: XYZGridResponse | null = null;
    let isGridLoading: boolean = false;
    let gridError: string | null = null;

    // Handler for successful grid submission
    function handleGridSubmit(event: CustomEvent<XYZGridResponse>) {
        latestGridResponse = event.detail;
        gridError = null; // Clear previous error on success
        console.log("Received grid results in page:", latestGridResponse);
    }

    // Handler for grid errors
    function handleGridError(event: CustomEvent<string>) {
        gridError = event.detail;
        latestGridResponse = null; // Clear previous results on error
    }

     // Handler for grid loading state
    function handleGridLoading(event: CustomEvent<boolean>) {
        isGridLoading = event.detail;
         if (isGridLoading) {
            gridError = null; // Clear error when loading starts
            latestGridResponse = null; // Clear previous results when new grid starts
        }
    }

    // Handler for rerunning a cell from the display
    function handleRerun(event: CustomEvent<Record<string, any>>) {
        const params = event.detail;
        console.log("Rerun requested from grid:", params);
        // TODO: Populate the TextToImageForm with these params
        // This requires a way for this page component to interact with the state
        // *inside* TextToImageForm. Options:
        // 1. Lift state up from TextToImageForm to this page.
        // 2. Use a Svelte store for the T2I form data.
        // 3. Pass a "setParams" function down to TextToImageForm as a prop.
        // For now, just alert:
        alert(`Rerun T2I requested with params: ${JSON.stringify(params, null, 2)}. Implement form update.`);
    }

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
        <!-- Text to Image Form -->
        <div>
             <TextToImageForm />
        </div>
         <!-- XYZ Grid Form -->
         <div>
            <XYZGridForm
                on:submit={handleGridSubmit}
                on:error={handleGridError}
                on:loading={handleGridLoading}
            />
         </div>
         {#if gridError && !isGridLoading} <!-- Show grid error separately if needed -->
            <div class="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 dark:bg-red-900 dark:text-red-200 dark:border-red-700 rounded text-sm" role="alert">
                <strong class="font-bold">Grid Error:</strong> {gridError}
            </div>
         {/if}
    </section>
    <section>
        <LLMChat />
    </section>
</div>

<!-- Grid Display Section (shows when data is available AND not loading) -->
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