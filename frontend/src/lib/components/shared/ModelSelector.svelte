<!-- frontend/src/lib/components/shared/ModelSelector.svelte -->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';

    // --- Props ---
	/** The currently selected model identifier */
	export let selectedModelId: string | null | undefined = '';
	/** The list of available model identifiers */
	export let availableModels: string[] = [];
	/** The label text for the selector */
    export let label: string = "Model ID:";
	/** The HTML id attribute for the select element */
    export let id: string = "model_selector";
	/** Whether the selector should be disabled */
    export let disabled: boolean = false; // Added disabled prop

    // --- Event Dispatcher ---
	const dispatch = createEventDispatcher<{ change: string }>(); // Dispatch selected value string

    // --- Event Handler ---
    // Handler to dispatch the change event when the select value changes
    function handleChange(event: Event) {
        const target = event.target as HTMLSelectElement;
        dispatch('change', target.value); // Dispatch the selected string value
    }
</script>

<!-- Template -->
<div>
	<label for={id} class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
        {label}
    </label>
	<select
		{id}
        value={selectedModelId ?? ''} 
        on:change={handleChange}    
        required                    
        {disabled}                  
		class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50 disabled:cursor-not-allowed"
	>
        <!-- Optional: Add a disabled placeholder option if desired -->
        {#if !availableModels.includes(selectedModelId ?? '') && selectedModelId}
             <!-- Show selected value even if not in list initially, e.g. from store -->
            <option value={selectedModelId} selected disabled hidden>{selectedModelId} (loading...)</option>
        {/if}
        <option value="" disabled={!!selectedModelId} selected={!selectedModelId}>-- Select Model --</option>

        <!-- Iterate through available models -->
		{#each availableModels as model (model)}
			<option value={model}>{model}</option>
		{/each}
	</select>
	<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
		Installed models. Download more in the Models tab.
	</p>
</div>