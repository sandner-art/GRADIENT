<!-- frontend/src/lib/components/shared/ModelSelector.svelte -->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let selectedModelId: string | null | undefined = ''; // Value coming in
	export let availableModels: string[] = []; // List of options
    export let label: string = "Model ID:"; // Customizable label
    export let id: string = "model_selector"; // Unique ID for label association

	const dispatch = createEventDispatcher<{ change: string }>(); // Dispatch selected value

    // Handler to dispatch the change event
    function handleChange(event: Event) {
        const target = event.target as HTMLSelectElement;
        dispatch('change', target.value);
    }
</script>

<div>
	<label for={id} class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
		>{label}</label
	>
	<select
		{id}
        value={selectedModelId} 
        on:change={handleChange} 
        required
		class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
	>
        <!-- Optional: Add a disabled default option if needed -->
        <!-- <option value="" disabled selected={!selectedModelId}>-- Select Model --</option> -->
		{#each availableModels as model (model)}
			<option value={model}>{model}</option>
		{/each}
        <!-- Consider adding an "Other..." option later to allow custom input -->
	</select>
	<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
		Select loaded/Hub model. Add models to config/search paths.
	</p>
</div>