<!-- frontend/src/lib/components/shared/SeedInput.svelte -->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	// Props
	export let id: string = 'seed_input'; // Allow parent to set ID for label association
	export let label: string = 'Seed:';
	export let seedValue: number | null | undefined = null; // The bound seed value from the parent/store

	const dispatch = createEventDispatcher<{
		change: number | null; // Dispatch the new value when input changes
		randomize: void; // Dispatch event when random button clicked
		clear: void; // Dispatch event when clear button clicked
	}>();

	// Handle direct input changes in the number field
	function handleInputChange(event: Event) {
		const target = event.target as HTMLInputElement;
		const value = target.value;
		// Dispatch null if empty, otherwise dispatch the number
		dispatch('change', value === '' ? null : Number(value));
	}

	// Handle button clicks by dispatching specific events
	function handleRandomizeClick() {
		dispatch('randomize');
	}

	function handleClearClick() {
		dispatch('clear');
	}
</script>

<div>
	<label for={id} class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
		>{label}</label
	>
	<div class="flex items-center space-x-2">
		<input
			type="number"
			{id}
			placeholder="Random (null)"
            value={seedValue ?? ''} 
            on:input={handleInputChange} 
			min="0"
            max="4294967295" 
            step="1"
			class="flex-grow px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
		/>
		<button
			type="button"
            on:click={handleRandomizeClick}
            title="Generate Random Seed"
			class="p-2 bg-gray-200 dark:bg-gray-600 rounded hover:bg-gray-300 dark:hover:bg-gray-500 text-lg transition-colors duration-150"
			>🎲</button
		>
		<button
			type="button"
            on:click={handleClearClick}
            title="Use Random Seed (null)"
			class="p-2 bg-gray-200 dark:bg-gray-600 rounded hover:bg-gray-300 dark:hover:bg-gray-500 text-lg transition-colors duration-150"
			disabled={seedValue === null || seedValue === undefined}  
            >❌</button
		>
	</div>
</div>