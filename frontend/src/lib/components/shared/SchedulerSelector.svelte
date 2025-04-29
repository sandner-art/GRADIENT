<!-- frontend/src/lib/components/shared/SchedulerSelector.svelte -->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';

    // --- Props ---
	/** The currently selected scheduler ID (null represents default) */
	export let selectedSchedulerId: string | null | undefined = null;
	/** List of available scheduler names (should include 'Default') */
	export let availableSchedulers: string[] = ['Default'];
	/** The label text for the selector */
    export let label: string = "Scheduler:";
	/** The HTML id attribute for the select element */
    export let id: string = "scheduler_selector";
	/** Whether the selector should be disabled */
    export let disabled: boolean = false; // Added disabled prop

    // --- Event Dispatcher ---
	const dispatch = createEventDispatcher<{ change: string | null }>(); // Can dispatch null

    // --- Constants ---
    // Use a placeholder value in the <option> for null, as option value cannot be truly null
    const nullPlaceholder = '__NULL_PLACEHOLDER__';

    // --- Event Handler ---
    function handleChange(event: Event) {
        const target = event.target as HTMLSelectElement;
        const value = target.value;
        // Map the placeholder back to null when dispatching
        const dispatchValue = value === nullPlaceholder ? null : value;
        dispatch('change', dispatchValue);
    }
</script>

<!-- Template -->
<div>
	<label for={id} class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
        {label}
    </label>
	<select
		{id}
        value={selectedSchedulerId === null || selectedSchedulerId === undefined ? nullPlaceholder : selectedSchedulerId} 
        on:change={handleChange}
        {disabled}                  
		class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50 disabled:cursor-not-allowed"
	>
        <!-- Option representing null (default) -->
		<option value={nullPlaceholder}>Default (Pipeline Default)</option>

        <!-- Iterate through actual scheduler names -->
		{#each availableSchedulers.filter(s => s !== 'Default') as scheduler (scheduler)}
			<option value={scheduler}>{scheduler}</option>
		{/each}
	</select>
</div>