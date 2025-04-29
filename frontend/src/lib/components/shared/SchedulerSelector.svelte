<!-- frontend/src/lib/components/shared/SchedulerSelector.svelte -->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';

	export let selectedSchedulerId: string | null | undefined = null; // Value coming in (can be null for default)
	export let availableSchedulers: string[] = []; // List of options (including 'Default')
    export let label: string = "Scheduler:";
    export let id: string = "scheduler_selector";

	const dispatch = createEventDispatcher<{ change: string | null }>(); // Dispatch selected value (can be null)

    // Handler to dispatch the change event, mapping 'Default' to null
    function handleChange(event: Event) {
        const target = event.target as HTMLSelectElement;
        const value = target.value;
        // Check if the value corresponds to the placeholder for null ('__NULL_PLACEHOLDER__')
        const dispatchValue = value === '__NULL_PLACEHOLDER__' ? null : value;
        dispatch('change', dispatchValue);
    }

    // Use a placeholder value in the option for null, as select option value cannot be null directly
    const nullPlaceholder = '__NULL_PLACEHOLDER__';

</script>

<div>
	<label for={id} class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
		>{label}</label
	>
	<select
		{id}
        value={selectedSchedulerId === null ? nullPlaceholder : selectedSchedulerId} 
        on:change={handleChange}
		class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
	>
        <!-- Option representing null -->
		<option value={nullPlaceholder}>Default (Pipeline Default)</option>

        <!-- Other schedulers -->
		{#each availableSchedulers.filter(s => s !== 'Default') as scheduler (scheduler)}
			<option value={scheduler}>{scheduler}</option>
		{/each}
	</select>
</div>