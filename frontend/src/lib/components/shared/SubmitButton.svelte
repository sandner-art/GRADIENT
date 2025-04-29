<!-- frontend/src/lib/components/shared/SubmitButton.svelte -->
<script lang="ts">
	// Props
	export let isLoading: boolean = false; // Receive loading state from parent
	export let disabled: boolean = false; // Allow parent to disable for other reasons (e.g., validation)
	export let buttonText: string = 'Generate'; // Default text
	export let loadingText: string = 'Generating...'; // Text during loading
</script>

<button
	type="submit"
	disabled={isLoading || disabled}  
	class="w-full inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 dark:focus:ring-offset-gray-800 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
>
	{#if isLoading}
		<!-- Spinner SVG -->
		<svg
			class="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
			xmlns="http://www.w3.org/2000/svg"
			fill="none"
			viewBox="0 0 24 24"
		>
			<circle
				class="opacity-25"
				cx="12"
				cy="12"
				r="10"
				stroke="currentColor"
				stroke-width="4"
			/>
			<path
				class="opacity-75"
				fill="currentColor"
				d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
			/>
		</svg>
		{loadingText}
	{:else}
		{buttonText}
	{/if}
</button>