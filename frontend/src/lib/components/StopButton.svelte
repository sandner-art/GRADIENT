<!-- frontend/src/lib/components/StopButton.svelte -->
<!-- Stop GRADIENT from the browser, like the first Ctrl+C in its window: a running job
     stops at its next step. Useful when Ctrl+C doesn't reach it (some IDE terminals). -->
<script lang="ts">
	import { post } from '#lib/apiClient.js';
	import { queue } from '#lib/stores/queue.svelte.js';
	import { notify } from '#lib/stores/ui.svelte.js';
	import { apiErrorMessage } from '#lib/utils/format.js';

	let confirming = $state(false);
	let stopping = $state(false);

	async function stop() {
		stopping = true;
		try {
			await post('/system/shutdown', {});
		} catch (e) {
			notify(apiErrorMessage(e), 'error');
			stopping = false;
		}
		confirming = false;
	}

	const busy = $derived(Object.values(queue.jobs).some((j) => j.status === 'running'));
</script>

{#if queue.connected && !stopping}
	{#if confirming}
		<span
			class="flex items-center gap-2 text-sm"
			role="group"
			aria-label="Confirm stopping GRADIENT"
		>
			<span class="text-gray-600 dark:text-gray-300"
				>Stop GRADIENT?{busy ? ' The running job stops at its next step.' : ''}</span
			>
			<button
				type="button"
				class="rounded bg-red-600 px-2.5 py-1 text-white hover:bg-red-700"
				onclick={stop}>Stop</button
			>
			<button
				type="button"
				class="rounded border border-gray-300 px-2.5 py-1 dark:border-gray-600"
				onclick={() => (confirming = false)}>Cancel</button
			>
		</span>
	{:else}
		<button
			type="button"
			class="rounded-md border border-gray-300 px-2.5 py-1.5 text-sm text-gray-600 hover:border-red-400 hover:text-red-700 dark:border-gray-600 dark:text-gray-300 dark:hover:text-red-400"
			title="Stop GRADIENT (like Ctrl+C in its window)"
			onclick={() => (confirming = true)}>⏻ Stop</button
		>
	{/if}
{/if}
