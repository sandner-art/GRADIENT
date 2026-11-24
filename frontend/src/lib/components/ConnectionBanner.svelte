<!-- frontend/src/lib/components/ConnectionBanner.svelte -->
<!-- Says when GRADIENT isn't reachable (stopped with Ctrl+C, closed or crashed) and when it
     came back as a newer version. The page reconnects by itself and keeps what you entered. -->
<script lang="ts">
	import { onDestroy } from 'svelte';
	import { queue } from '#lib/stores/queue.svelte.js';

	let { updatedTo = null, pageVersion = '' }: { updatedTo?: string | null; pageVersion?: string } =
		$props();

	const GRACE_MS = 2000; // a quick reconnect shouldn't flash a banner
	let now = $state(Date.now());
	const timer = setInterval(() => (now = Date.now()), 1000);
	onDestroy(() => clearInterval(timer));

	const offline = $derived(
		queue.everConnected &&
			!queue.connected &&
			queue.offlineSince !== null &&
			(queue.stopped || now - queue.offlineSince >= GRACE_MS)
	);
	const since = $derived(
		queue.offlineSince === null
			? ''
			: new Date(queue.offlineSince).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
	);
</script>

{#if offline}
	<div
		class="sticky top-0 z-40 mb-4 flex flex-wrap items-center gap-x-3 gap-y-1 rounded border border-red-300 bg-red-50 p-3 text-sm text-red-900 shadow-sm dark:border-red-800 dark:bg-red-950 dark:text-red-200"
		role="alert"
	>
		<span class="inline-block h-2.5 w-2.5 animate-pulse rounded-full bg-red-500" aria-hidden="true"
		></span>
		{#if queue.stopped}
			<b>GRADIENT was stopped</b>
			<span
				>at {since}. Start it again (run.bat); this page reconnects by itself and keeps what you
				entered.</span
			>
		{:else}
			<b>No connection to GRADIENT</b>
			<span
				>since {since}. It may have been closed or have stopped: check its window. Reconnecting…</span
			>
		{/if}
	</div>
{:else if updatedTo}
	<div
		class="mb-4 flex flex-wrap items-center gap-3 rounded border border-indigo-300 bg-indigo-50 p-3 text-sm text-indigo-900 dark:border-indigo-700 dark:bg-indigo-950 dark:text-indigo-200"
		role="status"
	>
		<span>
			GRADIENT was restarted as <b>v{updatedTo}</b>; this page is from v{pageVersion}. Reload it to
			use the new version.
		</span>
		<button
			type="button"
			class="rounded bg-indigo-600 px-3 py-1 text-white hover:bg-indigo-700"
			onclick={() => location.reload()}>Reload</button
		>
	</div>
{/if}
