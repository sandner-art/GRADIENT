<!-- frontend/src/routes/+layout.svelte -->
<script lang="ts">
	import type { Snippet } from 'svelte';
	import '../app.css';

	let { children }: { children: Snippet } = $props();
</script>

<div class="min-h-screen bg-gray-100 text-gray-900 dark:bg-gray-900 dark:text-gray-100">
	<main class="container mx-auto p-4">
		{@render children()}
	</main>
</div>
