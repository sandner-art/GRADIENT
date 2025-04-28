<!-- frontend/src/routes/+layout.svelte -->
<script lang="ts">
	import '../app.css';
</script>

<div class="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
    <!-- Optional Header -->
    <main class="container mx-auto p-4">
        <slot /> <!-- MAKE SURE THIS LINE IS PRESENT AND UNCOMMENTED -->
    </main>
    <!-- Optional Footer -->
</div>

<style lang="postcss">
    /* ... */
</style>