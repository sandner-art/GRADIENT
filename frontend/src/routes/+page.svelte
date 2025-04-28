<!-- frontend/src/routes/+page.svelte -->
<script lang="ts">
	import TextToImageForm from '$lib/components/TextToImageForm.svelte';
	import LLMChat from '$lib/components/LLMChat.svelte'; // <-- Import Chat component
</script>

<svelte:head>
	<title>GRADIENT Workbench</title>
	<meta name="description" content="GRADIENT AI Workbench" />
</svelte:head>

<section class="text-center mb-8">
	<h1 class="text-3xl font-bold">Welcome to GRADIENT</h1>
	<p class="text-lg text-gray-600 dark:text-gray-400">Experimental AI Workbench</p>
</section>

<!-- Layout components side-by-side or stacked -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <section>
        <TextToImageForm />
    </section>
    <section>
        <LLMChat />
    </section>
</div>