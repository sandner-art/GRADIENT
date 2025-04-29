<!-- frontend/src/lib/components/TextToImageForm.svelte -->
<script lang="ts">
	// --- Imports ---
	import { post, isApiError, type ApiError } from '$lib/apiClient';
	import type { Text2ImageRequest, Text2ImageResponse, ImageResult } from '$lib/types';
	import { t2iFormStore } from '$lib/stores/t2iFormStore'; // Import the store
	import { onMount } from 'svelte'; // Import onMount

	// --- Component State (Managed Externally by Store) ---
	// Local state only for UI elements not directly part of the request payload
	// or for intermediate values like the sigma string.
	let isLoading = false;
	let error: string | null = null;
	let results: ImageResult[] | null = null;
	let customSigmasString: string = ''; // Input string for custom sigmas

	// --- Placeholder Data for Selectors ---
	// TODO: Fetch these from backend API later
	let availableModels: string[] = [
		'photomatix_v3.fp16.safetensors',
		'realisticVisionV51_v51VAE.safetensors',
        'emilianJR/epiCRealism', // Example common checkpoint name
		'sd_xl_base_1.0.safetensors', // Example XL checkpoint
		'/path/to/absolute/model.ckpt' // Example absolute
	];
	let availableSchedulers: string[] = [
		'Default', // Represents null/pipeline default
		'DDIMScheduler',
		'DPMSolverMultistepScheduler', // DPM++ 2M Karras
		'EulerAncestralDiscreteScheduler', // Euler A
		'EulerDiscreteScheduler', // Euler
		'UniPCMultistepScheduler'
	];

	// --- Reactive Parsing for Custom Sigmas (Updates the store) ---
	$: {
		if (customSigmasString && customSigmasString.trim() !== '') {
			try {
				const parsedSigmas = customSigmasString
					.split(/[\s,]+/)
					.filter((s) => s.trim() !== '')
					.map((s) => parseFloat(s));

				if (parsedSigmas.every((n) => !isNaN(n))) {
                    // Update store only if value changed to avoid infinite loop if typing non-numbers
                    if (JSON.stringify($t2iFormStore.custom_sigmas) !== JSON.stringify(parsedSigmas)) {
					    t2iFormStore.update((current) => ({ ...current, custom_sigmas: parsedSigmas }));
                    }
				} else {
                    if ($t2iFormStore.custom_sigmas !== null) {
					    t2iFormStore.update((current) => ({ ...current, custom_sigmas: null }));
                    }
					console.warn('Invalid sigma input detected - contains non-numbers.');
				}
			} catch (e) {
                 if ($t2iFormStore.custom_sigmas !== null) {
				    t2iFormStore.update((current) => ({ ...current, custom_sigmas: null }));
                 }
				console.error('Error parsing custom sigmas string:', e);
			}
		} else {
             if ($t2iFormStore.custom_sigmas !== null) {
			    t2iFormStore.update((current) => ({ ...current, custom_sigmas: null }));
             }
		}
	}

	// --- Initialize local customSigmasString from store on mount ---
	onMount(() => {
		const currentSigmas = $t2iFormStore.custom_sigmas;
		if (currentSigmas) {
			customSigmasString = currentSigmas.join(', ');
		} else {
			customSigmasString = '';
		}
        // Ensure scheduler dropdown reflects store's null value correctly
        if ($t2iFormStore.scheduler_id === null) {
             // If using a select element directly bound, Svelte handles this.
             // If building custom select, might need manual sync here.
        }
	});


	// --- Submit Handler ---
	async function generateImage() {
		isLoading = true;
		error = null;
		results = null;

		// --- Get current state FROM THE STORE ---
		const currentRequestData = $t2iFormStore; // Use $ for reactive access

		// Create payload from store data, ensuring correct types and trimming strings
		const payload: Text2ImageRequest = {
			model_id: currentRequestData.model_id?.trim() || '',
			prompt: currentRequestData.prompt?.trim() || '',
			negative_prompt: currentRequestData.negative_prompt?.trim() || '',
			height: Number(currentRequestData.height) || 512,
			width: Number(currentRequestData.width) || 512,
			num_inference_steps: Number(currentRequestData.num_inference_steps) || 25,
			guidance_scale: Number(currentRequestData.guidance_scale) || 7.0,
			num_outputs: Number(currentRequestData.num_outputs) || 1,
			seed: currentRequestData.seed === null || isNaN(Number(currentRequestData.seed)) ? null : Math.floor(Number(currentRequestData.seed)),
			scheduler_id: currentRequestData.scheduler_id === 'Default' || !currentRequestData.scheduler_id ? null : currentRequestData.scheduler_id.trim(), // Map 'Default' or empty to null
			custom_sigmas: currentRequestData.custom_sigmas // Already updated reactively
		};

		// --- Basic Client-Side Validation ---
		if (!payload.model_id) { error = 'Model ID cannot be empty.'; isLoading = false; return; }
		if (!payload.prompt) { error = 'Prompt cannot be empty.'; isLoading = false; return; }
        const checkHeight = payload.height ?? 512; // Use default for validation check
        const checkWidth = payload.width ?? 512;
		if (checkHeight <= 0 || checkWidth <= 0 || checkHeight % 8 !== 0 || checkWidth % 8 !== 0) {
			error = 'Height and Width must be positive multiples of 8.'; isLoading = false; return;
		}
        if ((payload.num_inference_steps ?? 0) <=0) { error = 'Steps must be positive.'; isLoading = false; return; }
        if ((payload.num_outputs ?? 0) <=0) { error = 'Number of outputs must be positive.'; isLoading = false; return; }
        // Add more checks if needed

		console.log('Sending payload:', payload);

		try {
			const response = await post<Text2ImageResponse>('/generate/text-to-image', payload);
			if (response && response.images) { results = response.images; }
            else { results = []; console.warn("API response OK but missing 'images' array:", response); }
			console.log('Generation successful:', response);
		} catch (e: unknown) {
			console.error('Generation failed:', e);
			if (isApiError(e)) {
                 if (typeof e.detail === 'string') { error = `Error ${e.status}: ${e.detail}`; }
                 else if (Array.isArray(e.detail)) { error = `Validation Error: ${e.detail.map(d => `${d.loc.slice(-1)}: ${d.msg}`).join(', ')}`; }
                 else if (typeof e.detail === 'object' && e.detail !== null) { error = `Error ${e.status}: ${JSON.stringify(e.detail)}`; }
                 else { error = `API Error ${e.status}: ${e.statusText || 'Unknown API error'}`; }
			} else if (e instanceof Error) { error = `Error: ${e.message}`; }
            else { error = `An unexpected error occurred: ${String(e)}`; }
			results = null;
		} finally {
			isLoading = false;
		}
	}

	// --- Seed Helpers (Update the store) ---
	function setRandomSeed() {
		t2iFormStore.update((current) => ({ ...current, seed: Math.floor(Math.random() * 2 ** 32) }));
	}
	function clearSeed() {
		t2iFormStore.update((current) => ({ ...current, seed: null }));
	}
</script>

<!-- Component Template (HTML binding to store) -->
<div
	class="max-w-2xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700"
>
	<h2 class="text-2xl font-semibold mb-6 text-center text-gray-800 dark:text-gray-200">
		Text-to-Image Generation
	</h2>

	<!-- Use $t2iFormStore for two-way binding -->
	<form on:submit|preventDefault={generateImage} class="space-y-5">
		<!-- Model Selector -->
		<div>
			<label for="model_id" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Model ID:</label
			>
			<select
				id="model_id"
				bind:value={$t2iFormStore.model_id} required
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			>
				{#each availableModels as model}
					<option value={model}>{model}</option>
				{/each}
			</select>
			<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
				Select loaded/Hub model. Add models to config/search paths.
			</p>
		</div>

		<!-- Scheduler Selector -->
		<div>
			<label
				for="scheduler_id"
				class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Scheduler:</label
			>
			<select
				id="scheduler_id"
				bind:value={$t2iFormStore.scheduler_id}
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			>
                <!-- Explicitly handle null mapping for Default -->
				<option value={null}>Default (Pipeline Default)</option>
				{#each availableSchedulers.filter(s => s !== 'Default') as scheduler}
					<option value={scheduler}>{scheduler}</option>
				{/each}
			</select>
		</div>

		<!-- Prompt -->
		<div>
			<label for="prompt" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Prompt:</label
			>
			<textarea
				id="prompt"
				rows="4"
				bind:value={$t2iFormStore.prompt} required
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			></textarea>
		</div>

		<!-- Negative Prompt -->
		<div>
			<label
				for="negative_prompt"
				class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Negative Prompt:</label
			>
			<textarea
				id="negative_prompt"
				rows="3"
				bind:value={$t2iFormStore.negative_prompt}
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			></textarea>
		</div>

		<!-- Dimensions -->
		<div class="grid grid-cols-2 gap-4">
			<div>
				<label for="width" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Width:</label
				>
				<input
					type="number"
					id="width"
					bind:value={$t2iFormStore.width}
					min="64" step="8" required
					class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
				/>
			</div>
			<div>
				<label for="height" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Height:</label
				>
				<input
					type="number"
					id="height"
					bind:value={$t2iFormStore.height}
					min="64" step="8" required
					class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
				/>
			</div>
		</div>

		<!-- Steps & CFG -->
		<div class="grid grid-cols-2 gap-4">
			<div>
				<label for="steps" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Steps: <span class="font-mono text-indigo-600 dark:text-indigo-400">{$t2iFormStore.num_inference_steps}</span
					></label
				>
				<input
					type="range" id="steps" bind:value={$t2iFormStore.num_inference_steps}
					min="1" max="150" step="1"
					class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
				/>
			</div>
			<div>
				<label for="cfg" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Guidance (CFG): <span class="font-mono text-indigo-600 dark:text-indigo-400">{$t2iFormStore.guidance_scale?.toFixed(1)}</span
					></label
				>
				<input
					type="range" id="cfg" bind:value={$t2iFormStore.guidance_scale}
					min="0" max="20" step="0.5"
					class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
				/>
			</div>
		</div>

		<!-- Seed -->
		<div>
			<label for="seed" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Seed:</label
			>
			<div class="flex items-center space-x-2">
				<input
					type="number" id="seed" placeholder="Random (null)" bind:value={$t2iFormStore.seed}
					min="0" step="1"
					class="flex-grow px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
				/>
				<button
					type="button" on:click={setRandomSeed} title="Generate Random Seed"
					class="p-2 bg-gray-200 dark:bg-gray-600 rounded hover:bg-gray-300 dark:hover:bg-gray-500 text-lg"
					>🎲</button
				>
				<button
					type="button" on:click={clearSeed} title="Use Random Seed (null)"
					class="p-2 bg-gray-200 dark:bg-gray-600 rounded hover:bg-gray-300 dark:hover:bg-gray-500 text-lg"
					disabled={$t2iFormStore.seed === null}>❌</button
				>
			</div>
		</div>

		<!-- Custom Sigmas (Optional) -->
		<div>
			<label
				for="custom_sigmas"
				class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Custom Sigmas (Optional):</label
			>
			<textarea
				id="custom_sigmas" rows="2" bind:value={customSigmasString} 
				placeholder="e.g., 14.6, 8.4, 5.2, ..."
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white font-mono text-sm"
			></textarea>
			<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
				Comma or space separated numbers. Overrides 'Steps' if provided. Backend currently only uses list length.
			</p>
		</div>

		<!-- Number of Outputs -->
		<div>
			<label
				for="num_outputs"
				class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Number of Images:</label
			>
			<input
				type="number" id="num_outputs" bind:value={$t2iFormStore.num_outputs}
				min="1" max="4" step="1" required
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			/>
		</div>

		<!-- Submit Button -->
		<div class="pt-4">
			<button
				type="submit"
				disabled={isLoading}
				class="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
			>
				{#if isLoading}
					<svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"> <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle> <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path> </svg>
					Generating...
				{:else}
					Generate Image
				{/if}
			</button>
		</div>
	</form>

	<!-- Error Display -->
	{#if error}
		<div
			class="mt-6 p-4 bg-red-100 border border-red-400 text-red-700 dark:bg-red-900 dark:text-red-200 dark:border-red-700 rounded"
			role="alert"
		>
			<strong class="font-bold">Error:</strong>
			<span class="block sm:inline break-words"> {error}</span>
		</div>
	{/if}

	<!-- Results Display -->
	{#if results && results.length > 0}
		<div class="mt-8">
			<h3 class="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">Results:</h3>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				{#each results as result, i}
					<div class="border dark:border-gray-700 rounded-lg overflow-hidden shadow bg-gray-50 dark:bg-gray-750">
						<img
							src="data:image/png;base64,{result.image_base64}"
							alt={`Generation result ${i + 1} for the prompt`} 
							class="w-full h-auto object-contain"
						/>
						<div class="p-2">
							<p class="text-xs text-gray-600 dark:text-gray-400">Seed: {result.seed}</p>
						</div>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>