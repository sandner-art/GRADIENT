<!-- frontend/src/lib/components/ImageToImageForm.svelte -->
<script lang="ts">
	// --- Imports ---
	import { post, isApiError, type ApiError } from '$lib/apiClient';
	// --- Import types from centralized file ---
	import type { ImageToImageRequest, ImageToImageResponse, ImageResult } from '$lib/types';
	import { i2iFormStore } from '$lib/stores/i2iFormStore'; // Import the CORRECT store
	import { onMount, onDestroy } from 'svelte'; // Import lifecycle functions

	// --- Component State (Managed Externally by Store) ---
	// Local state only for UI elements not directly part of the request payload
	let isLoading = false;
	let error: string | null = null;
	let results: ImageResult[] | null = null;
	let uploadedFileName: string | null = null; // To display the name of the uploaded file
	let imagePreviewUrl: string | null = null; // For showing a preview of the uploaded image

	// --- Placeholder Data for Selectors ---
	// TODO: Fetch these from backend API later
	let availableModels: string[] = [
		'photomatix_v3.fp16.safetensors',
		'realisticVisionV51_v51VAE.safetensors',
        'emilianJR/epiCRealism', // Example common checkpoint name
		'sd_xl_base_1.0.safetensors',
		'/path/to/absolute/model.ckpt'
	];
	let availableSchedulers: string[] = [
		'Default', // Represents null/pipeline default
		'DDIMScheduler',
		'DPMSolverMultistepScheduler',
		'EulerAncestralDiscreteScheduler',
		'EulerDiscreteScheduler',
		'UniPCMultistepScheduler'
	];

    // --- Initialize Preview on Mount if Store has Image ---
    // Note: Base64 data isn't persisted across page reloads by default store,
    // but this handles re-mounting if navigating within SPA without reload.
    onMount(() => {
        const currentBase64 = $i2iFormStore.image_base64;
        if (currentBase64) {
            // Recreate object URL for preview if base64 exists in store
             try {
                const byteString = atob(currentBase64);
                const mimeString = 'image/png'; // Assume PNG or determine from data? Hard for now.
                const ab = new ArrayBuffer(byteString.length);
                const ia = new Uint8Array(ab);
                for (let i = 0; i < byteString.length; i++) {
                    ia[i] = byteString.charCodeAt(i);
                }
                const blob = new Blob([ab], { type: mimeString });
                imagePreviewUrl = URL.createObjectURL(blob);
                uploadedFileName = "Image from previous session"; // Placeholder name
            } catch(e) {
                console.error("Could not create preview from stored base64:", e);
                // Clear stored base64 if invalid?
                i2iFormStore.update(s => ({ ...s, image_base64: '' }));
            }
        } else {
            imagePreviewUrl = null;
            uploadedFileName = null;
        }
    });


	// --- File Handling (Updates the store) ---
	async function handleFileChange(event: Event) {
        // Clean up previous preview URL first
        if (imagePreviewUrl) {
             URL.revokeObjectURL(imagePreviewUrl);
             imagePreviewUrl = null;
        }

		const input = event.target as HTMLInputElement;
		if (input.files && input.files[0]) {
			const file = input.files[0];
			uploadedFileName = file.name;
			imagePreviewUrl = URL.createObjectURL(file); // Create temporary URL for preview

			const reader = new FileReader();
			reader.onload = (e) => {
				if (e.target?.result && typeof e.target.result === 'string') {
					const base64 = e.target.result.split(',')[1];
                    // Update the store
					i2iFormStore.update(current => ({ ...current, image_base64: base64 }));
					error = null;
				} else {
                    error = 'Failed to read file data.';
                     i2iFormStore.update(current => ({ ...current, image_base64: '' }));
					uploadedFileName = null;
					URL.revokeObjectURL(imagePreviewUrl ?? '');
					imagePreviewUrl = null;
				}
			};
			reader.onerror = () => {
                error = 'Error reading the selected file.';
                i2iFormStore.update(current => ({ ...current, image_base64: '' }));
				uploadedFileName = null;
				URL.revokeObjectURL(imagePreviewUrl ?? '');
				imagePreviewUrl = null;
			};
			reader.readAsDataURL(file);
		} else {
            i2iFormStore.update(current => ({ ...current, image_base64: '' }));
			uploadedFileName = null;
            // URL already revoked if previously set
			imagePreviewUrl = null;
		}
	}

	// --- Submit Handler ---
	async function generateImage() {
		isLoading = true;
		error = null;
		results = null;

		// --- Get current state FROM THE STORE ---
        const currentRequestData = $i2iFormStore;

		// --- Basic Client-Side Validation ---
		if (!currentRequestData.image_base64) { error = 'Please select an initial image file.'; isLoading = false; return; }
		if (!currentRequestData.model_id?.trim()) { error = 'Model ID cannot be empty.'; isLoading = false; return; }
		if (!currentRequestData.prompt?.trim()) { error = 'Prompt cannot be empty.'; isLoading = false; return; }
		const checkHeight = currentRequestData.height ?? 512;
		const checkWidth = currentRequestData.width ?? 512;
		if (checkHeight <= 0 || checkWidth <= 0 || checkHeight % 8 !== 0 || checkWidth % 8 !== 0) {
			error = 'Height and Width must be positive multiples of 8.'; isLoading = false; return;
		}
        const checkStrength = currentRequestData.strength ?? 0.75;
		if (checkStrength < 0 || checkStrength > 1) { error = 'Strength must be between 0.0 and 1.0.'; isLoading = false; return; }
        if ((currentRequestData.num_inference_steps ?? 0) <=0) { error = 'Steps must be positive.'; isLoading = false; return; }
        if ((currentRequestData.num_outputs ?? 0) <=0) { error = 'Number of outputs must be positive.'; isLoading = false; return; }

		// Create payload from store data
		const payload: ImageToImageRequest = {
			...currentRequestData,
            model_id: currentRequestData.model_id.trim(),
            prompt: currentRequestData.prompt.trim(),
            negative_prompt: currentRequestData.negative_prompt?.trim() || '',
			height: Number(currentRequestData.height) || 512,
			width: Number(currentRequestData.width) || 512,
			num_inference_steps: Number(currentRequestData.num_inference_steps) || 35,
			guidance_scale: Number(currentRequestData.guidance_scale) || 7.0,
			strength: Number(currentRequestData.strength) || 0.75,
			num_outputs: Number(currentRequestData.num_outputs) || 1,
			seed: currentRequestData.seed === null || isNaN(Number(currentRequestData.seed)) ? null : Math.floor(Number(currentRequestData.seed)),
			scheduler_id: currentRequestData.scheduler_id === 'Default' || !currentRequestData.scheduler_id ? null : currentRequestData.scheduler_id.trim(),
            // image_base64 is already in currentRequestData from the store
		};

		console.log('Sending Img2Img payload:', payload);

		try {
			const response = await post<ImageToImageResponse>('/generate/image-to-image', payload);
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
		i2iFormStore.update((current) => ({ ...current, seed: Math.floor(Math.random() * 2 ** 32) }));
	}
	function clearSeed() {
		i2iFormStore.update((current) => ({ ...current, seed: null }));
	}

	// --- Cleanup object URL ---
	onDestroy(() => {
		if (imagePreviewUrl) {
			URL.revokeObjectURL(imagePreviewUrl);
            console.log("Revoked preview URL on destroy");
		}
	});
</script>

<!-- Component Template (HTML binding to store) -->
<div
	class="max-w-2xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700"
>
	<h2 class="text-2xl font-semibold mb-6 text-center text-gray-800 dark:text-gray-200">
		Image-to-Image Generation
	</h2>

    <!-- Use $i2iFormStore for two-way binding -->
	<form on:submit|preventDefault={generateImage} class="space-y-5">
		<!-- Image Upload -->
		<div>
			<label for="init_image" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Initial Image:</label
			>
			<input
				type="file" id="init_image" required
				accept="image/png, image/jpeg, image/webp" on:change={handleFileChange}
				class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400"
			/>
			{#if uploadedFileName}
				<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">Selected: {uploadedFileName}</p>
			{/if}
			{#if imagePreviewUrl}
				<div class="mt-2 border dark:border-gray-600 rounded overflow-hidden max-w-[128px]">
					<img src={imagePreviewUrl} alt="Uploaded preview" class="w-full h-auto object-contain" />
				</div>
			{/if}
		</div>

		<!-- Model Selector -->
		<div>
			<label for="model_id_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Model ID:</label
			>
			<select
				id="model_id_i2i" bind:value={$i2iFormStore.model_id} required
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			>
				{#each availableModels as model}
					<option value={model}>{model}</option>
				{/each}
			</select>
		</div>

		<!-- Prompt -->
		<div>
			<label for="prompt_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Prompt:</label
			>
			<textarea
				id="prompt_i2i" rows="3" bind:value={$i2iFormStore.prompt} required
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			></textarea>
		</div>

		<!-- Negative Prompt -->
		<div>
			<label
				for="negative_prompt_i2i"
				class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Negative Prompt:</label
			>
			<textarea
				id="negative_prompt_i2i" rows="2" bind:value={$i2iFormStore.negative_prompt}
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			></textarea>
		</div>

		<!-- Strength -->
		<div>
			<label for="strength" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Strength: <span class="font-mono text-indigo-600 dark:text-indigo-400">{$i2iFormStore.strength?.toFixed(2)}</span
				></label
			>
			<input
				type="range" id="strength" bind:value={$i2iFormStore.strength}
				min="0" max="1" step="0.01"
				class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
			/>
			<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
				Controls how much the prompt influences the image (0=original, 1=ignore original).
			</p>
		</div>

		<!-- Dimensions -->
		<div class="grid grid-cols-2 gap-4">
			<div>
				<label for="width_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Width:</label
				>
				<input
					type="number" id="width_i2i" bind:value={$i2iFormStore.width}
					min="64" step="8" required
					class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
				/>
			</div>
			<div>
				<label for="height_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Height:</label
				>
				<input
					type="number" id="height_i2i" bind:value={$i2iFormStore.height}
					min="64" step="8" required
					class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
				/>
			</div>
		</div>

		<!-- Steps & CFG -->
		<div class="grid grid-cols-2 gap-4">
			<div>
				<label for="steps_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Steps: <span class="font-mono text-indigo-600 dark:text-indigo-400">{$i2iFormStore.num_inference_steps}</span
					></label
				>
				<input
					type="range" id="steps_i2i" bind:value={$i2iFormStore.num_inference_steps}
					min="1" max="150" step="1"
					class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
				/>
			</div>
			<div>
				<label for="cfg_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Guidance (CFG): <span class="font-mono text-indigo-600 dark:text-indigo-400">{$i2iFormStore.guidance_scale?.toFixed(1)}</span
					></label
				>
				<input
					type="range" id="cfg_i2i" bind:value={$i2iFormStore.guidance_scale}
					min="0" max="20" step="0.5"
					class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
				/>
			</div>
		</div>

		<!-- Scheduler Selector -->
		<div>
			<label
				for="scheduler_id_i2i"
				class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Scheduler:</label
			>
			<select
				id="scheduler_id_i2i" bind:value={$i2iFormStore.scheduler_id}
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			>
                <option value={null}>Default (Pipeline Default)</option>
				{#each availableSchedulers.filter(s => s !== 'Default') as scheduler}
					<option value={scheduler}>{scheduler}</option>
				{/each}
			</select>
		</div>


		<!-- Seed -->
		<div>
			<label for="seed_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Seed:</label
			>
			<div class="flex items-center space-x-2">
				<input
					type="number" id="seed_i2i" placeholder="Random (null)" bind:value={$i2iFormStore.seed}
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
					disabled={$i2iFormStore.seed === null}>❌</button
				>
			</div>
		</div>

		<!-- Number of Outputs -->
		<div>
			<label
				for="num_outputs_i2i"
				class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Number of Images:</label
			>
			<input
				type="number" id="num_outputs_i2i" bind:value={$i2iFormStore.num_outputs}
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
					Generate Image Variation
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
					<div
						class="border dark:border-gray-700 rounded-lg overflow-hidden shadow bg-gray-50 dark:bg-gray-750"
					>
						<img
							src="data:image/png;base64,{result.image_base64}"
							alt={`Image-to-image result ${i + 1} for the prompt`} 
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