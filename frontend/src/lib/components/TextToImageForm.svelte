<!-- frontend/src/lib/components/TextToImageForm.svelte -->
<script lang="ts">
	import { post, isApiError, type ApiError } from '$lib/apiClient'; // Import ApiError type too

	// Define types matching backend Pydantic models (ideally move to src/lib/types.ts later)
	// Removed 'export' keyword as these are local type definitions for this component
	interface DiffusionTaskBase {
		model_id: string;
		prompt: string;
		negative_prompt?: string;
		height?: number;
		width?: number;
		num_inference_steps?: number;
		guidance_scale?: number;
		seed?: number | null;
        custom_sigmas?: number[] | null;
	}

	interface T2IRequest extends DiffusionTaskBase {
		num_outputs?: number;
	}

	interface ImageResult {
		image_base64: string;
		seed: number;
		parameters: Record<string, any>;
	}

	interface Text2ImageResponse {
		images: ImageResult[];
		request_params: T2IRequest;
		performance_ms?: number;
	}

    // Component State
    let requestData: T2IRequest = {
        model_id: 'realisticVisionV51_v51VAE.safetensors', // Default model
        prompt: 'photo of a tabby cat programmer writing code, high resolution',
        negative_prompt: 'ugly, blurry, deformed, text, words, low quality',
        height: 512,
        width: 512,
        num_inference_steps: 25,
        guidance_scale: 7.0,
        seed: null, // Use null for random seed
        num_outputs: 1,
        custom_sigmas: null // Default to null
    };

    let isLoading = false;
    let error: string | null = null;
    let results: ImageResult[] | null = null;
    let customSigmasString: string = ''; // Input string for custom sigmas

    // Reactive statement to parse customSigmasString into requestData.custom_sigmas
    $: {
        if (customSigmasString && customSigmasString.trim() !== '') {
            try {
                const parsedSigmas = customSigmasString
                    .split(/[\s,]+/) // Split by space or comma
                    .filter(s => s.trim() !== '') // Remove empty entries
                    .map(s => parseFloat(s)); // Convert to numbers

                // Check if all parsed values are valid numbers
                if (parsedSigmas.every(n => !isNaN(n))) {
                    requestData.custom_sigmas = parsedSigmas;
                } else {
                    // Reset or indicate error if parsing results in NaN
                    requestData.custom_sigmas = null;
                    // Optionally set a specific validation error message here
                    console.warn("Invalid sigma input detected - contains non-numbers.");
                }
            } catch (e) {
                // Catch potential errors during split/map/parseFloat
                requestData.custom_sigmas = null; // Reset on error
                console.error("Error parsing custom sigmas string:", e);
            }
        } else {
            requestData.custom_sigmas = null; // Set back to null if textarea is empty
        }
    }


    async function generateImage() {
        isLoading = true;
        error = null;
        results = null;

        // Create payload, ensuring correct types for numbers and null seed
        const payload: T2IRequest = {
            ...requestData,
            // Ensure numeric types are numbers, using defaults if conversion fails
            height: Number(requestData.height) || 512,
            width: Number(requestData.width) || 512,
            num_inference_steps: Number(requestData.num_inference_steps) || 25,
            guidance_scale: Number(requestData.guidance_scale) || 7.0,
            num_outputs: Number(requestData.num_outputs) || 1,
            // Ensure seed is explicitly null or a valid integer
            seed: requestData.seed === null || isNaN(Number(requestData.seed)) ? null : Math.floor(Number(requestData.seed)),
             // custom_sigmas is updated reactively by the $: block
        };

        console.log('Sending payload:', payload); // Debug: Check payload

        try {
            const response = await post<Text2ImageResponse>('/generate/text-to-image', payload);
            results = response.images;
            console.log('Generation successful:', response);
        } catch (e: unknown) { // Catch as 'unknown' type
            console.error('Generation failed:', e);
            // --- Type Narrowing for Error Handling --- vvv
            if (isApiError(e)) { // Check if it matches our ApiError type
                 error = `API Error (${e.status}): ${JSON.stringify(e.detail || e.statusText)}`;
            } else if (e instanceof Error) { // Check if it's a standard JS Error
                 error = `Error: ${e.message}`;
            } else {
                 // Fallback for other types of thrown objects
                 error = `An unexpected error occurred: ${String(e)}`;
            }
            // --- Type Narrowing for Error Handling --- ^^^
            results = null;
        } finally {
            isLoading = false;
        }
    }

    // Helper for random seed buttons
    function setRandomSeed() {
        requestData.seed = Math.floor(Math.random() * 2**32);
    }
    function clearSeed() {
        requestData.seed = null;
    }

</script>

<!-- Component Template (HTML with Tailwind CSS) -->
<div class="max-w-2xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
	<h2 class="text-2xl font-semibold mb-6 text-center text-gray-800 dark:text-gray-200">Text-to-Image Generation</h2>

	<form on:submit|preventDefault={generateImage} class="space-y-5">
		<!-- Model ID -->
		<div>
			<label for="model_id" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Model ID:</label>
			<input type="text" id="model_id" bind:value={requestData.model_id} required class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
            <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">Filename in checkpoints/ or diffusers/ folder, or Hub ID.</p>
		</div>

		<!-- Prompt -->
		<div>
			<label for="prompt" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Prompt:</label>
			<textarea id="prompt" rows="4" bind:value={requestData.prompt} required class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"></textarea>
		</div>

		<!-- Negative Prompt -->
		<div>
			<label for="negative_prompt" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Negative Prompt:</label>
			<textarea id="negative_prompt" rows="3" bind:value={requestData.negative_prompt} class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"></textarea>
		</div>

		<!-- Dimensions -->
		<div class="grid grid-cols-2 gap-4">
			<div>
				<label for="width" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Width:</label>
				<input type="number" id="width" bind:value={requestData.width} min="64" step="64" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
			</div>
			<div>
				<label for="height" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Height:</label>
				<input type="number" id="height" bind:value={requestData.height} min="64" step="64" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
			</div>
		</div>

        <!-- Steps & CFG -->
        <div class="grid grid-cols-2 gap-4">
			<div>
                <label for="steps" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Steps: <span class="font-mono text-indigo-600 dark:text-indigo-400">{requestData.num_inference_steps}</span></label>
				<input type="range" id="steps" bind:value={requestData.num_inference_steps} min="1" max="150" step="1" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"/>
			</div>
			<div>
                <label for="cfg" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Guidance (CFG): <span class="font-mono text-indigo-600 dark:text-indigo-400">{requestData.guidance_scale?.toFixed(1)}</span></label>
				<input type="range" id="cfg" bind:value={requestData.guidance_scale} min="0" max="20" step="0.5" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"/>
			</div>
		</div>

        <!-- Seed -->
        <div>
            <label for="seed" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Seed:</label>
            <div class="flex items-center space-x-2">
                <input type="number" id="seed" placeholder="Random (null)" bind:value={requestData.seed} min="0" step="1" class="flex-grow px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
                <button type="button" on:click={setRandomSeed} title="Generate Random Seed" class="p-2 bg-gray-200 dark:bg-gray-600 rounded hover:bg-gray-300 dark:hover:bg-gray-500 text-lg">🎲</button>
                 <button type="button" on:click={clearSeed} title="Use Random Seed (null)" class="p-2 bg-gray-200 dark:bg-gray-600 rounded hover:bg-gray-300 dark:hover:bg-gray-500 text-lg" disabled={requestData.seed === null}>❌</button>
            </div>
		</div>

        <!-- Custom Sigmas (Optional) -->
        <div>
			<label for="custom_sigmas" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Custom Sigmas (Optional):</label>
			<textarea id="custom_sigmas" rows="2" bind:value={customSigmasString} placeholder="e.g., 14.6, 8.4, 5.2, ..." class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white font-mono text-sm"></textarea>
            <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">Comma or space separated numbers. Overrides 'Steps' if provided. Backend currently only uses list length.</p>
		</div>

        <!-- Number of Outputs -->
		<div>
            <label for="num_outputs" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Number of Images:</label>
            <input type="number" id="num_outputs" bind:value={requestData.num_outputs} min="1" max="4" step="1" class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
        </div>


		<!-- Submit Button -->
		<div class="pt-4">
			<button type="submit" disabled={isLoading} class="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out">
				{#if isLoading}
                    <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Generating...
                {:else}
				    Generate Image
                {/if}
			</button>
		</div>
	</form>

    <!-- Loading Indicator (Alternative simple text) -->
    <!-- {#if isLoading}
        <div class="mt-6 text-center text-sm text-gray-500 dark:text-gray-400">
             Generating, please wait...
        </div>
    {/if} -->

    <!-- Error Display -->
     {#if error}
        <div class="mt-6 p-4 bg-red-100 border border-red-400 text-red-700 dark:bg-red-900 dark:text-red-200 dark:border-red-700 rounded" role="alert">
            <strong class="font-bold">Error:</strong>
            <span class="block sm:inline break-words"> {error}</span>
        </div>
    {/if}

    <!-- Results Display -->
    {#if results && results.length > 0}
        <div class="mt-8">
            <h3 class="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">Results:</h3>
            <!-- Corrected comment syntax -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4"> <!-- Grid for multiple images -->
                {#each results as result, i}
                    <div class="border dark:border-gray-700 rounded-lg overflow-hidden shadow bg-gray-50 dark:bg-gray-750">
                        <img src="data:image/png;base64,{result.image_base64}" alt="Generated image {i+1}" class="w-full h-auto object-contain"/>
                        <div class="p-2">
                            <p class="text-xs text-gray-600 dark:text-gray-400">Seed: {result.seed}</p>
                            <!-- Expandable details TBD -->
                        </div>
                    </div>
                {/each}
            </div>
        </div>
    {/if}

</div>