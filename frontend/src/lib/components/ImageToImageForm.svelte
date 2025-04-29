<!-- frontend/src/lib/components/ImageToImageForm.svelte -->
<script lang="ts">
	// --- Imports ---
	import { get, post, isApiError, type ApiError } from '$lib/apiClient';
	import type { ImageToImageRequest, ImageToImageResponse, ImageResult, PresetRead, PresetSetting, PresetCreate } from '$lib/types'; // Use Img2Img types, add Preset types
	import { i2iFormStore } from '$lib/stores/i2iFormStore'; // Import the CORRECT store
	import { onMount, onDestroy, beforeUpdate } from 'svelte'; // Import lifecycle functions
    // --- Import Shared Components ---
	import ModelSelector from './shared/ModelSelector.svelte';
	import SchedulerSelector from './shared/SchedulerSelector.svelte';
	import SeedInput from './shared/SeedInput.svelte';
	import SubmitButton from './shared/SubmitButton.svelte';

	// --- Component State ---
	let isLoading = false;
	let error: string | null = null;
	let results: ImageResult[] | null = null;
	let uploadedFileName: string | null = null;
	let imagePreviewUrl: string | null = null;

	// --- State for Selectors ---
	// TODO: Fetch these from backend API later
    // --- Updated model list --- vvv
	let availableModels: string[] = [
        'photomatix_v3.fp16.safetensors', // Added
        'realisticVisionV51_v51VAE.safetensors', // Added
		'runwayml/stable-diffusion-v1-5', // Kept as example
		'stabilityai/stable-diffusion-xl-base-1.0', // Kept as example
	];
    // --- Updated model list --- ^^^
	let availableSchedulers: string[] = [
		'Default',
		'DDIMScheduler',
		'DPMSolverMultistepScheduler',
		'EulerAncestralDiscreteScheduler',
		'EulerDiscreteScheduler',
		'UniPCMultistepScheduler'
	];

	// --- State for Presets ---
	let availablePresets: PresetRead[] = [];
	let selectedPresetId: number | string = 'none';
	let isFetchingPresets = false;
    let currentModelId: string | null | undefined = null; // Track previous model ID for fetch logic


    // --- Initialize Preview & Fetch Presets on Mount ---
	onMount(() => {
		const currentBase64 = $i2iFormStore.image_base64;
		if (currentBase64) { /* ... logic to restore preview ... */ }
        else { imagePreviewUrl = null; uploadedFileName = null; }
        // Initial Fetch
        currentModelId = $i2iFormStore.model_id;
        if(currentModelId) { fetchPresets(currentModelId); }
	});

    // --- Fetch Presets when Model ID changes ---
    async function fetchPresets(modelId: string | null | undefined) {
        if (!modelId) { availablePresets = []; selectedPresetId = 'none'; return; }
        isFetchingPresets = true; console.log(`Fetching presets for model: ${modelId}`);
        try {
            const encodedModelId = encodeURIComponent(modelId);
            const presets = await get<PresetRead[]>(`/presets?model_identifier=${encodedModelId}`);
            availablePresets = presets || []; selectedPresetId = 'none'; console.log('Fetched presets:', availablePresets);
        } catch (err) { console.error("Failed to fetch presets:", err); availablePresets = []; selectedPresetId = 'none'; }
        finally { isFetchingPresets = false; }
    }
    // --- Refetch Presets using beforeUpdate ---
     beforeUpdate(() => {
        if ($i2iFormStore.model_id !== currentModelId) {
            console.log("I2I Model ID changed, refetching presets...");
            currentModelId = $i2iFormStore.model_id;
            fetchPresets(currentModelId);
        }
    });


    // --- Apply Preset when selected ---
    function applyPreset(presetIdValue: number | string | null) {
         const idToFind = typeof presetIdValue === 'string' ? parseInt(presetIdValue, 10) : presetIdValue;
         selectedPresetId = idToFind ?? 'none';
         if (idToFind === null || isNaN(idToFind)) { console.log("No preset selected or invalid ID."); return; }
         const preset = availablePresets.find(p => p.id === idToFind);
         if (preset) {
             console.log("Applying preset:", preset.preset_name, preset.settings);
             i2iFormStore.update(current => {
                 const updatedState = { ...current };
                 for (const key in preset.settings) {
                     if (Object.prototype.hasOwnProperty.call(preset.settings, key)) { (updatedState as any)[key] = (preset.settings as any)[key]; }
                 }
                 if (Object.prototype.hasOwnProperty.call(preset.settings, 'scheduler_id') && preset.settings.scheduler_id === null) { updatedState.scheduler_id = null; }
                 // Add other specific handling if needed (like clearing sigmas if steps is set)
                 return updatedState;
             });
         } else { console.warn(`Preset with ID ${idToFind} not found.`); }
    }

	// --- File Handling (Updates the store) ---
	async function handleFileChange(event: Event) {
		if (imagePreviewUrl) { URL.revokeObjectURL(imagePreviewUrl); imagePreviewUrl = null; }
		const input = event.target as HTMLInputElement;
		if (input.files && input.files[0]) {
			const file = input.files[0]; uploadedFileName = file.name; imagePreviewUrl = URL.createObjectURL(file);
			const reader = new FileReader();
			reader.onload = (e) => {
				if (e.target?.result && typeof e.target.result === 'string') {
					const base64 = e.target.result.split(',')[1];
					i2iFormStore.update(current => ({ ...current, image_base64: base64 })); error = null;
				} else {
                    error = 'Failed to read file data.'; i2iFormStore.update(current => ({ ...current, image_base64: '' }));
					uploadedFileName = null; if(imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl); imagePreviewUrl = null;
				}
			};
			reader.onerror = () => { /* ... error handling ... */ };
			reader.readAsDataURL(file);
		} else { i2iFormStore.update(current => ({ ...current, image_base64: '' })); uploadedFileName = null; imagePreviewUrl = null; }
	}

	// --- Submit Handler ---
	async function generateImage() {
		isLoading = true; error = null; results = null;
        const currentRequestData = $i2iFormStore;
		// Validation
		if (!currentRequestData.image_base64) { error = 'Please select an initial image file.'; isLoading = false; return; }
		if (!currentRequestData.model_id?.trim()) { error = 'Model ID cannot be empty.'; isLoading = false; return; }
		if (!currentRequestData.prompt?.trim()) { error = 'Prompt cannot be empty.'; isLoading = false; return; }
        // ... other validations ...
        const checkStrength = currentRequestData.strength ?? 0.75;
		if (checkStrength < 0 || checkStrength > 1) { error = 'Strength must be between 0.0 and 1.0.'; isLoading = false; return; }

		const payload: ImageToImageRequest = {
             model_id: currentRequestData.model_id.trim(), prompt: currentRequestData.prompt.trim(),
             negative_prompt: currentRequestData.negative_prompt?.trim() || null, height: Number(currentRequestData.height) || 512,
			 width: Number(currentRequestData.width) || 512, num_inference_steps: Number(currentRequestData.num_inference_steps) || 35,
			 guidance_scale: Number(currentRequestData.guidance_scale) || 7.0, strength: Number(currentRequestData.strength) || 0.75,
			 num_outputs: Number(currentRequestData.num_outputs) || 1,
			 seed: currentRequestData.seed === null || isNaN(Number(currentRequestData.seed)) ? null : Math.floor(Number(currentRequestData.seed)),
			 scheduler_id: currentRequestData.scheduler_id?.trim() || null, image_base64: currentRequestData.image_base64
		};
		console.log('Sending Img2Img payload:', payload);
		try {
			const response = await post<ImageToImageResponse>('/generate/image-to-image', payload);
			if (response && response.images) { results = response.images; } else { results = []; console.warn("API response OK but missing 'images' array:", response); }
			console.log('Generation successful:', response);
		} catch (e: unknown) { /* ... error handling ... */ }
        finally { isLoading = false; }
	}

	// --- Handlers for SeedInput events (update the i2i store) ---
	function handleSeedChange(event: CustomEvent<number | null>) { i2iFormStore.update(s => ({...s, seed: event.detail })); }
    function handleSeedRandomize() { i2iFormStore.update((current) => ({ ...current, seed: Math.floor(Math.random() * 2 ** 32) })); }
    function handleSeedClear() { i2iFormStore.update((current) => ({ ...current, seed: null })); }

    // --- Save Preset Logic ---
     async function saveCurrentAsPreset() {
        const name = prompt("Enter a name for this preset:");
        if (!name || name.trim() === '') { alert("Preset name cannot be empty."); return; }
        const currentSettings = $i2iFormStore;
        if (!currentSettings.model_id) { alert("Cannot save preset without a selected Model ID."); return; }
        const presetSettings: PresetSetting = {
             negative_prompt: currentSettings.negative_prompt, height: currentSettings.height, width: currentSettings.width,
             num_inference_steps: currentSettings.num_inference_steps, guidance_scale: currentSettings.guidance_scale,
             scheduler_id: currentSettings.scheduler_id, strength: currentSettings.strength, // Include strength
        };
        const payload: PresetCreate = {
             model_identifier: currentSettings.model_id, preset_name: name.trim(), settings: presetSettings,
             description: `Saved from I2I form on ${new Date().toLocaleDateString()}`
        };
        console.log("Saving preset:", payload);
        const wasLoading = isLoading; isLoading = true; const oldError = error; error = null;
        try {
            await post<PresetRead>('/presets', payload); alert(`Preset "${name}" saved successfully!`);
            fetchPresets(currentSettings.model_id);
        } catch (err: unknown) { /* ... error handling ... */ }
        finally { isLoading = false; if(error === null && !wasLoading) error = oldError; }
    }

	// --- Cleanup object URL ---
	onDestroy(() => {
		if (imagePreviewUrl) { URL.revokeObjectURL(imagePreviewUrl); }
	});
</script>

<!-- Component Template (Uses Shared Components & Explicit Attributes) -->
<div class="max-w-2xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
	<h2 class="text-2xl font-semibold mb-6 text-center text-gray-800 dark:text-gray-200">
		Image-to-Image Generation
	</h2>

    <!-- Use $i2iFormStore for bindings -->
	<form on:submit|preventDefault={generateImage} class="space-y-5">
		<!-- Image Upload -->
		<div>
			<label for="init_image_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Initial Image:</label>
			<input type="file" id="init_image_i2i" required accept="image/png, image/jpeg, image/webp" on:change={handleFileChange} class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400"/>
			{#if uploadedFileName} <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">Selected: {uploadedFileName}</p> {/if}
			{#if imagePreviewUrl} <div class="mt-2 border dark:border-gray-600 rounded overflow-hidden max-w-[128px]"> <img src={imagePreviewUrl} alt="Uploaded preview" class="w-full h-auto object-contain"/> </div> {/if}
		</div>

		<!-- USE SHARED ModelSelector -->
		<ModelSelector
			id="model_id_i2i" label="Model ID:"
			availableModels={availableModels}
			selectedModelId={$i2iFormStore.model_id}
			on:change={(e) => i2iFormStore.update(s => ({ ...s, model_id: e.detail }))}
		/>

        <!-- Preset Selector -->
        <div>
			<label for="preset_selector_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Load Preset:</label>
            <div class="flex items-center space-x-2">
                <select
                    id="preset_selector_i2i" bind:value={selectedPresetId}
                    on:change={(e) => applyPreset((e.target as HTMLSelectElement).value)}
                    disabled={isFetchingPresets || availablePresets.length === 0}
                    class="flex-grow w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
                >
                    <option value="none" disabled={selectedPresetId !== 'none'}>-- Select a Preset --</option>
                    {#each availablePresets as preset (preset.id)} <option value={preset.id}>{preset.preset_name} {preset.is_user_defined ? '' : '[System]'}</option> {/each}
                </select>
                {#if isFetchingPresets} <div class="w-5 h-5 border-t-2 border-b-2 border-indigo-500 rounded-full animate-spin"></div> {/if}
            </div>
		</div>

		<!-- Prompt -->
		<div>
			<label for="prompt_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Prompt:</label>
			<textarea id="prompt_i2i" rows="3" bind:value={$i2iFormStore.prompt} required class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"></textarea>
		</div>

		<!-- Negative Prompt -->
		<div>
			<label for="negative_prompt_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Negative Prompt:</label>
			<textarea id="negative_prompt_i2i" rows="2" bind:value={$i2iFormStore.negative_prompt} class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"></textarea>
		</div>

		<!-- Strength -->
		<div>
			<label for="strength_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Strength: <span class="font-mono text-indigo-600 dark:text-indigo-400">{$i2iFormStore.strength?.toFixed(2)}</span></label>
			<input type="range" id="strength_i2i" bind:value={$i2iFormStore.strength} min="0" max="1" step="0.01" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"/>
			<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">Controls influence (0=original, 1=ignore original).</p>
		</div>

		<!-- Dimensions -->
		<div class="grid grid-cols-2 gap-4">
			<div>
				<label for="width_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Width:</label>
				<input type="number" id="width_i2i" bind:value={$i2iFormStore.width} min="64" step="8" required class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
			</div>
			<div>
				<label for="height_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Height:</label>
				<input type="number" id="height_i2i" bind:value={$i2iFormStore.height} min="64" step="8" required class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
			</div>
		</div>

		<!-- Steps & CFG -->
        <div class="grid grid-cols-2 gap-4">
			<div>
                <label for="steps_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Steps: <span class="font-mono text-indigo-600 dark:text-indigo-400">{$i2iFormStore.num_inference_steps}</span></label>
				<input type="range" id="steps_i2i" bind:value={$i2iFormStore.num_inference_steps} min="1" max="150" step="1" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"/>
			</div>
			<div>
                <label for="cfg_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Guidance (CFG): <span class="font-mono text-indigo-600 dark:text-indigo-400">{$i2iFormStore.guidance_scale?.toFixed(1)}</span></label>
				<input type="range" id="cfg_i2i" bind:value={$i2iFormStore.guidance_scale} min="0" max="20" step="0.5" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"/>
			</div>
		</div>

		<!-- USE SHARED SchedulerSelector -->
        <SchedulerSelector
            id="scheduler_id_i2i" label="Scheduler:"
            availableSchedulers={availableSchedulers}
            selectedSchedulerId={$i2iFormStore.scheduler_id}
            on:change={(e) => i2iFormStore.update(s => ({...s, scheduler_id: e.detail }))}
        />

		<!-- USE SHARED SeedInput -->
        <SeedInput
            id="seed_i2i" label="Seed:"
            seedValue={$i2iFormStore.seed}
            on:change={handleSeedChange}
            on:randomize={handleSeedRandomize}
            on:clear={handleSeedClear}
        />

		<!-- Number of Outputs -->
		<div>
            <label for="num_outputs_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Number of Images:</label>
            <input type="number" id="num_outputs_i2i" bind:value={$i2iFormStore.num_outputs} min="1" max="4" step="1" required class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
        </div>

		<!-- USE SHARED SubmitButton -->
		<div class="pt-4">
            <SubmitButton
                 isLoading={isLoading}
                 buttonText="Generate Image Variation"
                 loadingText="Generating..."
             />
		</div>

        <!-- Save Preset Button -->
         <div class="pt-1">
            <button
				type="button" on:click={saveCurrentAsPreset}
				disabled={isLoading || !$i2iFormStore.model_id}
				class="w-full inline-flex justify-center py-2 px-4 border border-gray-300 dark:border-gray-600 shadow-sm text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
			>
				💾 Save Current Settings as Preset...
			</button>
        </div>
	</form>

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
			<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
				{#each results as result, i}
					<div class="border dark:border-gray-700 rounded-lg overflow-hidden shadow bg-gray-50 dark:bg-gray-750">
						<img src="data:image/png;base64,{result.image_base64}" alt={`Image-to-image result ${i + 1} for the prompt`} class="w-full h-auto object-contain" />
						<div class="p-2"> <p class="text-xs text-gray-600 dark:text-gray-400">Seed: {result.seed}</p> </div>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>