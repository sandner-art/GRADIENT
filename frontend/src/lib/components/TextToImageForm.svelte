<!-- frontend/src/lib/components/TextToImageForm.svelte -->
<script lang="ts">
	// --- Imports ---
	import { get, post, isApiError, type ApiError } from '$lib/apiClient';
	import type { Text2ImageRequest, Text2ImageResponse, ImageResult, PresetRead, PresetSetting, PresetCreate } from '$lib/types'; // Added PresetCreate
	import { t2iFormStore } from '$lib/stores/t2iFormStore';
	import { onMount, beforeUpdate } from 'svelte'; // Import beforeUpdate
    import ModelSelector from './shared/ModelSelector.svelte';
    import SchedulerSelector from './shared/SchedulerSelector.svelte';
    import SeedInput from './shared/SeedInput.svelte';
    import SubmitButton from './shared/SubmitButton.svelte';

	// --- Component State ---
	let isLoading = false;
	let error: string | null = null;
	let results: ImageResult[] | null = null;
	let customSigmasString: string = '';
	// TODO: Fetch these from backend API later
	let availableModels: string[] = [
        'photomatix_v3.fp16.safetensors', // Added
        'realisticVisionV51_v51VAE.safetensors', // Added
		'runwayml/stable-diffusion-v1-5', // Kept as example
		'stabilityai/stable-diffusion-xl-base-1.0', // Kept as example
	];
    
	let availableSchedulers: string[] = [
		'Default',
		'DDIMScheduler',
		'DPMSolverMultistepScheduler',
		'EulerAncestralDiscreteScheduler',
		'EulerDiscreteScheduler',
		'UniPCMultistepScheduler'
	];
    let availablePresets: PresetRead[] = [];
    let selectedPresetId: number | string = 'none'; // Tracks the <select> value
    let isFetchingPresets = false;
    let currentModelId: string | null | undefined = null; // Track previous model ID for fetch logic

	// --- Reactive Parsing for Custom Sigmas ---
	$: {
		if (customSigmasString && customSigmasString.trim() !== '') {
			try {
				const parsedSigmas = customSigmasString.split(/[\s,]+/).filter((s) => s.trim() !== '').map((s) => parseFloat(s));
				if (parsedSigmas.every((n) => !isNaN(n))) {
                    if (JSON.stringify($t2iFormStore.custom_sigmas) !== JSON.stringify(parsedSigmas)) { t2iFormStore.update((current) => ({ ...current, custom_sigmas: parsedSigmas })); }
				} else {
                    if ($t2iFormStore.custom_sigmas !== null) { t2iFormStore.update((current) => ({ ...current, custom_sigmas: null })); }
					console.warn('Invalid sigma input detected - contains non-numbers.');
				}
			} catch (e) {
                 if ($t2iFormStore.custom_sigmas !== null) { t2iFormStore.update((current) => ({ ...current, custom_sigmas: null })); }
				 console.error('Error parsing custom sigmas string:', e);
			}
		} else {
             if ($t2iFormStore.custom_sigmas !== null) { t2iFormStore.update((current) => ({ ...current, custom_sigmas: null })); }
		}
	}

    // --- Fetch Presets Function ---
    async function fetchPresets(modelId: string | null | undefined) {
        if (!modelId) { availablePresets = []; selectedPresetId = 'none'; return; }
        isFetchingPresets = true; console.log(`Fetching presets for model: ${modelId}`);
        try {
            const encodedModelId = encodeURIComponent(modelId);
            const presets = await get<PresetRead[]>(`/presets?model_identifier=${encodedModelId}`);
            availablePresets = presets || [];
            selectedPresetId = 'none'; // Reset selection dropdown when presets reload for a new model
            console.log('Fetched presets:', availablePresets);
        } catch (err) { console.error("Failed to fetch presets:", err); availablePresets = []; selectedPresetId = 'none'; }
        finally { isFetchingPresets = false; }
    }

    // --- Apply Preset when selected ---
    function applyPreset(presetIdValue: number | string | null) { // Value from select change event
         selectedPresetId = presetIdValue ?? 'none'; // Update state tracking the select element value
         const idToFind = typeof presetIdValue === 'string' ? parseInt(presetIdValue, 10) : presetIdValue;

         if (idToFind === null || isNaN(idToFind)) { console.log("No preset selected or invalid ID."); return; }

         const preset = availablePresets.find(p => p.id === idToFind);
         if (preset) {
             console.log("Applying preset:", preset.preset_name, preset.settings);
             t2iFormStore.update(current => {
                 const updatedState = { ...current };
                 for (const key in preset.settings) {
                     if (Object.prototype.hasOwnProperty.call(preset.settings, key)) { (updatedState as any)[key] = (preset.settings as any)[key]; }
                 }
                 if (Object.prototype.hasOwnProperty.call(preset.settings, 'scheduler_id') && preset.settings.scheduler_id === null) { updatedState.scheduler_id = null; }
                 if (preset.settings.num_inference_steps !== undefined && preset.settings.num_inference_steps !== null) { updatedState.custom_sigmas = null; customSigmasString = ''; }
                 return updatedState;
             });
         } else { console.warn(`Preset with ID ${idToFind} not found.`); }
    }

	// --- Initialize and Handle Model Changes ---
	onMount(() => {
		const currentSigmas = $t2iFormStore.custom_sigmas;
		if (currentSigmas) { customSigmasString = currentSigmas.join(', '); } else { customSigmasString = ''; }
        currentModelId = $t2iFormStore.model_id; // Store initial ID
        if(currentModelId) { fetchPresets(currentModelId); }
	});

    beforeUpdate(() => {
        // Check if model ID in store changed
        if ($t2iFormStore.model_id !== currentModelId) {
            console.log("Model ID changed in store, refetching presets...");
            currentModelId = $t2iFormStore.model_id; // Update tracker
            fetchPresets(currentModelId); // Fetch for the new model
            // DO NOT reset selectedPresetId here, fetchPresets does it
        }
    });

	// --- Submit Handler ---
	async function generateImage() {
		isLoading = true; error = null; results = null;
		const currentRequestData = $t2iFormStore;
		const payload: Text2ImageRequest = {
			model_id: currentRequestData.model_id?.trim() || '', prompt: currentRequestData.prompt?.trim() || '',
			negative_prompt: currentRequestData.negative_prompt?.trim() || null, height: Number(currentRequestData.height) || 512,
			width: Number(currentRequestData.width) || 512, num_inference_steps: Number(currentRequestData.num_inference_steps) || 25,
			guidance_scale: Number(currentRequestData.guidance_scale) || 7.0, num_outputs: Number(currentRequestData.num_outputs) || 1,
			seed: currentRequestData.seed === null || isNaN(Number(currentRequestData.seed)) ? null : Math.floor(Number(currentRequestData.seed)),
			scheduler_id: currentRequestData.scheduler_id?.trim() || null, custom_sigmas: currentRequestData.custom_sigmas
		};
		// Validation
		if (!payload.model_id) { error = 'Model ID cannot be empty.'; isLoading = false; return; }
		if (!payload.prompt) { error = 'Prompt cannot be empty.'; isLoading = false; return; }
        const checkHeight = payload.height ?? 512; const checkWidth = payload.width ?? 512;
		if (checkHeight <= 0 || checkWidth <= 0 || checkHeight % 8 !== 0 || checkWidth % 8 !== 0) { error = 'Height/Width must be positive multiples of 8.'; isLoading = false; return; }
        if ((payload.num_inference_steps ?? 0) <=0) { error = 'Steps must be positive.'; isLoading = false; return; }
        if ((payload.num_outputs ?? 0) <=0) { error = 'Number of outputs must be positive.'; isLoading = false; return; }

		console.log('Sending payload:', payload);
		try {
			const response = await post<Text2ImageResponse>('/generate/text-to-image', payload);
			if (response && response.images) { results = response.images; } else { results = []; console.warn("API response OK but missing 'images' array:", response); }
			console.log('Generation successful:', response);
		} catch (e: unknown) {
			console.error('Generation failed:', e);
			if (isApiError(e)) {
                 if (typeof e.detail === 'string') { error = `Error ${e.status}: ${e.detail}`; }
                 else if (Array.isArray(e.detail)) { error = `Validation Error: ${e.detail.map(d => `${d.loc?.slice(-1) ?? 'field'}: ${d.msg}`).join(', ')}`; }
                 else if (typeof e.detail === 'object' && e.detail !== null) { error = `Error ${e.status}: ${JSON.stringify(e.detail)}`; }
                 else { error = `API Error ${e.status}: ${e.statusText || 'Unknown API error'}`; }
			} else if (e instanceof Error) { error = `Error: ${e.message}`; } else { error = `An unexpected error occurred: ${String(e)}`; }
			results = null;
		} finally { isLoading = false; }
     }

    // --- Handlers for SeedInput events ---
    function handleSeedChange(event: CustomEvent<number | null>) { t2iFormStore.update(s => ({...s, seed: event.detail })); }
    function handleSeedRandomize() { t2iFormStore.update((current) => ({ ...current, seed: Math.floor(Math.random() * 2 ** 32) })); }
    function handleSeedClear() { t2iFormStore.update((current) => ({ ...current, seed: null })); }

    // --- Save Preset Logic ---
    async function saveCurrentAsPreset() {
        const name = prompt("Enter a name for this preset:");
        if (!name || name.trim() === '') { alert("Preset name cannot be empty."); return; }
        const currentSettings = $t2iFormStore;
        if (!currentSettings.model_id) { alert("Cannot save preset without a selected Model ID."); return; }
        const presetSettings: PresetSetting = {
             negative_prompt: currentSettings.negative_prompt, height: currentSettings.height, width: currentSettings.width,
             num_inference_steps: currentSettings.num_inference_steps, guidance_scale: currentSettings.guidance_scale,
             scheduler_id: currentSettings.scheduler_id,
        };
        const payload: PresetCreate = {
             model_identifier: currentSettings.model_id, preset_name: name.trim(), settings: presetSettings,
             description: `Saved from T2I form on ${new Date().toLocaleDateString()}`
        };
        console.log("Saving preset:", payload);
        const wasLoading = isLoading; isLoading = true; const oldError = error; error = null;
        try {
            await post<PresetRead>('/presets', payload); alert(`Preset "${name}" saved successfully!`);
            fetchPresets(currentSettings.model_id); // Refetch presets for current model
        } catch (err: unknown) {
             console.error("Failed to save preset:", err);
             if (isApiError(err)) { error = `Save Preset Error: ${err.status} - ${JSON.stringify(err.detail || err.statusText)}`; }
             else if (err instanceof Error) { error = `Save Preset Error: ${err.message}`; }
             else { error = `Save Preset Error: ${String(err)}`; }
        } finally { isLoading = false; if(error === null && !wasLoading) error = oldError; } // Restore non-save error only if not already loading
    }
</script>

<!-- Component Template (Uses Shared Components & Explicit Attributes) -->
<div
	class="max-w-2xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700"
>
	<h2 class="text-2xl font-semibold mb-6 text-center text-gray-800 dark:text-gray-200">
		Text-to-Image Generation
	</h2>

	<form on:submit|preventDefault={generateImage} class="space-y-5">
		<!-- Use Shared ModelSelector -->
		<ModelSelector
			id="model_id_t2i"
			label="Model ID:"
			availableModels={availableModels}
			selectedModelId={$t2iFormStore.model_id}
			on:change={(e) => t2iFormStore.update(s => ({ ...s, model_id: e.detail }))}
		/>

		<!-- Preset Selector -->
		<div>
			<label for="preset_selector_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Load Preset:</label
			>
			<div class="flex items-center space-x-2">
				<select
					id="preset_selector_t2i"
                    bind:value={selectedPresetId} 
					on:change={(e) => applyPreset((e.target as HTMLSelectElement).value)}
					disabled={isFetchingPresets || availablePresets.length === 0}
					class="flex-grow w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
				>
                    <!-- Ensure the value is treated as string for comparison below -->
					<option value="none" disabled={selectedPresetId !== 'none'}>-- Select a Preset --</option>
					{#each availablePresets as preset (preset.id)}
						<option value={preset.id}>
							{preset.preset_name} {preset.is_user_defined ? '' : '[System]'}
						</option>
					{/each}
				</select>
				{#if isFetchingPresets}
					<div class="w-5 h-5 border-t-2 border-b-2 border-indigo-500 rounded-full animate-spin"></div>
				{/if}
			</div>
		</div>

		<!-- Use Shared SchedulerSelector -->
		<SchedulerSelector
			id="scheduler_id_t2i"
			label="Scheduler:"
			availableSchedulers={availableSchedulers}
			selectedSchedulerId={$t2iFormStore.scheduler_id}
			on:change={(e) => t2iFormStore.update(s => ({ ...s, scheduler_id: e.detail }))}
		/>

		<!-- Prompt -->
		<div>
			<label for="prompt_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Prompt:</label
			>
			<textarea
				id="prompt_t2i" rows="4" bind:value={$t2iFormStore.prompt} required
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			></textarea>
		</div>
		<!-- Negative Prompt -->
		<div>
			<label
				for="negative_prompt_t2i"
				class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Negative Prompt:</label
			>
			<textarea
				id="negative_prompt_t2i" rows="3" bind:value={$t2iFormStore.negative_prompt}
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			></textarea>
		</div>
		<!-- Dimensions -->
		<div class="grid grid-cols-2 gap-4">
			<div>
				<label for="width_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Width:</label
				>
				<input
					type="number" id="width_t2i" bind:value={$t2iFormStore.width}
					min="64" step="8" required
					class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
				/>
			</div>
			<div>
				<label for="height_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Height:</label
				>
				<input
					type="number" id="height_t2i" bind:value={$t2iFormStore.height}
					min="64" step="8" required
					class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
				/>
			</div>
		</div>
		<!-- Steps & CFG -->
		<div class="grid grid-cols-2 gap-4">
			<div>
				<label for="steps_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Steps: <span class="font-mono text-indigo-600 dark:text-indigo-400">{$t2iFormStore.num_inference_steps}</span
					></label
				>
				<input
					type="range" id="steps_t2i" bind:value={$t2iFormStore.num_inference_steps}
					min="1" max="150" step="1"
					class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
				/>
			</div>
			<div>
				<label for="cfg_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
					>Guidance (CFG): <span class="font-mono text-indigo-600 dark:text-indigo-400">{$t2iFormStore.guidance_scale?.toFixed(1)}</span
					></label
				>
				<input
					type="range" id="cfg_t2i" bind:value={$t2iFormStore.guidance_scale}
					min="0" max="20" step="0.5"
					class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
				/>
			</div>
		</div>

		<!-- Use Shared SeedInput -->
		<SeedInput
			id="seed_t2i"
			label="Seed:"
			seedValue={$t2iFormStore.seed}
			on:change={handleSeedChange}
			on:randomize={handleSeedRandomize}
			on:clear={handleSeedClear}
		/>

		<!-- Custom Sigmas (Optional) -->
		<div>
			<label
				for="custom_sigmas_t2i"
				class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Custom Sigmas (Optional):</label
			>
			<textarea
				id="custom_sigmas_t2i" rows="2" bind:value={customSigmasString}
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
				for="num_outputs_t2i"
				class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300"
				>Number of Images:</label
			>
			<input
				type="number" id="num_outputs_t2i" bind:value={$t2iFormStore.num_outputs}
				min="1" max="4" step="1" required
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			/>
		</div>

		<!-- Use Shared SubmitButton -->
		<div class="pt-4">
			 <SubmitButton
                isLoading={isLoading}
                buttonText="Generate Image"
                loadingText="Generating..."
             />
		</div>

        <!-- Save Preset Button -->
         <div class="pt-1">
            <button
				type="button"
                on:click={saveCurrentAsPreset}
				disabled={isLoading || !$t2iFormStore.model_id}
				class="w-full inline-flex justify-center py-2 px-4 border border-gray-300 dark:border-gray-600 shadow-sm text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out"
			>
				💾 Save Current Settings as Preset...
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