<!-- frontend/src/lib/components/ImageToImageForm.svelte -->
<script lang="ts">
	// --- Imports ---
	import { get, post, isApiError, type ApiError } from '$lib/apiClient';
	import type { ImageToImageRequest, ImageToImageResponse, ImageResult, PresetRead, PresetSetting, PresetCreate } from '$lib/types';
	import { i2iFormStore } from '$lib/stores/i2iFormStore'; // Correct store
	import { onMount, onDestroy, beforeUpdate, createEventDispatcher } from 'svelte';
    // Shared Components
	import ModelSelector from './shared/ModelSelector.svelte';
	import SchedulerSelector from './shared/SchedulerSelector.svelte';
	import SeedInput from './shared/SeedInput.svelte';
	import SubmitButton from './shared/SubmitButton.svelte';
    // Utils
    import { fetchPresetsForModel, applyPresetToStore, savePresetFromStore } from '$lib/utils/presetManager';

    // --- Dispatcher ---
    const dispatch = createEventDispatcher<{success: ImageResult[]}>();

	// --- Component State ---
	let isLoading = false;
	let error: string | null = null;
	let results: ImageResult[] | null = null;
	let uploadedFileName: string | null = null;
	let imagePreviewUrl: string | null = null;

	// Dynamic Selector State
	let availableModels: string[] = [];
	let availableSchedulers: string[] = ['Default'];
	let availableLoras: string[] = [];
	let availableEmbeddings: string[] = [];
	let isFetchingSystemInfo = false;

	// Preset State
	let availablePresets: PresetRead[] = [];
	let selectedPresetId: number | string = 'none';
	let isFetchingPresets = false;
    let currentModelId: string | null | undefined = null;

    // Selected LoRAs/Embeddings State
    let selectedLoras: { name: string; weight: number }[] = [];
    let selectedEmbeddings: string[] = [];


    // --- Fetch Dynamic Data ---
    async function fetchSystemInfo(): Promise<void> {
        if (isFetchingSystemInfo) return; isFetchingSystemInfo = true;
        try {
            const [models, schedulers, loras, embeddings] = await Promise.all([
                get<string[]>('/system/models/diffusion').catch(e => { console.error("Fetch models error:", e); return []; }),
                get<string[]>('/system/schedulers').catch(e => { console.error("Fetch schedulers error:", e); return ['Default']; }),
                get<string[]>('/system/models/loras').catch(e => { console.error("Fetch loras error:", e); return []; }),
                get<string[]>('/system/models/embeddings').catch(e => { console.error("Fetch embeddings error:", e); return []; })
            ]);
            availableModels = models; availableSchedulers = ['Default', ...schedulers.filter(s => s !== 'Default')];
            availableLoras = loras; availableEmbeddings = embeddings;
            // Validation of stored values
            if (models.length > 0) {
                const currentModelIdInStore: string = $i2iFormStore.model_id; // Use correct store
                let isModelValid = currentModelIdInStore && typeof currentModelIdInStore === 'string' && currentModelIdInStore.trim() !== '' && models.includes(currentModelIdInStore);
                if (!isModelValid) { console.warn("Stored model '" + (currentModelIdInStore || "[EMPTY]") + "' not found."); }
            }
            const currentSchedulerIdInStore = $i2iFormStore.scheduler_id; // Use correct store
            let isSchedulerValid = currentSchedulerIdInStore === null || (currentSchedulerIdInStore && availableSchedulers.includes(currentSchedulerIdInStore));
            if (!isSchedulerValid && currentSchedulerIdInStore !== null) {
                 console.warn("Stored scheduler '" + (currentSchedulerIdInStore || "[EMPTY]") + "' invalid. Resetting.");
                 i2iFormStore.update(s => ({...s, scheduler_id: null})); // Use correct store
            }
        } catch (err) { console.error("Error fetching system info:", err); error = "Failed to load options."; availableModels = [$i2iFormStore.model_id || '']; }
        finally { isFetchingSystemInfo = false; }
    }

    // --- Preset Handling ---
    async function handleModelChangeForPresets(modelId: string | null | undefined): Promise<void> {
        isFetchingPresets = true; availablePresets = await fetchPresetsForModel(modelId);
        selectedPresetId = 'none'; isFetchingPresets = false;
    }
    function handlePresetSelectionChange(event: Event): void {
        const target = event.target as HTMLSelectElement; applyPreset(target.value);
    }
    function applyPreset(presetIdValue: number | string | null): void {
         const idToFind = typeof presetIdValue === 'string' ? parseInt(presetIdValue, 10) : presetIdValue;
         selectedPresetId = idToFind ?? 'none';
         if (idToFind === null || isNaN(idToFind)) { return; }
         const preset = availablePresets.find(p => p.id === idToFind);
         // Apply to the i2i store
         applyPresetToStore(preset, i2iFormStore); // Pass correct store
         if (!preset) { console.warn("Preset with ID " + idToFind + " not found."); }
    }
    async function handleSavePreset(): Promise<void> {
        const wasLoading = isLoading; isLoading = true; const oldError = error; error = null;
        try {
            const settingsExtractor = (current: ImageToImageRequest): PresetSetting => ({ // Use I2I type
                negative_prompt: current.negative_prompt, height: current.height, width: current.width,
                num_inference_steps: current.num_inference_steps, guidance_scale: current.guidance_scale,
                scheduler_id: current.scheduler_id, strength: current.strength, // Include strength
            });
            // Use the i2i store
            const newPreset = await savePresetFromStore(i2iFormStore, settingsExtractor);
            if (newPreset) { await handleModelChangeForPresets($i2iFormStore.model_id); } // Refetch for current model
        } catch (e) { console.error("Error in handleSavePreset:", e); }
        finally { isLoading = false; if(error === null && !wasLoading) error = oldError; }
    }

	// --- File Handling ---
	async function handleFileChange(event: Event): Promise<void> {
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
			reader.onerror = () => {
                error = 'Error reading file.'; i2iFormStore.update(current => ({ ...current, image_base64: '' }));
                uploadedFileName = null; if(imagePreviewUrl) URL.revokeObjectURL(imagePreviewUrl); imagePreviewUrl = null;
            };
			reader.readAsDataURL(file);
		} else { i2iFormStore.update(current => ({ ...current, image_base64: '' })); uploadedFileName = null; imagePreviewUrl = null; }
	}

    // --- Lifecycle ---
    onMount(() => {
        // Initialize preview if needed
        const currentBase64 = $i2iFormStore.image_base64;
		if (currentBase64) {
            try {
                const byteString = atob(currentBase64); const mimeString = 'image/png';
                const ab = new ArrayBuffer(byteString.length); const ia = new Uint8Array(ab);
                for (let i = 0; i < byteString.length; i++) { ia[i] = byteString.charCodeAt(i); }
                const blob = new Blob([ab], { type: mimeString });
                imagePreviewUrl = URL.createObjectURL(blob); uploadedFileName = "Image from previous session";
            } catch(e) { console.error("Restore preview failed:", e); i2iFormStore.update(s => ({ ...s, image_base64: '' })); imagePreviewUrl = null; uploadedFileName = null; }
        } else { imagePreviewUrl = null; uploadedFileName = null; }
        // Fetch system info then presets
        fetchSystemInfo().then(() => {
             currentModelId = $i2iFormStore.model_id;
             if(currentModelId) { handleModelChangeForPresets(currentModelId); }
        });
	});
    beforeUpdate(() => {
        // Refetch presets if model ID changes
        if ($i2iFormStore.model_id !== currentModelId) {
            currentModelId = $i2iFormStore.model_id;
            handleModelChangeForPresets(currentModelId);
            selectedLoras = []; selectedEmbeddings = []; // Clear LoRAs/Embeddings
        }
    });
	onDestroy(() => { if (imagePreviewUrl) { URL.revokeObjectURL(imagePreviewUrl); } });

	// --- Submit Handler ---
	async function generateImage(): Promise<void> {
		isLoading = true; error = null; results = null;
        const currentRequestData = $i2iFormStore;
		// Validation
		if (!currentRequestData.image_base64) { error = 'Please select an initial image file.'; isLoading = false; return; }
		if (!currentRequestData.model_id?.trim()) { error = 'Model ID cannot be empty.'; isLoading = false; return; }
		if (!currentRequestData.prompt?.trim()) { error = 'Prompt cannot be empty.'; isLoading = false; return; }
        const checkHeight = currentRequestData.height ?? 512; const checkWidth = currentRequestData.width ?? 512;
        if (checkHeight <= 0 || checkWidth <= 0 || checkHeight % 8 !== 0 || checkWidth % 8 !== 0) { error = 'Height/Width must be positive multiples of 8.'; isLoading = false; return; }
        const checkStrength = currentRequestData.strength ?? 0.75;
		if (checkStrength < 0 || checkStrength > 1) { error = 'Strength must be between 0.0 and 1.0.'; isLoading = false; return; }
        if ((currentRequestData.num_inference_steps ?? 0) <=0) { error = 'Steps must be positive.'; isLoading = false; return; }
        if ((currentRequestData.num_outputs ?? 0) <=0) { error = 'Number of outputs must be positive.'; isLoading = false; return; }

		// Create payload
		const payload: ImageToImageRequest & { loras?: typeof selectedLoras, embeddings?: string[] } = {
            model_id: currentRequestData.model_id.trim(), prompt: currentRequestData.prompt.trim(),
            negative_prompt: currentRequestData.negative_prompt?.trim() || null, height: Number(currentRequestData.height) || 512,
			width: Number(currentRequestData.width) || 512, num_inference_steps: Number(currentRequestData.num_inference_steps) || 35,
			guidance_scale: Number(currentRequestData.guidance_scale) || 7.0, strength: Number(currentRequestData.strength) || 0.75,
			num_outputs: Number(currentRequestData.num_outputs) || 1,
			seed: currentRequestData.seed === null || isNaN(Number(currentRequestData.seed)) ? null : Math.floor(Number(currentRequestData.seed)),
			scheduler_id: currentRequestData.scheduler_id?.trim() || null, image_base64: currentRequestData.image_base64,
            loras: selectedLoras.filter(l => l.name), embeddings: selectedEmbeddings
		};

		try {
			const response = await post<ImageToImageResponse>('/generate/image-to-image', payload);
			if (response && response.images) { results = response.images; dispatch('success', results); }
            else { results = []; dispatch('success', []); }
		} catch (e: unknown) {
            console.error('Generation failed:', e);
			if (isApiError(e)) {
                 if (typeof e.detail === 'string') { error = `Error ${e.status}: ${e.detail}`; }
                 else if (Array.isArray(e.detail)) { error = `Validation Error: ${e.detail.map(d => `${d.loc?.slice(-1) ?? 'field'}: ${d.msg}`).join(', ')}`; }
                 else if (typeof e.detail === 'object' && e.detail !== null) { error = `Error ${e.status}: ${JSON.stringify(e.detail)}`; }
                 else { error = `API Error ${e.status}: ${e.statusText || 'Unknown API error'}`; }
			} else if (e instanceof Error) { error = `Error: ${e.message}`; }
            else { error = `An unexpected error occurred: ${String(e)}`; }
			results = null;
        } finally { isLoading = false; }
	}

	// --- Handlers for SeedInput events (update the i2i store) ---
	function handleSeedChange(event: CustomEvent<number | null>): void { i2iFormStore.update(s => ({...s, seed: event.detail })); }
    function handleSeedRandomize(): void { i2iFormStore.update((current) => ({ ...current, seed: Math.floor(Math.random() * 2 ** 32) })); }
    function handleSeedClear(): void { i2iFormStore.update((current) => ({ ...current, seed: null })); }

    // --- LoRA Management Helpers ---
    function addLora(): void { selectedLoras = [...selectedLoras, { name: '', weight: 0.8 }]; }
    function removeLora(index: number): void { selectedLoras = selectedLoras.filter((_, i) => i !== index); }
    function updateLoraName(index: number, event: Event & { currentTarget: EventTarget & HTMLSelectElement }): void { selectedLoras[index].name = event.currentTarget.value; selectedLoras = selectedLoras; }
    function updateLoraWeight(index: number, event: Event & { currentTarget: EventTarget & HTMLInputElement }): void { selectedLoras[index].weight = parseFloat(event.currentTarget.value); selectedLoras = selectedLoras; }

    // --- Embedding Management Helpers ---
    function addEmbedding(): void { const name = prompt("Enter embedding name/trigger word:", ""); if (name && name.trim() !== '' && !selectedEmbeddings.includes(name.trim())) { selectedEmbeddings = [...selectedEmbeddings, name.trim()]; } }
    function removeEmbedding(index: number): void { selectedEmbeddings = selectedEmbeddings.filter((_, i) => i !== index); }

</script>

<!-- Component Template -->
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
		<ModelSelector id="model_id_i2i" label="Model ID:" availableModels={availableModels} selectedModelId={$i2iFormStore.model_id} on:change={(e) => i2iFormStore.update(s => ({ ...s, model_id: e.detail }))} disabled={isFetchingSystemInfo} />

        <!-- Preset Selector -->
        <div>
			<label for="preset_selector_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Load Preset:</label>
            <div class="flex items-center space-x-2">
                <select id="preset_selector_i2i" bind:value={selectedPresetId} on:change={handlePresetSelectionChange} disabled={isFetchingPresets || availablePresets.length === 0} class="flex-grow w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50">
                    <option value="none" disabled={selectedPresetId !== 'none'}>-- Select a Preset --</option>
                    {#each availablePresets as preset (preset.id)} <option value={preset.id}>{preset.preset_name} {preset.is_user_defined ? '' : '[System]'}</option> {/each}
                </select>
                {#if isFetchingPresets}<div class="w-5 h-5 border-t-2 border-b-2 border-indigo-500 rounded-full animate-spin"></div>{/if}
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

		<!-- LoRA Selection -->
        <div class="border dark:border-gray-600 rounded p-3 space-y-2">
            <div class="block text-sm font-medium text-gray-700 dark:text-gray-300">LoRAs:</div>
            {#if Array.isArray(availableLoras) && availableLoras.length === 0 && !isFetchingSystemInfo}
                 <p class="text-xs text-gray-500 dark:text-gray-400">No local LoRAs found.</p>
            {:else if isFetchingSystemInfo && (!availableLoras || availableLoras.length === 0)}
                 <p class="text-xs text-gray-500 dark:text-gray-400">Loading LoRAs...</p>
            {:else if Array.isArray(availableLoras)}
                {#each selectedLoras as lora, index (index)}
                    <div class="flex items-center space-x-2 text-sm">
                        <select class="flex-grow px-2 py-1 border text-xs border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                            value={lora.name} on:change={(e) => updateLoraName(index, e)} required>
                            <option value="" disabled>-- Select LoRA --</option>
                            {#each availableLoras as loraName (loraName)} <option value={loraName}>{loraName}</option> {/each}
                        </select>
                        <input type="range" class="w-20 h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"
                            min="-1.0" max="2.0" step="0.05" value={lora.weight} on:input={(e) => updateLoraWeight(index, e)} />
                        <span class="text-xs font-mono w-8 text-right">{lora.weight.toFixed(2)}</span>
                        <button type="button" on:click={() => removeLora(index)} title="Remove LoRA" class="p-1 bg-red-100 text-red-700 dark:bg-red-800 dark:text-red-200 rounded hover:bg-red-200 dark:hover:bg-red-700 text-xs">X</button>
                    </div>
                {/each}
                <button type="button" on:click={addLora} disabled={!Array.isArray(availableLoras) || availableLoras.length === 0} class="text-xs px-2 py-1 border border-dashed border-gray-400 dark:border-gray-500 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed">
                    + Add LoRA
                </button>
            {/if}
        </div>

         <!-- Embedding Selection -->
        <div class="border dark:border-gray-600 rounded p-3 space-y-2">
             <div class="block text-sm font-medium text-gray-700 dark:text-gray-300">Embeddings (Textual Inversion):</div>
             {#if isFetchingSystemInfo && (!availableEmbeddings || availableEmbeddings.length === 0)}
                 <p class="text-xs text-gray-500 dark:text-gray-400">Loading Embeddings...</p>
             {:else if Array.isArray(availableEmbeddings)}
                <div class="flex flex-wrap gap-1">
                    {#each selectedEmbeddings as embedding, index (embedding)}
                         <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
                            {embedding}
                            <button type="button" class="ml-1 -mr-0.5 flex-shrink-0 h-4 w-4 rounded-full inline-flex items-center justify-center text-indigo-400 hover:bg-indigo-200 hover:text-indigo-500 focus:outline-none focus:bg-indigo-500 focus:text-white dark:hover:bg-indigo-800"
                                on:click={() => removeEmbedding(index)} title="Remove Embedding">
                                <span class="sr-only">Remove</span> <svg class="h-2 w-2" stroke="currentColor" fill="none" viewBox="0 0 8 8"><path stroke-linecap="round" stroke-width="1.5" d="M1 1l6 6m0-6L1 7" /></svg>
                            </button>
                         </span>
                    {/each}
                </div>
                 <button type="button" on:click={addEmbedding} class="text-xs px-2 py-1 border border-dashed border-gray-400 dark:border-gray-500 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700">
                     + Add Embedding (Manual Entry)
                 </button>
                 {#if availableEmbeddings.length > 0}
                     <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">Found: {availableEmbeddings.join(', ')}. Enter trigger words manually.</p>
                 {:else if !isFetchingSystemInfo}
                     <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">No local embeddings found.</p>
                 {/if}
             {/if}
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
        <SchedulerSelector id="scheduler_id_i2i" label="Scheduler:" availableSchedulers={availableSchedulers} selectedSchedulerId={$i2iFormStore.scheduler_id} on:change={(e) => i2iFormStore.update(s => ({...s, scheduler_id: e.detail }))} disabled={isFetchingSystemInfo} />

		<!-- USE SHARED SeedInput -->
        <SeedInput id="seed_i2i" label="Seed:" seedValue={$i2iFormStore.seed} on:change={handleSeedChange} on:randomize={handleSeedRandomize} on:clear={handleSeedClear}/>

		<!-- Number of Outputs -->
		<div>
            <label for="num_outputs_i2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Number of Images:</label>
            <input type="number" id="num_outputs_i2i" bind:value={$i2iFormStore.num_outputs} min="1" max="4" step="1" required class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white" />
        </div>

		<!-- USE SHARED SubmitButton -->
		<div class="pt-4">
            <SubmitButton isLoading={isLoading} buttonText="Generate Image Variation" loadingText="Generating..." />
        </div>

        <!-- Save Preset Button -->
         <div class="pt-1">
            <button type="button" on:click={handleSavePreset} disabled={isLoading || !$i2iFormStore.model_id} class="w-full inline-flex justify-center py-2 px-4 border border-gray-300 dark:border-gray-600 shadow-sm text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out">
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
				{#each results as result, i (result.seed)}
					<div class="border dark:border-gray-700 rounded-lg overflow-hidden shadow bg-gray-50 dark:bg-gray-750">
						<img src="data:image/png;base64,{result.image_base64}" alt={`Image-to-image result ${i + 1} for the prompt`} class="w-full h-auto object-contain" />
						<div class="p-2"> <p class="text-xs text-gray-600 dark:text-gray-400">Seed: {result.seed}</p> </div>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>