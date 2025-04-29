<!-- frontend/src/lib/components/TextToImageForm.svelte -->
<script lang="ts">
	// --- Imports ---
	import { get, post, isApiError, type ApiError } from '$lib/apiClient';
	import type { Text2ImageRequest, Text2ImageResponse, ImageResult, PresetRead, PresetSetting, PresetCreate } from '$lib/types';
	import { t2iFormStore } from '$lib/stores/t2iFormStore';
	import { onMount, beforeUpdate, createEventDispatcher } from 'svelte';
    import ModelSelector from './shared/ModelSelector.svelte';
    import SchedulerSelector from './shared/SchedulerSelector.svelte';
    import SeedInput from './shared/SeedInput.svelte';
    import SubmitButton from './shared/SubmitButton.svelte';
    import { fetchPresetsForModel, applyPresetToStore, savePresetFromStore } from '$lib/utils/presetManager';

    // --- Dispatcher ---
    const dispatch = createEventDispatcher<{success: ImageResult[]}>();

	// --- Component State ---
	let isLoading = false; // Tracks API call in progress
	let error: string | null = null; // Stores error messages for display
	let results: ImageResult[] | null = null; // Stores successful generation results
	let customSigmasString: string = ''; // Bound to the custom sigmas textarea

	// Dynamic Selector State
	let availableModels: string[] = []; // List fetched from API
	let availableSchedulers: string[] = ['Default']; // List fetched from API, starts with Default
	let availableLoras: string[] = []; // List fetched from API
	let availableEmbeddings: string[] = []; // List fetched from API
	let isFetchingSystemInfo = false; // Loading indicator for selector options

	// Preset State
	let availablePresets: PresetRead[] = []; // List fetched from API based on model
    let selectedPresetId: number | string = 'none'; // Tracks the <select> UI value ('none' or preset.id as string)
    let isFetchingPresets = false; // Loading indicator for presets
    let currentModelId: string | null | undefined = null; // Tracks store's model_id to detect changes

    // Selected LoRAs/Embeddings State
    let selectedLoras: { name: string; weight: number }[] = []; // Array for selected LoRAs
    let selectedEmbeddings: string[] = []; // Array for selected Embeddings

	// --- Reactive Parsing for Custom Sigmas ---
	$: {
		if (customSigmasString && customSigmasString.trim() !== '') {
			try {
				const parsedSigmas = customSigmasString.split(/[\s,]+/).filter((s) => s.trim() !== '').map((s) => parseFloat(s));
				if (parsedSigmas.every((n) => !isNaN(n))) {
                    if (JSON.stringify($t2iFormStore.custom_sigmas) !== JSON.stringify(parsedSigmas)) {
                        t2iFormStore.update((current) => ({ ...current, custom_sigmas: parsedSigmas }));
                    }
				} else {
                    if ($t2iFormStore.custom_sigmas !== null) {
                        t2iFormStore.update((current) => ({ ...current, custom_sigmas: null }));
                    }
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
            // Validate stored model
            if (models.length > 0) {
                const currentModelIdInStore: string = $t2iFormStore.model_id;
                let isModelValid = currentModelIdInStore && typeof currentModelIdInStore === 'string' && currentModelIdInStore.trim() !== '' && models.includes(currentModelIdInStore);
                if (!isModelValid) { console.warn("Stored model '" + (currentModelIdInStore || "[EMPTY]") + "' not found/valid."); }
            }
            // Validate stored scheduler
            const currentSchedulerIdInStore = $t2iFormStore.scheduler_id;
            let isSchedulerValid = currentSchedulerIdInStore === null || (currentSchedulerIdInStore && availableSchedulers.includes(currentSchedulerIdInStore));
            if (!isSchedulerValid && currentSchedulerIdInStore !== null) {
                console.warn("Stored scheduler '" + (currentSchedulerIdInStore || "[EMPTY]") + "' not valid. Resetting.");
                t2iFormStore.update(s => ({...s, scheduler_id: null}));
            }
        } catch (err) { console.error("Error fetching system info:", err); error = "Failed to load options."; availableModels = [$t2iFormStore.model_id || '']; }
        finally { isFetchingSystemInfo = false; }
    }

    // --- Preset Handling ---
    async function handleModelChangeForPresets(modelId: string | null | undefined): Promise<void> {
        isFetchingPresets = true;
        availablePresets = await fetchPresetsForModel(modelId); // Use util
        selectedPresetId = 'none';
        isFetchingPresets = false;
    }
    function handlePresetSelectionChange(event: Event): void {
         const target = event.target as HTMLSelectElement;
         applyPreset(target.value); // Pass selected value (string ID or 'none')
    }
    function applyPreset(presetIdValue: number | string | null): void {
         const idToFind = typeof presetIdValue === 'string' ? parseInt(presetIdValue, 10) : presetIdValue;
         selectedPresetId = idToFind ?? 'none';
         if (idToFind === null || isNaN(idToFind)) { return; }
         const preset = availablePresets.find(p => p.id === idToFind);
         // Use utility to apply to store, provide callback to clear local sigma string
         applyPresetToStore(preset, t2iFormStore, () => {
             customSigmasString = $t2iFormStore.custom_sigmas?.join(', ') ?? '';
         });
         if (!preset) { console.warn("Preset with ID " + idToFind + " not found."); }
    }
    async function handleSavePreset(): Promise<void> {
        const wasLoading = isLoading; isLoading = true; const oldError = error; error = null;
        try {
            const settingsExtractor = (current: Text2ImageRequest): PresetSetting => ({
                negative_prompt: current.negative_prompt, height: current.height, width: current.width,
                num_inference_steps: current.num_inference_steps, guidance_scale: current.guidance_scale,
                scheduler_id: current.scheduler_id,
            });
            const newPreset = await savePresetFromStore(t2iFormStore, settingsExtractor);
            if (newPreset) { await handleModelChangeForPresets($t2iFormStore.model_id); }
        } catch (e) { console.error("Error in handleSavePreset:", e); } // Error already alerted by util
        finally { isLoading = false; if(error === null && !wasLoading) error = oldError; }
    }

    // --- Lifecycle ---
    onMount(() => {
		const currentSigmas = $t2iFormStore.custom_sigmas;
		customSigmasString = currentSigmas?.join(', ') ?? '';
        fetchSystemInfo().then(() => {
             currentModelId = $t2iFormStore.model_id;
             if(currentModelId) { handleModelChangeForPresets(currentModelId); }
        });
	});
    beforeUpdate(() => {
        if ($t2iFormStore.model_id !== currentModelId) {
            currentModelId = $t2iFormStore.model_id;
            handleModelChangeForPresets(currentModelId);
            selectedLoras = []; selectedEmbeddings = [];
        }
    });

	// --- Submit Handler ---
	async function generateImage(): Promise<void> {
		isLoading = true; error = null; results = null;
		const currentRequestData = $t2iFormStore;
		const payload: Text2ImageRequest & { loras?: typeof selectedLoras, embeddings?: string[] } = {
			model_id: currentRequestData.model_id?.trim() || '', prompt: currentRequestData.prompt?.trim() || '',
			negative_prompt: currentRequestData.negative_prompt?.trim() || null, height: Number(currentRequestData.height) || 512,
			width: Number(currentRequestData.width) || 512, num_inference_steps: Number(currentRequestData.num_inference_steps) || 25,
			guidance_scale: Number(currentRequestData.guidance_scale) || 7.0, num_outputs: Number(currentRequestData.num_outputs) || 1,
			seed: currentRequestData.seed === null || isNaN(Number(currentRequestData.seed)) ? null : Math.floor(Number(currentRequestData.seed)),
			scheduler_id: currentRequestData.scheduler_id?.trim() || null, custom_sigmas: currentRequestData.custom_sigmas,
            loras: selectedLoras.filter(l => l.name), embeddings: selectedEmbeddings
		};
		// Validation
		if (!payload.model_id) { error = 'Model ID cannot be empty.'; isLoading = false; return; }
		if (!payload.prompt) { error = 'Prompt cannot be empty.'; isLoading = false; return; }
        const checkHeight = payload.height ?? 512; const checkWidth = payload.width ?? 512;
		if (checkHeight <= 0 || checkWidth <= 0 || checkHeight % 8 !== 0 || checkWidth % 8 !== 0) { error = 'Height/Width must be positive multiples of 8.'; isLoading = false; return; }
        if ((payload.num_inference_steps ?? 0) <=0) { error = 'Steps must be positive.'; isLoading = false; return; }
        if ((payload.num_outputs ?? 0) <=0) { error = 'Number of outputs must be positive.'; isLoading = false; return; }

		try {
			const response = await post<Text2ImageResponse>('/generate/text-to-image', payload);
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

    // --- Handlers for SeedInput events ---
    function handleSeedChange(event: CustomEvent<number | null>): void { t2iFormStore.update(s => ({...s, seed: event.detail })); }
    function handleSeedRandomize(): void { t2iFormStore.update((current) => ({ ...current, seed: Math.floor(Math.random() * 2 ** 32) })); }
    function handleSeedClear(): void { t2iFormStore.update((current) => ({ ...current, seed: null })); }

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
		Text-to-Image Generation
	</h2>

	<form on:submit|preventDefault={generateImage} class="space-y-5">
		<!-- Model Selector -->
		<ModelSelector
			id="model_id_t2i"
            label="Model ID:"
			availableModels={availableModels}
			selectedModelId={$t2iFormStore.model_id}
			on:change={(e) => t2iFormStore.update(s => ({ ...s, model_id: e.detail }))}
            disabled={isFetchingSystemInfo}
		/>

		<!-- Preset Selector -->
		<div>
			<label for="preset_selector_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Load Preset:</label>
            <div class="flex items-center space-x-2">
                <select
                    id="preset_selector_t2i"
                    bind:value={selectedPresetId}
					on:change={handlePresetSelectionChange}
					disabled={isFetchingPresets || availablePresets.length === 0}
					class="flex-grow w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white disabled:opacity-50"
				>
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

		<!-- Scheduler Selector -->
		<SchedulerSelector
			id="scheduler_id_t2i"
            label="Scheduler:"
			availableSchedulers={availableSchedulers}
			selectedSchedulerId={$t2iFormStore.scheduler_id}
			on:change={(e) => t2iFormStore.update(s => ({ ...s, scheduler_id: e.detail }))}
            disabled={isFetchingSystemInfo}
		/>

		<!-- Prompt -->
		<div>
			<label for="prompt_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Prompt:</label>
			<textarea
				id="prompt_t2i" rows="4" bind:value={$t2iFormStore.prompt} required
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			></textarea>
		</div>
		<!-- Negative Prompt -->
		<div>
			<label for="negative_prompt_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Negative Prompt:</label>
			<textarea
				id="negative_prompt_t2i" rows="3" bind:value={$t2iFormStore.negative_prompt}
				class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
			></textarea>
		</div>

        <!-- LoRA Selection -->
        <div class="border dark:border-gray-600 rounded p-3 space-y-2">
            <div class="block text-sm font-medium text-gray-700 dark:text-gray-300">LoRAs:</div>
            {#if availableLoras.length === 0 && !isFetchingSystemInfo}
                 <p class="text-xs text-gray-500 dark:text-gray-400">No local LoRAs found.</p>
            {:else if isFetchingSystemInfo}
                 <p class="text-xs text-gray-500 dark:text-gray-400">Loading LoRAs...</p>
            {:else}
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
                <button type="button" on:click={addLora} disabled={availableLoras.length === 0} class="text-xs px-2 py-1 border border-dashed border-gray-400 dark:border-gray-500 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed">
                    + Add LoRA
                </button>
            {/if}
        </div>

         <!-- Embedding Selection -->
        <div class="border dark:border-gray-600 rounded p-3 space-y-2">
             <div class="block text-sm font-medium text-gray-700 dark:text-gray-300">Embeddings (Textual Inversion):</div>
             {#if isFetchingSystemInfo && availableEmbeddings.length === 0}
                 <p class="text-xs text-gray-500 dark:text-gray-400">Loading Embeddings...</p>
             {:else}
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

		<!-- Dimensions -->
		<div class="grid grid-cols-2 gap-4">
			<div>
				<label for="width_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Width:</label>
				<input type="number" id="width_t2i" bind:value={$t2iFormStore.width} min="64" step="8" required class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"/>
			</div>
			<div>
				<label for="height_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Height:</label>
				<input type="number" id="height_t2i" bind:value={$t2iFormStore.height} min="64" step="8" required class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"/>
			</div>
		</div>
		<!-- Steps & CFG -->
		<div class="grid grid-cols-2 gap-4">
			<div>
				<label for="steps_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Steps: <span class="font-mono text-indigo-600 dark:text-indigo-400">{$t2iFormStore.num_inference_steps}</span></label>
				<input type="range" id="steps_t2i" bind:value={$t2iFormStore.num_inference_steps} min="1" max="150" step="1" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"/>
			</div>
			<div>
				<label for="cfg_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Guidance (CFG): <span class="font-mono text-indigo-600 dark:text-indigo-400">{$t2iFormStore.guidance_scale?.toFixed(1)}</span></label>
				<input type="range" id="cfg_t2i" bind:value={$t2iFormStore.guidance_scale} min="0" max="20" step="0.5" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-600"/>
			</div>
		</div>

		<!-- Use Shared SeedInput -->
		<SeedInput id="seed_t2i" label="Seed:" seedValue={$t2iFormStore.seed} on:change={handleSeedChange} on:randomize={handleSeedRandomize} on:clear={handleSeedClear}/>

		<!-- Custom Sigmas (Optional) -->
		<div>
			<label for="custom_sigmas_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Custom Sigmas (Optional):</label>
			<textarea id="custom_sigmas_t2i" rows="2" bind:value={customSigmasString} placeholder="e.g., 14.6, 8.4, 5.2, ..." class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white font-mono text-sm"></textarea>
			<p class="mt-1 text-xs text-gray-500 dark:text-gray-400"> Comma or space separated. Overrides 'Steps'. Backend only uses length now.</p>
		</div>
		<!-- Number of Outputs -->
		<div>
			<label for="num_outputs_t2i" class="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">Number of Images:</label>
			<input type="number" id="num_outputs_t2i" bind:value={$t2iFormStore.num_outputs} min="1" max="4" step="1" required class="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"/>
		</div>

		<!-- Use Shared SubmitButton -->
		<div class="pt-4">
			 <SubmitButton {isLoading} buttonText="Generate Image" loadingText="Generating..." />
		</div>

        <!-- Save Preset Button -->
         <div class="pt-1">
            <button type="button" on:click={handleSavePreset} disabled={isLoading || !$t2iFormStore.model_id} class="w-full inline-flex justify-center py-2 px-4 border border-gray-300 dark:border-gray-600 shadow-sm text-sm font-medium rounded-md text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out">
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
						<img src="data:image/png;base64,{result.image_base64}" alt={`Generation result ${i + 1} for the prompt`} class="w-full h-auto object-contain"/>
						<div class="p-2"> <p class="text-xs text-gray-600 dark:text-gray-400">Seed: {result.seed}</p> </div>
					</div>
				{/each}
			</div>
		</div>
	{/if}

</div>