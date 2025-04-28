<!-- frontend/src/lib/components/XYZGridForm.svelte -->
<script lang="ts">
	import { post, isApiError, type ApiError } from '$lib/apiClient';
	import type { XYZGridRequest, XYZGridResponse, DiffusionTaskBase, AxisParameter } from '$lib/types';
    import { createEventDispatcher } from 'svelte';

    const dispatch = createEventDispatcher<{submit: XYZGridResponse, error: string, loading: boolean}>();

    // --- State for the form ---
    let baseParams: DiffusionTaskBase = {
        model_id: 'realisticVisionV51_v51VAE.safetensors', // Default model
        prompt: 'photo of a {} cat astronaut, masterpiece', // {} placeholder optional for axis
        negative_prompt: 'ugly, blurry, deformed, text, words, low quality',
        height: 512,
        width: 512,
        num_inference_steps: 20,
        guidance_scale: 7.0,
        seed: 12345, // Starting seed
        custom_sigmas: null
    };

    // Available parameters for axes (extend this list as needed)
    const availableAxisParams = [
        'prompt', 'negative_prompt', 'seed', 'guidance_scale',
        'num_inference_steps', 'width', 'height'
        // TODO: Add LoRA weights, ControlNet weights, VAE etc. later
    ];

    // State for each axis
    let xAxisEnabled = true;
    let xAxisParam: string = 'guidance_scale';
    let xAxisValuesString: string = '5.0, 7.5, 10.0';

    let yAxisEnabled = false;
    let yAxisParam: string = 'seed';
    let yAxisValuesString: string = '10, 20, 30';

    let zAxisEnabled = false;
    let zAxisParam: string = 'num_inference_steps';
    let zAxisValuesString: string = '15, 25';

    let isLoading = false;
    let error: string | null = null;

    // --- Helper to parse axis values ---
    function parseAxisValues(inputString: string, paramType: string): any[] | null {
        if (!inputString || inputString.trim() === '') {
            return null;
        }
        try {
            const values = inputString
                .split(',') // Split primarily by comma
                .map(v => v.trim()) // Trim whitespace
                .filter(v => v !== ''); // Remove empty strings

            if (values.length === 0) return null;

            // Attempt type conversion based on known parameter types (basic)
            return values.map(val => {
                if (['seed', 'num_inference_steps', 'width', 'height'].includes(paramType)) {
                    const num = parseInt(val, 10);
                    if (isNaN(num)) throw new Error(`Invalid number for ${paramType}: ${val}`);
                    return num;
                } else if (['guidance_scale'].includes(paramType)) {
                     const num = parseFloat(val);
                     if (isNaN(num)) throw new Error(`Invalid float for ${paramType}: ${val}`);
                     return num;
                } else {
                    // Assume string for prompt, neg_prompt, etc.
                    return val;
                }
            });
        } catch (e: unknown) {
             console.error("Error parsing axis values:", e);
             error = (e instanceof Error) ? e.message : "Invalid format for axis values.";
             return null; // Indicate parsing failure
        }
    }


    // --- Form Submission ---
    async function generateGrid() {
        isLoading = true;
        error = null;
        dispatch('loading', true); // Notify parent we are loading

        // Construct axes based on enabled state and parsed values
        let x_axis: AxisParameter | null = null;
        let y_axis: AxisParameter | null = null;
        let z_axis: AxisParameter | null = null;
        let parseError = false;

        if (xAxisEnabled) {
            const values = parseAxisValues(xAxisValuesString, xAxisParam);
            if (values) {
                x_axis = { parameter: xAxisParam, values: values };
            } else {
                error = error || `Invalid values for X axis (${xAxisParam}).`; // Keep first parse error
                parseError = true;
            }
        }
        if (yAxisEnabled) {
             const values = parseAxisValues(yAxisValuesString, yAxisParam);
             if (values) {
                y_axis = { parameter: yAxisParam, values: values };
             } else {
                error = error || `Invalid values for Y axis (${yAxisParam}).`;
                parseError = true;
             }
        }
        if (zAxisEnabled) {
             const values = parseAxisValues(zAxisValuesString, zAxisParam);
             if (values) {
                 z_axis = { parameter: zAxisParam, values: values };
             } else {
                error = error || `Invalid values for Z axis (${zAxisParam}).`;
                parseError = true;
             }
        }

        // Check for duplicate axes parameters
        const activeAxesParams = [x_axis?.parameter, y_axis?.parameter, z_axis?.parameter].filter(p => p);
        if (activeAxesParams.length !== new Set(activeAxesParams).size) {
             error = "Cannot use the same parameter for multiple axes.";
             parseError = true;
        }

        if (parseError || (!x_axis && !y_axis && !z_axis)) {
            error = error || "At least one axis must be enabled with valid values.";
            isLoading = false;
            dispatch('loading', false);
            dispatch('error', error);
            return; // Stop if parsing failed or no axes active
        }

        const payload: XYZGridRequest = {
            base_parameters: { // Ensure base params are correctly typed
                ...baseParams,
                height: Number(baseParams.height) || 512,
                width: Number(baseParams.width) || 512,
                num_inference_steps: Number(baseParams.num_inference_steps) || 25,
                guidance_scale: Number(baseParams.guidance_scale) || 7.0,
                seed: baseParams.seed === null || isNaN(Number(baseParams.seed)) ? null : Math.floor(Number(baseParams.seed)),
                custom_sigmas: baseParams.custom_sigmas?.filter(n=>!isNaN(n)) ?? null // Ensure numbers in sigmas
            },
            x_axis: x_axis,
            y_axis: y_axis,
            z_axis: z_axis
        };

        console.log("Sending XYZ Grid payload:", payload); // Debug

        try {
            const response = await post<XYZGridResponse>('/generate/xyz-grid', payload);
            console.log('XYZ Grid successful:', response);
            dispatch('submit', response); // Send results to parent
        } catch (e: unknown) {
             console.error('XYZ Grid failed:', e);
             let errorMsg: string;
             if (isApiError(e)) {
                 errorMsg = `API Error (${e.status}): ${JSON.stringify(e.detail || e.statusText)}`;
             } else if (e instanceof Error) {
                 errorMsg = `Error: ${e.message}`;
             } else {
                 errorMsg = 'An unknown error occurred during grid generation.';
             }
             error = errorMsg;
             dispatch('error', errorMsg); // Notify parent of error
        } finally {
            isLoading = false;
            dispatch('loading', false); // Notify parent loading finished
        }
    }

</script>

<div class="max-w-2xl mx-auto p-6 bg-white dark:bg-gray-800 rounded-lg shadow-md border border-gray-200 dark:border-gray-700">
	<h2 class="text-xl font-semibold mb-6 text-center text-gray-800 dark:text-gray-200">XYZ Grid Generation</h2>

	<form on:submit|preventDefault={generateGrid} class="space-y-5">

        <!-- Base Parameters Section -->
        <details class="border dark:border-gray-600 rounded p-3" open>
             <summary class="cursor-pointer font-medium text-gray-700 dark:text-gray-300 text-sm">Base Parameters</summary>
             <div class="mt-4 space-y-4 text-sm">
                 <!-- Model ID -->
                <div>
                    <label for="base_model_id" class="block text-xs font-medium mb-1">Model ID:</label>
                    <input type="text" id="base_model_id" bind:value={baseParams.model_id} required class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]" />
                </div>
                <!-- Prompt -->
                <div>
                    <label for="base_prompt" class="block text-xs font-medium mb-1">Prompt:</label>
                    <textarea id="base_prompt" rows="3" bind:value={baseParams.prompt} required class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]"></textarea>
                </div>
                 <!-- Negative Prompt -->
                <div>
                    <label for="base_neg_prompt" class="block text-xs font-medium mb-1">Negative Prompt:</label>
                    <textarea id="base_neg_prompt" rows="2" bind:value={baseParams.negative_prompt} class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]"></textarea>
                </div>
                 <!-- Dimensions -->
                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label for="base_width" class="block text-xs font-medium mb-1">Width:</label>
                        <input type="number" id="base_width" bind:value={baseParams.width} min="64" step="64" class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]" />
                    </div>
                    <div>
                        <label for="base_height" class="block text-xs font-medium mb-1">Height:</label>
                        <input type="number" id="base_height" bind:value={baseParams.height} min="64" step="64" class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]" />
                    </div>
                </div>
                 <!-- Steps & CFG -->
                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label for="base_steps" class="block text-xs font-medium mb-1">Steps: <span class="font-mono">{baseParams.num_inference_steps}</span></label>
                        <input type="range" id="base_steps" bind:value={baseParams.num_inference_steps} min="1" max="150" step="1" class="w-full h-2 [...]"/>
                    </div>
                    <div>
                        <label for="base_cfg" class="block text-xs font-medium mb-1">Guidance (CFG): <span class="font-mono">{baseParams.guidance_scale?.toFixed(1)}</span></label>
                        <input type="range" id="base_cfg" bind:value={baseParams.guidance_scale} min="0" max="20" step="0.5" class="w-full h-2 [...]"/>
                    </div>
                </div>
                 <!-- Seed -->
                <div>
                    <label for="base_seed" class="block text-xs font-medium mb-1">Base Seed:</label>
                    <input type="number" id="base_seed" placeholder="Default (Random per cell if not axis)" bind:value={baseParams.seed} min="0" step="1" class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]" />
                     <p class="mt-1 text-xs text-gray-500 dark:text-gray-400">Used if seed is not an axis. Increments per cell.</p>
                </div>
             </div>
        </details>

        <!-- Axis Definitions -->
        <div class="space-y-4">
            <!-- X Axis -->
            <div class="border dark:border-gray-600 rounded p-3">
                 <div class="flex items-center mb-2">
                     <input type="checkbox" id="x_axis_enabled" bind:checked={xAxisEnabled} class="h-4 w-4 text-indigo-600 [...] rounded">
                     <label for="x_axis_enabled" class="ml-2 block text-sm font-medium text-gray-700 dark:text-gray-300">X Axis</label>
                 </div>
                 {#if xAxisEnabled}
                 <div class="grid grid-cols-3 gap-2 items-end text-sm">
                     <div>
                         <label for="x_axis_param" class="block text-xs font-medium mb-1">Parameter:</label>
                         <select id="x_axis_param" bind:value={xAxisParam} class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]">
                             {#each availableAxisParams as param}
                                 <option value={param}>{param}</option>
                             {/each}
                         </select>
                     </div>
                     <div class="col-span-2">
                          <label for="x_axis_values" class="block text-xs font-medium mb-1">Values (comma separated):</label>
                         <input type="text" id="x_axis_values" bind:value={xAxisValuesString} required={xAxisEnabled} placeholder="e.g., 10, 20, 30 or apple, banana" class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]"/>
                     </div>
                 </div>
                 {/if}
            </div>

             <!-- Y Axis -->
             <div class="border dark:border-gray-600 rounded p-3">
                 <div class="flex items-center mb-2">
                     <input type="checkbox" id="y_axis_enabled" bind:checked={yAxisEnabled} class="h-4 w-4 text-indigo-600 [...] rounded">
                     <label for="y_axis_enabled" class="ml-2 block text-sm font-medium text-gray-700 dark:text-gray-300">Y Axis</label>
                 </div>
                 {#if yAxisEnabled}
                 <div class="grid grid-cols-3 gap-2 items-end text-sm">
                     <div>
                         <label for="y_axis_param" class="block text-xs font-medium mb-1">Parameter:</label>
                         <select id="y_axis_param" bind:value={yAxisParam} class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]">
                              {#each availableAxisParams as param}
                                 <option value={param}>{param}</option>
                             {/each}
                         </select>
                     </div>
                     <div class="col-span-2">
                          <label for="y_axis_values" class="block text-xs font-medium mb-1">Values (comma separated):</label>
                         <input type="text" id="y_axis_values" bind:value={yAxisValuesString} required={yAxisEnabled} placeholder="e.g., 10, 20, 30 or apple, banana" class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]"/>
                     </div>
                 </div>
                 {/if}
             </div>

            <!-- Z Axis (Optional) -->
            <div class="border dark:border-gray-600 rounded p-3">
                 <div class="flex items-center mb-2">
                     <input type="checkbox" id="z_axis_enabled" bind:checked={zAxisEnabled} class="h-4 w-4 text-indigo-600 [...] rounded">
                     <label for="z_axis_enabled" class="ml-2 block text-sm font-medium text-gray-700 dark:text-gray-300">Z Axis</label>
                 </div>
                 {#if zAxisEnabled}
                 <div class="grid grid-cols-3 gap-2 items-end text-sm">
                     <div>
                         <label for="z_axis_param" class="block text-xs font-medium mb-1">Parameter:</label>
                         <select id="z_axis_param" bind:value={zAxisParam} class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]">
                              {#each availableAxisParams as param}
                                 <option value={param}>{param}</option>
                             {/each}
                         </select>
                     </div>
                     <div class="col-span-2">
                          <label for="z_axis_values" class="block text-xs font-medium mb-1">Values (comma separated):</label>
                         <input type="text" id="z_axis_values" bind:value={zAxisValuesString} required={zAxisEnabled} placeholder="e.g., 10, 20, 30 or apple, banana" class="w-full px-2 py-1 border text-xs [...] dark:bg-gray-700 [...]"/>
                     </div>
                 </div>
                 {/if}
            </div>

        </div>


        <!-- Submit Button -->
		<div class="pt-4">
			<button type="submit" disabled={isLoading} class="w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-teal-600 hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 disabled:opacity-50 disabled:cursor-not-allowed transition duration-150 ease-in-out">
				{#if isLoading}
                     <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Generating Grid...
                {:else}
				    Generate XYZ Grid
                {/if}
			</button>
		</div>
	</form>

     <!-- Error Display (Local parsing or API error) -->
     {#if error}
        <div class="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 dark:bg-red-900 dark:text-red-200 dark:border-red-700 rounded text-sm" role="alert">
            <strong class="font-bold">Error:</strong> {error}
        </div>
    {/if}

</div>