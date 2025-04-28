<!-- frontend/src/lib/components/XYZGridDisplay.svelte (NEW FILE) -->
<script lang="ts">
	import type { XYZGridResponse, XYZGridResult, AxisParameter } from '$lib/types'; // Import types
	import { createEventDispatcher } from 'svelte';

	export let gridResponse: XYZGridResponse | null = null;

    const dispatch = createEventDispatcher();

    // Function to re-run a specific cell's parameters (emits event)
    function rerunCell(cellParams: Record<string, any>) {
        console.log("Rerunning cell with params:", cellParams);
        // Dispatch an event upwards with the parameters needed for a Text2Image request
        // Parent component will handle submitting this to the T2I endpoint
        dispatch('rerun', cellParams);
    }

    // Reactive calculations for grid structure
    let gridData: (XYZGridResult | null)[][] = []; // 2D array for display (assuming only X/Y axes for now)
    let xValues: any[] = [];
    let yValues: any[] = [];
    let xAxisParam: string | null = null;
    let yAxisParam: string | null = null;

    $: { // Recalculate grid structure when gridResponse changes
        if (gridResponse && gridResponse.grid_results.length > 0) {
            const results = gridResponse.grid_results;
            const req = gridResponse.request_params;

            // Determine axes used and their unique values
            xAxisParam = req.x_axis?.parameter ?? null;
            yAxisParam = req.y_axis?.parameter ?? null;
            // Basic support for only X and Y axes for display initially
            // TODO: Handle Z axis display (e.g., multiple grids, dropdown selector)

            xValues = xAxisParam ? [...new Set(results.map(r => r.x_value))] : [null]; // Unique X values or single null column
            yValues = yAxisParam ? [...new Set(results.map(r => r.y_value))] : [null]; // Unique Y values or single null row

            // Sort values numerically if possible, otherwise alphabetically
            const sortFn = (a: any, b: any) => {
                const numA = Number(a);
                const numB = Number(b);
                if (!isNaN(numA) && !isNaN(numB)) {
                    return numA - numB;
                }
                return String(a).localeCompare(String(b));
            };
            if (xAxisParam) xValues.sort(sortFn);
            if (yAxisParam) yValues.sort(sortFn);

            // Create 2D grid structure (Y rows, X columns)
            gridData = yValues.map(yVal => {
                return xValues.map(xVal => {
                    // Find the result corresponding to this specific X/Y combination
                    return results.find(r => {
                        const xMatch = xAxisParam ? r.x_value === xVal : true; // Match if no x-axis
                        const yMatch = yAxisParam ? r.y_value === yVal : true; // Match if no y-axis
                        // TODO: Add Z check if implementing Z display
                        return xMatch && yMatch;
                    }) || null; // Return null if no matching result found (shouldn't happen ideally)
                });
            });
             console.log("Processed Grid Data:", gridData, "X:", xValues, "Y:", yValues);

        } else {
            gridData = [];
            xValues = [];
            yValues = [];
            xAxisParam = null;
            yAxisParam = null;
        }
    }

</script>

{#if gridResponse}
<div class="mt-8 p-4 bg-gray-50 dark:bg-gray-850 rounded-lg border dark:border-gray-700 shadow">
    <h3 class="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">XYZ Grid Results</h3>
    {#if gridData.length > 0}
        <div class="overflow-x-auto">
            <table class="min-w-full border-collapse border border-gray-300 dark:border-gray-600">
                <thead>
                    <tr class="bg-gray-100 dark:bg-gray-750">
                        <!-- Corner cell -->
                        <th class="border p-2 text-xs font-medium text-gray-600 dark:text-gray-400">
                            {#if yAxisParam}Y: {yAxisParam}{/if}
                            {#if xAxisParam && yAxisParam}<br>/{/if}
                            {#if xAxisParam}X: {xAxisParam}{/if}
                        </th>
                        <!-- X Axis Headers -->
                        {#each xValues as xVal}
                             <th class="border p-2 text-xs font-semibold text-gray-700 dark:text-gray-300 break-words max-w-[100px] whitespace-normal" title={xAxisParam ? `${xAxisParam}: ${xVal}` : ''}>
                                {String(xVal ?? 'N/A')}
                             </th>
                        {/each}
                    </tr>
                </thead>
                <tbody>
                     {#each gridData as row, yIndex}
                        <tr class="hover:bg-gray-50 dark:hover:bg-gray-800">
                            <!-- Y Axis Header -->
                             <th class="border p-2 text-xs font-semibold text-gray-700 dark:text-gray-300 break-words max-w-[100px] whitespace-normal align-top" title={yAxisParam ? `${yAxisParam}: ${yValues[yIndex]}` : ''}>
                                {String(yValues[yIndex] ?? 'N/A')}
                             </th>
                            <!-- Grid Cells -->
                            {#each row as cell, xIndex}
                                <td class="border p-1 align-top text-center min-w-[150px] min-h-[150px]">
                                    {#if cell}
                                        {#if cell.image_result}
                                            <div class="relative group">
                                                <img
                                                    src="data:image/png;base64,{cell.image_result.image_base64}"
                                                    alt="Grid cell X:{xValues[xIndex]} Y:{yValues[yIndex]}"
                                                    class="w-full h-auto object-contain rounded max-w-[256px] max-h-[256px] mx-auto"
                                                    loading="lazy"
                                                />
                                                <!-- Basic tooltip with parameters on hover -->
                                                <div class="absolute bottom-0 left-0 right-0 mb-2 px-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10">
                                                    <div class="bg-black bg-opacity-70 text-white text-xs rounded p-1 max-w-full break-words">
                                                         {#if cell.parameters.seed}Seed: {cell.parameters.seed}{/if}
                                                         <!-- Add other key params -->
                                                         {#if cell.parameters.guidance_scale} CFG: {cell.parameters.guidance_scale}{/if}
                                                         {#if cell.parameters.num_inference_steps} Steps: {cell.parameters.num_inference_steps}{/if}
                                                         <button
                                                             title="Rerun with these parameters"
                                                             class="ml-1 text-blue-300 hover:text-blue-100 text-sm"
                                                             on:click={() => rerunCell(cell?.parameters ?? {})}>
                                                                🔄
                                                         </button>
                                                    </div>
                                                </div>
                                            </div>
                                        {:else if cell.error}
                                            <div class="flex items-center justify-center h-full text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900 p-2 rounded break-words">
                                               Error: {cell.error}
                                            </div>
                                        {:else}
                                            <div class="flex items-center justify-center h-full text-xs text-gray-400">Empty Cell</div>
                                        {/if}
                                    {:else}
                                         <div class="flex items-center justify-center h-full text-xs text-gray-400">Missing Data</div>
                                    {/if}
                                </td>
                            {/each}
                        </tr>
                     {/each}
                </tbody>
            </table>
        </div>
    {:else}
        <p class="text-center text-gray-500 dark:text-gray-400">No grid data to display.</p>
    {/if}
</div>
{/if}