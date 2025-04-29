<!-- frontend/src/lib/components/XYZGridDisplay.svelte -->
<script lang="ts">
	import type { XYZGridResponse, XYZGridResult, AxisParameter } from '$lib/types';
	import { createEventDispatcher } from 'svelte';

	export let gridResponse: XYZGridResponse | null = null;
    const dispatch = createEventDispatcher<{rerun: Record<string, any>}>(); // Specify event detail type

    function rerunCell(cellParams: Record<string, any>) {
        console.log("Rerunning cell with params:", cellParams);
        dispatch('rerun', cellParams);
    }

    // Reactive calculations for grid structure
    let gridData: (XYZGridResult | null)[][] = [];
    let xValues: any[] = [];
    let yValues: any[] = [];
    let xAxisParam: string | null = null;
    let yAxisParam: string | null = null;

    $: { // Recalculate grid structure when gridResponse changes
        if (gridResponse && gridResponse.grid_results.length > 0) {
            const results = gridResponse.grid_results; const req = gridResponse.request_params;
            xAxisParam = req.x_axis?.parameter ?? null; yAxisParam = req.y_axis?.parameter ?? null;
            xValues = xAxisParam ? [...new Set(results.map(r => r.x_value))] : [null];
            yValues = yAxisParam ? [...new Set(results.map(r => r.y_value))] : [null];
            const sortFn = (a: any, b: any) => { const numA = Number(a); const numB = Number(b); if (!isNaN(numA) && !isNaN(numB)) { return numA - numB; } return String(a).localeCompare(String(b)); };
            if (xAxisParam) xValues.sort(sortFn); if (yAxisParam) yValues.sort(sortFn);
            gridData = yValues.map(yVal => xValues.map(xVal => results.find(r => (xAxisParam ? r.x_value === xVal : true) && (yAxisParam ? r.y_value === yVal : true)) || null));
            console.log("Processed Grid Data:", gridData.length > 0 ? gridData[0].length : 0, "x", gridData.length, "X Values:", xValues, "Y Values:", yValues); // Log dimensions
        } else { gridData = []; xValues = []; yValues = []; xAxisParam = null; yAxisParam = null; }
    }
</script>

{#if gridResponse}
<div class="mt-8 p-4 bg-gray-50 dark:bg-gray-850 rounded-lg border dark:border-gray-700 shadow">
    <h3 class="text-xl font-semibold mb-4 text-gray-800 dark:text-gray-200">XYZ Grid Results</h3>
    {#if gridData.length > 0}
        <div class="overflow-x-auto">
            <table class="min-w-full border-collapse border border-gray-300 dark:border-gray-600 table-fixed"> <!-- Use table-fixed for better layout control -->
                <thead>
                    <tr class="bg-gray-100 dark:bg-gray-750">
                        <!-- Corner cell -->
                        <th class="border p-1 text-xs font-medium text-gray-600 dark:text-gray-400 w-[100px] break-words"> <!-- Fixed width? -->
                            {#if yAxisParam}Y: {yAxisParam}{/if}
                            {#if xAxisParam && yAxisParam}<br>/{/if}
                            {#if xAxisParam}X: {xAxisParam}{/if}
                        </th>
                        <!-- X Axis Headers -->
                        {#each xValues as xVal}
                             <th class="border p-1 text-xs font-semibold text-gray-700 dark:text-gray-300 break-words w-[150px] whitespace-normal align-middle" title={xAxisParam ? `${xAxisParam}: ${xVal}` : ''}> <!-- Width? -->
                                {String(xVal ?? 'N/A')}
                             </th>
                        {/each}
                    </tr>
                </thead>
                <tbody>
                     {#each gridData as row, yIndex}
                        <tr> <!-- Removed hover effect for simplicity -->
                            <!-- Y Axis Header -->
                             <th class="border p-1 text-xs font-semibold text-gray-700 dark:text-gray-300 break-words w-[100px] whitespace-normal align-middle" title={yAxisParam ? `${yAxisParam}: ${yValues[yIndex]}` : ''}> <!-- Fixed width? -->
                                {String(yValues[yIndex] ?? 'N/A')}
                             </th>
                            <!-- Grid Cells -->
                            {#each row as cell, xIndex}
                                <td class="border p-1 align-top text-center w-[150px] h-[150px] overflow-hidden"> <!-- Fixed Size? -->
                                    {#if cell}
                                        {#if cell.image_result}
                                            <div class="relative group w-full h-full flex items-center justify-center">
                                                <img
                                                    src="data:image/png;base64,{cell.image_result.image_base64}"
                                                    alt={`Grid cell X:${xValues[xIndex]} Y:${yValues[yIndex]}`} 
                                                    class="max-w-full max-h-full object-contain rounded"
                                                    loading="lazy"
                                                />
                                                <!-- Tooltip -->
                                                <div class="absolute bottom-0 left-0 right-0 mb-1 px-1 opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10">
                                                    <div class="bg-black bg-opacity-70 text-white text-[10px] leading-tight rounded p-1 max-w-full break-words shadow-lg"> <!-- Smaller text -->
                                                         {#if cell.parameters.seed}Seed: {cell.parameters.seed}{/if}
                                                         {#if cell.parameters.guidance_scale} CFG: {cell.parameters.guidance_scale}{/if}
                                                         {#if cell.parameters.num_inference_steps} Steps: {cell.parameters.num_inference_steps}{/if}
                                                         <button
                                                             title="Rerun with these parameters"
                                                             class="ml-1 text-blue-300 hover:text-blue-100 text-xs align-middle"
                                                             on:click|stopPropagation={() => rerunCell(cell?.parameters ?? {})}> <!-- Stop propagation -->
                                                                🔄
                                                         </button>
                                                    </div>
                                                </div>
                                            </div>
                                        {:else if cell.error}
                                            <div class="flex items-center justify-center w-full h-full text-[10px] text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900 p-1 rounded break-words">
                                               <span class="font-semibold mr-1">Error:</span> {cell.error}
                                            </div>
                                        {:else}
                                            <div class="flex items-center justify-center w-full h-full text-xs text-gray-400">Empty</div>
                                        {/if}
                                    {:else}
                                         <div class="flex items-center justify-center w-full h-full text-xs text-gray-400">N/A</div>
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