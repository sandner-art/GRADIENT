<!-- frontend/src/lib/components/XYZGridForm.svelte -->
<script lang="ts">
	import { get, post, isApiError, type ApiError } from '#lib/apiClient.js';
	import type {
		XYZGridRequest,
		XYZGridResponse,
		DiffusionTaskBase,
		AxisParameter
	} from '#lib/types.js';
	import { createEventDispatcher, onMount } from 'svelte';
	import ModelSelector from './shared/ModelSelector.svelte';
	import SchedulerSelector from './shared/SchedulerSelector.svelte';
	import RecommendedSettings from './shared/RecommendedSettings.svelte';

	// Model / scheduler lists (same sources as the Text-to-Image form)
	let availableModels: string[] = [];
	let availableSchedulers: string[] = ['Default'];
	let isFetchingSystemInfo = true;
	onMount(async () => {
		try {
			const [models, schedulers] = await Promise.all([
				get<string[]>('/system/models/diffusion').catch((): string[] => []),
				get<string[]>('/system/schedulers').catch((): string[] => ['Default'])
			]);
			availableModels = models;
			availableSchedulers = ['Default', ...schedulers.filter((s) => s !== 'Default')];
			if (models.length && !models.includes(baseParams.model_id)) baseParams.model_id = models[0];
		} finally {
			isFetchingSystemInfo = false;
		}
	});

	const dispatch = createEventDispatcher<{
		submit: XYZGridResponse;
		error: string;
		loading: boolean;
	}>();

	// --- State for the form ---
	// Initialize with defaults matching DiffusionTaskBase structure
	let baseParams: DiffusionTaskBase = {
		model_id: 'photomatix_v3.fp16.safetensors', // Use provided default
		prompt: 'photo of a {} cat astronaut, masterpiece',
		negative_prompt: 'ugly, blurry, deformed, text, words, low quality',
		height: 512,
		width: 512,
		num_inference_steps: 20,
		guidance_scale: 7.0,
		seed: null, // Default seed null (random per cell if not axis)
		custom_sigmas: null,
		scheduler_id: null // Add scheduler_id if it's in DiffusionTaskBase type
	};

	// Available parameters for axes
	const availableAxisParams = [
		'prompt',
		'negative_prompt',
		'seed',
		'guidance_scale',
		'num_inference_steps',
		'width',
		'height',
		'scheduler_id',
		'model_id'
		// TODO: Add 'strength' if allowing Img2Img base params? Needs careful handling.
		// TODO: Add LoRA weights, ControlNet weights, VAE etc. later
	];

	// State for each axis
	let xAxisEnabled = true;
	let xAxisParam: string = 'guidance_scale';
	let xAxisValuesString: string = '5.0, 7.5, 10.0';

	let yAxisEnabled = false;
	let yAxisParam: string = 'seed';
	let yAxisValuesString: string = '10, 20, 30'; // Example: Needs valid seeds

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
				.split(',')
				.map((v) => v.trim())
				.filter((v) => v !== '');
			if (values.length === 0) return null;

			return values.map((val) => {
				// Handle known numeric types
				if (['seed', 'num_inference_steps', 'width', 'height'].includes(paramType)) {
					const num = parseInt(val, 10);
					if (isNaN(num) || (paramType === 'seed' && num < 0))
						throw new Error(`Invalid integer for ${paramType}: ${val}`);
					// Add check for multiples of 8 for dimensions
					if (['width', 'height'].includes(paramType) && (num <= 0 || num % 8 !== 0))
						throw new Error(`Invalid ${paramType} (must be positive multiple of 8): ${val}`);
					return num;
				} else if (['guidance_scale'].includes(paramType)) {
					// TODO: Add 'strength' check here later (0-1 range)
					const num = parseFloat(val);
					if (isNaN(num)) throw new Error(`Invalid float for ${paramType}: ${val}`);
					return num;
				} else if (paramType === 'scheduler_id') {
					// Optional: Validate against known schedulers? For now, accept string.
					if (!val) throw new Error(`Scheduler ID cannot be empty string.`);
					return val === 'Default' ? null : val; // Map 'Default' string to null
				}
				// Assume string otherwise (prompt, negative_prompt)
				return val;
			});
		} catch (e: unknown) {
			console.error('Error parsing axis values:', e);
			error = e instanceof Error ? e.message : 'Invalid format for axis values.';
			return null;
		}
	}

	// --- Form Submission ---
	async function generateGrid() {
		isLoading = true;
		error = null;
		dispatch('loading', true);

		let x_axis: AxisParameter | null = null;
		let y_axis: AxisParameter | null = null;
		let z_axis: AxisParameter | null = null;
		let parseError = false;
		let firstErrorMsg = '';

		function setParseError(msg: string) {
			if (!parseError) firstErrorMsg = msg; // Store only the first error encountered
			parseError = true;
		}

		if (xAxisEnabled) {
			const values = parseAxisValues(xAxisValuesString, xAxisParam);
			if (values) {
				x_axis = { parameter: xAxisParam, values: values };
			} else {
				setParseError(`Invalid values for X axis (${xAxisParam}).`);
			}
		}
		if (yAxisEnabled) {
			const values = parseAxisValues(yAxisValuesString, yAxisParam);
			if (values) {
				y_axis = { parameter: yAxisParam, values: values };
			} else {
				setParseError(`Invalid values for Y axis (${yAxisParam}).`);
			}
		}
		if (zAxisEnabled) {
			const values = parseAxisValues(zAxisValuesString, zAxisParam);
			if (values) {
				z_axis = { parameter: zAxisParam, values: values };
			} else {
				setParseError(`Invalid values for Z axis (${zAxisParam}).`);
			}
		}

		const activeAxesParams = [x_axis?.parameter, y_axis?.parameter, z_axis?.parameter].filter(
			(p) => p
		);
		if (activeAxesParams.length !== new Set(activeAxesParams).size) {
			setParseError('Cannot use the same parameter for multiple axes.');
		}

		if (parseError || (!x_axis && !y_axis && !z_axis)) {
			error = firstErrorMsg || 'At least one axis must be enabled with valid values.';
			isLoading = false;
			dispatch('loading', false);
			dispatch('error', error);
			return;
		}

		const payload: XYZGridRequest = {
			base_parameters: {
				model_id: baseParams.model_id, // Assume string
				prompt: baseParams.prompt, // Assume string
				negative_prompt: baseParams.negative_prompt || null, // Handle empty string
				height: Number(baseParams.height) || 512,
				width: Number(baseParams.width) || 512,
				num_inference_steps: Number(baseParams.num_inference_steps) || 25,
				guidance_scale: Number(baseParams.guidance_scale) || 7.0,
				seed:
					baseParams.seed === null || isNaN(Number(baseParams.seed))
						? null
						: Math.floor(Number(baseParams.seed)),
				// custom_sigmas: baseParams.custom_sigmas?.filter(n=>!isNaN(n)) ?? null, // Needs parsing if input is string
				custom_sigmas: null, // Not handled in base params UI yet
				scheduler_id: baseParams.scheduler_id || null // Handle empty string
			},
			x_axis: x_axis,
			y_axis: y_axis,
			z_axis: z_axis
		};

		console.log('Sending XYZ Grid payload:', payload);

		try {
			const response = await post<XYZGridResponse>('/generate/xyz-grid', payload);
			console.log('XYZ Grid successful:', response);
			dispatch('submit', response);
		} catch (e: unknown) {
			console.error('XYZ Grid failed:', e);
			let errorMsg: string;
			if (isApiError(e)) {
				errorMsg = `API Error (${e.status}): ${JSON.stringify(e.detail || e.statusText)}`;
			} else if (e instanceof Error) {
				errorMsg = `Error: ${e.message}`;
			} else {
				errorMsg = 'An unknown error occurred.';
			}
			error = errorMsg;
			dispatch('error', errorMsg);
		} finally {
			isLoading = false;
			dispatch('loading', false);
		}
	}
</script>

<div
	class="mx-auto max-w-2xl rounded-lg border border-gray-200 bg-white p-6 shadow-md dark:border-gray-700 dark:bg-gray-800"
>
	<h2 class="mb-6 text-center text-xl font-semibold text-gray-800 dark:text-gray-200">
		XYZ Grid Generation
	</h2>

	<form on:submit|preventDefault={generateGrid} class="space-y-5">
		<!-- Base Parameters Section -->
		<details class="rounded border p-3 dark:border-gray-600" open>
			<summary class="cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300"
				>Base Parameters</summary
			>
			<div class="mt-4 space-y-4 text-sm">
				<!-- Model ID -->
				<ModelSelector
					id="base_model_id_xyz"
					label="Model ID:"
					{availableModels}
					selectedModelId={baseParams.model_id}
					on:change={(e) => (baseParams = { ...baseParams, model_id: e.detail })}
					disabled={isFetchingSystemInfo}
				/>
				<RecommendedSettings
					modelId={baseParams.model_id}
					onapply={(d) =>
						(baseParams = {
							...baseParams,
							width: d.width ?? baseParams.width,
							height: d.height ?? baseParams.height,
							num_inference_steps: d.steps ?? baseParams.num_inference_steps,
							guidance_scale: d.guidance_scale ?? baseParams.guidance_scale,
							scheduler_id: d.scheduler ?? baseParams.scheduler_id
						})}
				/>
				<!-- Prompt -->
				<div>
					<label for="base_prompt_xyz" class="mb-1 block text-xs font-medium">Prompt:</label>
					<!-- Unique ID -->
					<textarea
						id="base_prompt_xyz"
						rows="3"
						bind:value={baseParams.prompt}
						required
						class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
					></textarea>
				</div>
				<!-- Negative Prompt -->
				<div>
					<label for="base_neg_prompt_xyz" class="mb-1 block text-xs font-medium"
						>Negative Prompt:</label
					>
					<!-- Unique ID -->
					<textarea
						id="base_neg_prompt_xyz"
						rows="2"
						bind:value={baseParams.negative_prompt}
						class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
					></textarea>
				</div>
				<!-- Dimensions -->
				<div class="grid grid-cols-2 gap-3">
					<div>
						<label for="base_width_xyz" class="mb-1 block text-xs font-medium">Width:</label>
						<!-- Unique ID -->
						<input
							type="number"
							id="base_width_xyz"
							bind:value={baseParams.width}
							min="64"
							step="8"
							required
							class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
						/>
					</div>
					<div>
						<label for="base_height_xyz" class="mb-1 block text-xs font-medium">Height:</label>
						<!-- Unique ID -->
						<input
							type="number"
							id="base_height_xyz"
							bind:value={baseParams.height}
							min="64"
							step="8"
							required
							class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
						/>
					</div>
				</div>
				<!-- Steps & CFG -->
				<div class="grid grid-cols-2 gap-3">
					<div>
						<label for="base_steps_xyz" class="mb-1 block text-xs font-medium"
							>Steps: <span class="font-mono">{baseParams.num_inference_steps}</span></label
						>
						<!-- Unique ID -->
						<input
							type="range"
							id="base_steps_xyz"
							bind:value={baseParams.num_inference_steps}
							min="1"
							max="150"
							step="1"
							class="h-2 w-full cursor-pointer appearance-none rounded-lg bg-gray-200 dark:bg-gray-600"
						/>
					</div>
					<div>
						<label for="base_cfg_xyz" class="mb-1 block text-xs font-medium"
							>Guidance (CFG): <span class="font-mono">{baseParams.guidance_scale?.toFixed(1)}</span
							></label
						>
						<!-- Unique ID -->
						<input
							type="range"
							id="base_cfg_xyz"
							bind:value={baseParams.guidance_scale}
							min="0"
							max="20"
							step="0.5"
							class="h-2 w-full cursor-pointer appearance-none rounded-lg bg-gray-200 dark:bg-gray-600"
						/>
					</div>
				</div>
				<!-- Scheduler -->
				<SchedulerSelector
					id="base_scheduler_id_xyz"
					label="Scheduler:"
					{availableSchedulers}
					selectedSchedulerId={baseParams.scheduler_id}
					on:change={(e) => (baseParams = { ...baseParams, scheduler_id: e.detail })}
					disabled={isFetchingSystemInfo}
				/>
				<!-- Seed -->
				<div>
					<label for="base_seed_xyz" class="mb-1 block text-xs font-medium">Base Seed:</label>
					<!-- Unique ID -->
					<input
						type="number"
						id="base_seed_xyz"
						placeholder="Random (one seed for the whole grid)"
						bind:value={baseParams.seed}
						min="0"
						step="1"
						class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
					/>
					<p class="mt-1 text-xs text-gray-500 dark:text-gray-400">
						Every cell uses this seed unless seed is an axis, so cells differ only by the axis
						values.
					</p>
				</div>
				<!-- Custom Sigmas (Not implemented in base UI yet) -->
			</div>
		</details>

		<!-- Axis Definitions -->
		<div class="space-y-4">
			<!-- X Axis -->
			<div class="rounded border p-3 dark:border-gray-600">
				<div class="mb-2 flex items-center">
					<input
						type="checkbox"
						id="x_axis_enabled"
						bind:checked={xAxisEnabled}
						class="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-indigo-600"
					/>
					<label
						for="x_axis_enabled"
						class="ml-2 block text-sm font-medium text-gray-700 dark:text-gray-300">X Axis</label
					>
				</div>
				{#if xAxisEnabled}
					<div class="grid grid-cols-3 items-end gap-2 text-sm">
						<div>
							<label for="x_axis_param" class="mb-1 block text-xs font-medium">Parameter:</label>
							<select
								id="x_axis_param"
								bind:value={xAxisParam}
								class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
							>
								{#each availableAxisParams as param}
									<option value={param}>{param}</option>
								{/each}
							</select>
						</div>
						<div class="col-span-2">
							<label for="x_axis_values" class="mb-1 block text-xs font-medium"
								>Values (comma separated):</label
							>
							<input
								type="text"
								id="x_axis_values"
								bind:value={xAxisValuesString}
								required={xAxisEnabled}
								placeholder="e.g., 5.0, 7.5, 10.0"
								class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
							/>
						</div>
					</div>
				{/if}
			</div>

			<!-- Y Axis -->
			<div class="rounded border p-3 dark:border-gray-600">
				<div class="mb-2 flex items-center">
					<input
						type="checkbox"
						id="y_axis_enabled"
						bind:checked={yAxisEnabled}
						class="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-indigo-600"
					/>
					<label
						for="y_axis_enabled"
						class="ml-2 block text-sm font-medium text-gray-700 dark:text-gray-300">Y Axis</label
					>
				</div>
				{#if yAxisEnabled}
					<div class="grid grid-cols-3 items-end gap-2 text-sm">
						<div>
							<label for="y_axis_param" class="mb-1 block text-xs font-medium">Parameter:</label>
							<select
								id="y_axis_param"
								bind:value={yAxisParam}
								class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
							>
								{#each availableAxisParams as param}
									<option value={param}>{param}</option>
								{/each}
							</select>
						</div>
						<div class="col-span-2">
							<label for="y_axis_values" class="mb-1 block text-xs font-medium"
								>Values (comma separated):</label
							>
							<input
								type="text"
								id="y_axis_values"
								bind:value={yAxisValuesString}
								required={yAxisEnabled}
								placeholder="e.g., 10, 20, 30"
								class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
							/>
						</div>
					</div>
				{/if}
			</div>

			<!-- Z Axis (Optional) -->
			<div class="rounded border p-3 dark:border-gray-600">
				<div class="mb-2 flex items-center">
					<input
						type="checkbox"
						id="z_axis_enabled"
						bind:checked={zAxisEnabled}
						class="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:ring-offset-gray-800 dark:focus:ring-indigo-600"
					/>
					<label
						for="z_axis_enabled"
						class="ml-2 block text-sm font-medium text-gray-700 dark:text-gray-300">Z Axis</label
					>
				</div>
				{#if zAxisEnabled}
					<div class="grid grid-cols-3 items-end gap-2 text-sm">
						<div>
							<label for="z_axis_param" class="mb-1 block text-xs font-medium">Parameter:</label>
							<select
								id="z_axis_param"
								bind:value={zAxisParam}
								class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
							>
								{#each availableAxisParams as param}
									<option value={param}>{param}</option>
								{/each}
							</select>
						</div>
						<div class="col-span-2">
							<label for="z_axis_values" class="mb-1 block text-xs font-medium"
								>Values (comma separated):</label
							>
							<input
								type="text"
								id="z_axis_values"
								bind:value={zAxisValuesString}
								required={zAxisEnabled}
								placeholder="e.g., 15, 25"
								class="w-full rounded-md border border-gray-300 px-2 py-1 text-xs shadow-sm focus:border-indigo-500 focus:ring-indigo-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
							/>
						</div>
					</div>
				{/if}
			</div>
		</div>

		<!-- Submit Button -->
		<div class="pt-4">
			<!-- Use shared button component if desired -->
			<!-- <SubmitButton isLoading={isLoading} buttonText="Generate XYZ Grid" loadingText="Generating Grid..." /> -->
			<!-- Or keep specific button for now -->
			<button
				type="submit"
				disabled={isLoading}
				class="inline-flex w-full justify-center rounded-md border border-transparent bg-teal-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition duration-150 ease-in-out hover:bg-teal-700 focus:ring-2 focus:ring-teal-500 focus:ring-offset-2 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
			>
				{#if isLoading}
					<svg
						class="mr-3 -ml-1 h-5 w-5 animate-spin text-white"
						xmlns="http://www.w3.org/2000/svg"
						fill="none"
						viewBox="0 0 24 24"
					>
						<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"
						></circle>
						<path
							class="opacity-75"
							fill="currentColor"
							d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
						></path>
					</svg>
					Generating Grid...
				{:else}
					Generate XYZ Grid
				{/if}
			</button>
		</div>
	</form>

	<!-- Error Display -->
	{#if error}
		<div
			class="mt-4 rounded border border-red-400 bg-red-100 p-3 text-sm text-red-700 dark:border-red-700 dark:bg-red-900 dark:text-red-200"
			role="alert"
		>
			<strong class="font-bold">Error:</strong>
			{error}
		</div>
	{/if}
</div>
