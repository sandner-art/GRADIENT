// frontend/src/lib/utils/presetManager.ts
import { get, post, isApiError, type ApiError } from '#lib/apiClient.js'; // Import isApiError and ApiError type
import type { PresetRead, PresetCreate, PresetSetting, DiffusionTaskBase } from '#lib/types.js';
import type { Writable } from 'svelte/store'; // Import Writable type for store parameter

/**
 * Fetches presets for a given model identifier from the API.
 * @param modelId The model identifier string.
 * @returns Promise resolving to an array of PresetRead objects or empty array on error.
 */
export async function fetchPresetsForModel(modelId: string | null | undefined): Promise<PresetRead[]> {
	if (!modelId) {
		return []; // No model selected, no presets
	}
	console.log(`Fetching presets utility for model: ${modelId}`); // Keep log for debugging fetch triggers
	try {
		const encodedModelId = encodeURIComponent(modelId);
		const presets = await get<PresetRead[]>(`/presets?model_identifier=${encodedModelId}`);
		console.log(`Fetched ${presets?.length ?? 0} presets for ${modelId}`);
		return presets || []; // Return fetched presets or empty array
	} catch (err) {
		console.error(`Failed to fetch presets for ${modelId}:`, err);
		// Optionally display a non-blocking error to the user via a store or notification system
		return []; // Return empty array on error
	}
}

/**
 * Applies the settings from a selected preset to a Svelte store.
 * @param preset The PresetRead object to apply.
 * @param store The Svelte writable store (e.g., t2iFormStore, i2iFormStore) to update.
 * @param updateCallback Optional callback function to run after store update (e.g., clear related local state like customSigmasString).
 */
export function applyPresetToStore<T extends DiffusionTaskBase>(
	preset: PresetRead | null,
	store: Writable<T>,
	updateCallback?: () => void
): void {
	if (!preset) {
		// console.log("No preset provided to apply."); // Can be noisy, remove if not needed
		return;
	}

	console.log('Applying preset to store:', preset.preset_name, preset.settings);
	store.update((current) => {
		const updatedState = { ...current }; // Create a new object based on current state
		// Iterate over settings defined IN THE PRESET
		for (const key in preset.settings) {
			if (Object.prototype.hasOwnProperty.call(preset.settings, key)) {
				// Apply setting only if the key actually exists as a property on the store's state type
				// This prevents applying arbitrary keys from the preset's 'extra: allow' config
				if (key in updatedState) {
					(updatedState as any)[key] = (preset.settings as any)[key]; // Apply the preset value
				} else {
					console.warn(`Preset setting '${key}' not found in store type, skipping.`);
				}
			}
		}
		// Explicitly handle null scheduler from preset if it exists in settings
		// Ensures we overwrite a string value with null if the preset intends it
		if (Object.prototype.hasOwnProperty.call(preset.settings, 'scheduler_id') && preset.settings.scheduler_id === null) {
			updatedState.scheduler_id = null;
		}
		// Reset custom sigmas if preset defined steps (and steps is not null)
		if (preset.settings.num_inference_steps !== undefined && preset.settings.num_inference_steps !== null) {
			updatedState.custom_sigmas = null;
		}
        // TODO: Add logic here later if presets should store/apply LoRA/Embedding selections
		return updatedState; // Return the fully updated state object
	});

	// Run callback if provided (e.g., to clear related local UI state)
	if (updateCallback) {
		updateCallback();
	}
}

/**
 * Saves the current settings from a store as a new preset via the API.
 * @param store The Svelte writable store containing the current settings.
 * @param getSettingsToSave Function that takes the current store value and returns a PresetSetting object containing only the fields to save.
 * @returns Promise resolving to the newly created PresetRead object or null on error/cancellation.
 */
export async function savePresetFromStore<T extends DiffusionTaskBase>(
	store: Writable<T>,
	getSettingsToSave: (currentSettings: T) => PresetSetting
): Promise<PresetRead | null> {

	// Helper to get the current value from the store non-reactively
    function getCurrentStoreValue(storeInstance: Writable<T>): T {
        let value: T | undefined;
        const unsubscribe = storeInstance.subscribe(v => { value = v; });
        unsubscribe(); // Immediately unsubscribe
        if (value === undefined) {
            throw new Error("Could not get current store value."); // Should not happen with writable
        }
        return value;
    }

    const currentSettings = getCurrentStoreValue(store);

	if (!currentSettings.model_id) {
		alert('Cannot save preset without a selected Model ID.');
		return null;
	}

	const name = prompt('Enter a name for this preset:');
	if (!name || name.trim() === '') {
		alert('Preset name cannot be empty.');
		return null;
	}

	// Use the provided callback to extract ONLY the settings we want to save
	const presetSettingsToSave = getSettingsToSave(currentSettings);

	const payload: PresetCreate = {
		model_identifier: currentSettings.model_id,
		preset_name: name.trim(),
		settings: presetSettingsToSave, // Use the specifically extracted settings
		description: `Saved from UI on ${new Date().toLocaleDateString()}` // Optional default description
	};

	console.log('Saving preset via utility:', payload);

	try {
		// Assume caller handles loading state if needed (e.g., around the button click)
		const newPreset = await post<PresetRead>('/presets', payload);
		alert(`Preset "${name}" saved successfully!`);
		return newPreset; // Return the created preset (caller might use it, e.g., to refetch)
	} catch (err: unknown) { // Catch error as unknown type
		console.error('Failed to save preset:', err);
		let errorMsg: string;
		// Use the imported type guard
		if (isApiError(err)) {
			// Now TypeScript knows 'err' has ApiError properties
			errorMsg = `Save Preset Error: ${err.status} - ${JSON.stringify(err.detail || err.statusText)}`;
		} else if (err instanceof Error) {
			errorMsg = `Save Preset Error: ${err.message}`;
		} else {
			errorMsg = `Save Preset Error: ${String(err)}`;
		}
		alert(errorMsg); // Show user the error
		return null; // Indicate failure
	}
}
