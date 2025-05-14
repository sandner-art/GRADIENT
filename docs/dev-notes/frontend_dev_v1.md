Changes and refactoring applied to the frontend (`frontend/`) codebase during our recent development work:

1.  **Created Core UI Components:**
    *   Added `TextToImageForm.svelte` for T2I parameters and results display.
    *   Added `ImageToImageForm.svelte` for I2I parameters (including image upload, strength slider) and results display.
    *   Added `XYZGridForm.svelte` for configuring XYZ grid parameters.
    *   Added `XYZGridDisplay.svelte` for visualizing the results of an XYZ grid in a table format.
    *   Added placeholder/basic `LLMChat.svelte`.

2.  **Refactored Common UI Elements:**
    *   Created `shared/ModelSelector.svelte` to provide a consistent dropdown/selection UI for the `model_id`.
    *   Created `shared/SchedulerSelector.svelte` for selecting diffusion schedulers, handling the "Default" (null) option.
    *   Created `shared/SeedInput.svelte` to encapsulate the number input, randomize button (🎲), and clear button (❌) for the seed parameter.
    *   Created `shared/SubmitButton.svelte` to provide a consistent button with integrated loading state (spinner and text change).
    *   Integrated these shared components into `TextToImageForm.svelte` and `ImageToImageForm.svelte`, reducing code duplication in their templates.

3.  **Implemented Svelte Stores for State Persistence:**
    *   Created `stores/t2iFormStore.ts` (using `writable`) to hold the state for `TextToImageForm`.
    *   Created `stores/i2iFormStore.ts` (using `writable`) to hold the state for `ImageToImageForm`.
    *   Refactored `TextToImageForm.svelte` and `ImageToImageForm.svelte` to read from and bind inputs directly to their respective stores (`$t2iFormStore`, `$i2iFormStore`). This ensures form values persist when switching tabs.

4.  **Added Preset Management UI:**
    *   Added state within T2I/I2I forms to track available presets (`availablePresets`) and the currently selected one (`selectedPresetId`).
    *   Implemented logic (`fetchPresetsForModel`, `handleModelChangeForPresets`) using `onMount` and `beforeUpdate` to fetch presets from the backend `/api/v1/presets` endpoint whenever the selected `model_id` changes.
    *   Added a "Load Preset" `<select>` dropdown to both forms, populated with fetched presets.
    *   Implemented logic (`applyPreset`, `handlePresetSelectionChange`) using the `presetManager.ts` utility to update the relevant form's store when a preset is selected.
    *   Added a "Save Current Settings as Preset" button to both forms, calling the `savePresetFromStore` utility to interact with the backend API.

5.  **Added LoRA / Embedding Selection UI:**
    *   Added state within T2I/I2I forms to manage lists of available (`availableLoras`, `availableEmbeddings`) and selected (`selectedLoras`, `selectedEmbeddings`) items.
    *   Implemented UI sections in both forms:
        *   For LoRAs: Allows adding multiple LoRAs, selecting the specific LoRA file from a dropdown (populated dynamically), adjusting its weight with a slider, and removing entries.
        *   For Embeddings: Allows manually adding embedding trigger words (via `prompt()`) and displays them as removable tags. Also displays the list of *found* embedding files for user information.
    *   Updated the form submission logic (`generateImage`) to include `loras` (list of objects with name/weight) and `embeddings` (list of strings) in the payload sent to the backend API.

6.  **Implemented Dynamic Selectors (Data Fetching):**
    *   Added `fetchSystemInfo` function (called in `onMount`) to T2I/I2I forms to fetch lists of available models, schedulers, LoRAs, and embeddings from the backend `/api/v1/system/...` endpoints.
    *   Populated the `ModelSelector` and `SchedulerSelector` dropdowns using this fetched data instead of hardcoded lists.
    *   Added loading indicators (`isFetchingSystemInfo`, `isFetchingPresets`) and basic validation/resetting if stored values weren't found in fetched lists.

7.  **Enhanced Main Page (`+page.svelte`):**
    *   Implemented a tabbed interface (`activeTab` state) to switch between T2I, I2I, Grid, and Chat components in the left column.
    *   Managed state specifically for the Grid generation (`latestGridResponse`, `isGridLoading`, `gridError`) based on events dispatched from `XYZGridForm`.
    *   Conditionally rendered the `XYZGridDisplay` component (or loading/error states) in the right column when the 'grid' tab is active.
    *   Added "Last Result" indicators/previews/links in the right column for T2I, I2I, and Grid results when their respective tabs are *not* active, improving UX awareness.
    *   Implemented `handleRerun` functionality, triggered by `XYZGridDisplay`, which populates the `t2iFormStore` and switches to the T2I tab.

8.  **TypeScript Integration & Refinements:**
    *   Created `lib/types.ts` to centralize TypeScript interfaces matching backend Pydantic models.
    *   Refactored preset logic into `lib/utils/presetManager.ts`.
    *   Added type hints to functions and event handlers.
    *   Resolved various TypeScript errors related to store type inference, prop types, and template syntax.

9.  **Accessibility:**
    *   Corrected `alt` text on generated images to avoid redundancy (`a11y_img_redundant_alt`).
    *   Corrected section headings in LoRA/Embedding sections to use `<div>` instead of `<label>` to resolve `a11y_label_has_associated_control` warnings.

In essence, the frontend has evolved from a very basic shell to a multi-tabbed interface with two fully functional generation forms (T2I, I2I) featuring state persistence, dynamic options, preset management, shared UI components, and placeholders for advanced features like LoRAs and Embeddings. The XYZ Grid configuration form and a basic display are also integrated.