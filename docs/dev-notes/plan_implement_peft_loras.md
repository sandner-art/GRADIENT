Instructions geared towards developers of an application like "GRADIENT UI" (or any similar UI for Stable Diffusion) on how to best implement direct compatibility with PEFT LoRAs, especially those produced by a framework like Adaptra which uses the `peft` library.

The goal is to seamlessly load LoRAs saved in two common PEFT-generated formats:

1.  **PEFT Directory Format:** A directory containing `adapter_model.safetensors` (or `.bin`) and `adapter_config.json`.
2.  **Single File LoRA (`.safetensors`):** A single file containing only the LoRA weights, typically created for broader UI compatibility (like the one Adaptra now tries to generate as `unet_lora_{tag}_ui.safetensors`).

---

## Instructions for GRADIENT UI Developers: Implementing PEFT LoRA Compatibility

To provide robust support for LoRAs trained using the Hugging Face `peft` library (as done by frameworks like Adaptra), your UI should ideally handle both the standard PEFT directory format and single-file LoRA weights. This involves leveraging the `diffusers` library's LoRA loading capabilities, which are designed to work with PEFT-generated adapters.

**Core Principle:** Use `pipeline.load_lora_weights()` (or component-specific methods like `unet.load_lora_weights()` or `unet.load_attn_procs()`) from the `diffusers` library. This method is becoming increasingly adept at handling various LoRA formats, including those from PEFT.

### Scenario 1: Loading LoRA from a PEFT Directory

This is the format saved by `PeftModel.save_pretrained()`, which Adaptra produces as the `unet_lora_peft_format/` directory. It contains:
*   `adapter_model.safetensors` (or `adapter_model.bin`): The LoRA weights.
*   `adapter_config.json`: PEFT configuration (rank, alpha, target_modules, etc.).

**How GRADIENT UI should handle it:**

When the user points to such a directory (or a `.safetensors`/`.bin` file *within* such a directory where `adapter_config.json` is also present):

1.  **Use `pipeline.load_lora_weights(path_to_peft_directory, adapter_name="my_lora")`:**
    *   `path_to_peft_directory`: The path to the directory (e.g., `unet_lora_peft_format`).
    *   `adapter_name`: An optional name for this adapter if loading multiple.
    *   `diffusers` (versions >= 0.19.0, ideally >=0.21.0 for best PEFT compatibility) will:
        *   Read `adapter_config.json` from this directory.
        *   Understand the `target_modules` specified (which will have PEFT's internal prefixes like `base_model.model.down_blocks...`).
        *   Correctly map these `target_modules` to the layers of the pipeline's *base* UNet (or text encoder).
        *   Load the weights from `adapter_model.safetensors` into the newly created/modified LoRA layers in the UNet (and/or text encoder).

**Example (Conceptual Python for UI Backend):**

```python
from diffusers import StableDiffusionPipeline

# Assume 'pipeline' is your loaded base StableDiffusionPipeline instance
# Assume 'lora_path_from_user' is the path to the PEFT directory 
# (e.g., ".../unet_lora_peft_format/")

try:
    # For UNet LoRA
    pipeline.load_lora_weights(
        lora_path_from_user, 
        adapter_name="unet_adaptra_lora" # Give it a unique name
    )
    print(f"Successfully loaded UNet LoRA from PEFT directory: {lora_path_from_user}")

    # If you also have a separate PEFT directory for text_encoder LoRA:
    # path_to_text_encoder_lora_dir = "..."
    # pipeline.load_lora_weights(
    #     path_to_text_encoder_lora_dir,
    #     adapter_name="text_encoder_adaptra_lora"
    # )
    # print(f"Successfully loaded Text Encoder LoRA from PEFT directory: {path_to_text_encoder_lora_dir}")

    # To activate a specific LoRA if multiple are loaded:
    # pipeline.set_adapters(["unet_adaptra_lora", "text_encoder_adaptra_lora"], adapter_weights=[1.0, 0.8])
    # Or for a single one:
    pipeline.set_adapters(["unet_adaptra_lora"], adapter_weights=[desired_lora_strength])


except ValueError as e:
    print(f"Error loading LoRA from PEFT directory {lora_path_from_user}: {e}")
    print("This might happen if target_modules in adapter_config.json cannot be found in the base model, "
          "or if the diffusers/PEFT versions are incompatible.")
except Exception as e:
    print(f"An unexpected error occurred while loading LoRA from PEFT directory: {e}")

```

**Key for GRADIENT UI Developers:**
*   Rely on `pipeline.load_lora_weights()` to handle the PEFT directory format.
*   The `ValueError: Target modules {...} not found in the base model.` error you previously saw when loading `adaptra_oil.json` suggests that your UI's loading mechanism at that time was trying to use the PEFT `adapter_config.json` directly with `peft.mapping.inject_adapter_in_model` on a *raw* UNet. `diffusers.load_lora_weights` is designed to be the bridge that correctly interprets this PEFT config for its own raw UNet.

### Scenario 2: Loading a Single-File LoRA (`.safetensors`)

This is the format that Adaptra now also attempts to save as `unet_lora_{tag}_ui.safetensors`. This file should ideally contain:
*   Only the LoRA weights (e.g., `lora_A.weight`, `lora_B.weight` for each layer).
*   Keys that are directly mappable to the layers of a *raw* UNet (e.g., `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight` without any `base_model.model.` prefix).
*   Optionally, metadata within the safetensors file itself for `alpha` (scale).

**How GRADIENT UI should handle it:**

1.  **Use `pipeline.load_lora_weights(path_to_single_safetensors_file, adapter_name="my_lora")`:**
    *   `diffusers` will attempt to load this single file.
    *   It tries to match keys from the state dictionary in the `.safetensors` file to layers in the UNet (and text encoder, if applicable and keys match).
    *   It often infers `alpha` from the weights themselves or from metadata if present in the safetensors file.
    *   An `adapter_config.json` is **not strictly required** by `diffusers.load_lora_weights` when loading a single, well-formatted LoRA safetensors file, though providing one could offer more explicit control if the UI wants to pass it.

**Example (Conceptual Python for UI Backend):**

```python
from diffusers import StableDiffusionPipeline

# Assume 'pipeline' is your loaded base StableDiffusionPipeline instance
# Assume 'single_lora_file_path' is the path to "unet_lora_steps_XX_ui.safetensors"

try:
    pipeline.load_lora_weights(
        single_lora_file_path,
        adapter_name="unet_adaptra_lora_ui"
    )
    print(f"Successfully loaded UNet LoRA from single file: {single_lora_file_path}")
    
    # Activate it
    pipeline.set_adapters(["unet_adaptra_lora_ui"], adapter_weights=[desired_lora_strength])

except Exception as e:
    print(f"An error occurred while loading single-file LoRA {single_lora_file_path}: {e}")
    print("This could be due to mismatched keys in the safetensors file, or an incompatible format.")

```

**The `ValueError: Target modules ... not found` you saw with `adaptra_oil.json` indicates that GRADIENT UI's `pipeline.load_lora_weights()` was indeed trying to use the `adapter_config.json` alongside the `.safetensors` file, and the `target_modules` in that JSON were the problem.**

**Recommendations for GRADIENT UI Developers to Fix the Error:**

1.  **Prioritize `diffusers.StableDiffusionPipeline.load_lora_weights()`:** This should be the primary method for loading any LoRA, whether it's a PEFT directory or a single file.
2.  **When loading a PEFT directory (containing `adapter_config.json`):**
    *   Ensure your UI is passing the *path to the directory* to `load_lora_weights`.
    *   Trust that `diffusers` (version >= 0.21.0 is good for PEFT compatibility) will handle the key mapping from the PEFT-prefixed `target_modules` in `adapter_config.json` to its internal base UNet layers. If it still fails with "Target modules not found," it might indicate an issue with the `diffusers` version or a very specific incompatibility.
3.  **When loading a single `.safetensors` LoRA file (like Adaptra's `_ui.safetensors`):**
    *   Pass the direct path to this single file to `load_lora_weights`.
    *   **Crucially, if an `adaptra_oil.json` (renamed `adapter_config.json`) is present in the same directory as `adaptra_oil.safetensors`, `load_lora_weights` might still try to read it and use its `target_modules`.** This would re-introduce the prefix problem.
    *   **Test:** Try loading the `unet_lora_{tag}_ui.safetensors` *without* any accompanying `.json` config file in the same directory (or ensure the UI doesn't pass the path to that json if it's trying to be "smart"). The single safetensors file created by the refined Adaptra `save_checkpoint` (using `get_peft_model_state_dict`) should have prefix-less keys, and `diffusers` should be able to apply it by matching these keys.

**If `ValueError: Target modules ... not found` persists even when GRADIENT UI calls `pipeline.load_lora_weights(PEFT_ADAPTER_DIR)`:**

This would be surprising for recent `diffusers` versions. It would imply that `diffusers` itself is not correctly remapping the PEFT-prefixed target modules from `adapter_config.json` when injecting the adapter. In such a scenario, the UI would indeed need LoRAs where the `adapter_config.json` *also* has prefix-less target modules. This would require Adaptra to save a modified, UI-specific `adapter_config.json`.

**Path for Adaptra to produce a more universally compatible "UI LoRA package":**

The current `unet_lora_{tag}_ui.safetensors` saved by Adaptra (using `get_peft_model_state_dict`) contains weights with keys that *should* be stripped of the `base_model.model.` prefix (e.g., keys like `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight`). This is good.

The missing piece for full, robust compatibility if UIs *insist* on using `adapter_config.json` and can't handle PEFT's prefixed targets is for Adaptra to also generate a *modified* `adapter_config.json` for UI use:

```python
# In Adaptra's DiffusionTrainer.save_checkpoint, after creating ui_compatible_lora_final_path

# Load the original adapter_config.json from the PEFT save
original_adapter_config_path = unet_lora_peft_dir_path / "adapter_config.json"
if original_adapter_config_path.exists():
    import json
    with open(original_adapter_config_path, 'r') as f:
        ui_adapter_config_data = json.load(f)
    
    # Modify target_modules: strip "base_model.model." prefix
    if "target_modules" in ui_adapter_config_data and isinstance(ui_adapter_config_data["target_modules"], list):
        new_target_modules = []
        for target in ui_adapter_config_data["target_modules"]:
            if isinstance(target, str):
                new_target_modules.append(target.replace("base_model.model.", ""))
            else: # Should be a list of strings if from PEFT LoraConfig
                new_target_modules.append(target) # Keep as is if not string (should not happen)
        ui_adapter_config_data["target_modules"] = new_target_modules
        
        # Save the modified adapter_config.json for the UI
        ui_adapter_config_json_path = final_output_path / f"unet_lora_{tag}_ui_adapter_config.json"
        with open(ui_adapter_config_json_path, 'w') as f:
            json.dump(ui_adapter_config_data, f, indent=4)
        log.info(f"Saved UI-compatible adapter_config.json to: {ui_adapter_config_json_path}")
```
Then, GRADIENT UI developers would be instructed to use `unet_lora_{tag}_ui.safetensors` along with `unet_lora_{tag}_ui_adapter_config.json`.

**First, ask GRADIENT UI developers to ensure they are calling `pipeline.load_lora_weights(PATH_TO_PEFT_DIRECTORY)` when the user selects the `unet_lora_peft_format` directory saved by Adaptra. This is the most `diffusers`-native way to load PEFT adapters.** If that still fails with the prefix error, then providing the stripped `adapter_config.json` becomes necessary for their specific loading implementation.