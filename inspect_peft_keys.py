from safetensors.torch import load_file
import pprint

# Path to your PEFT LoRA's safetensors file - to inspect peft keys
lora_file_path = "data/models/lora/unet_lora_steps_50_ui.safetensors" # Adjust path as needed inspect_peft_keys.py

try:
    state_dict = load_file(lora_file_path)
    print(f"Keys in {lora_file_path}:")
    pprint.pprint(list(state_dict.keys()))
except Exception as e:
    print(f"Error loading or inspecting LoRA file {lora_file_path}: {e}")