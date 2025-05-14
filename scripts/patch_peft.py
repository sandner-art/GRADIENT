def apply_patch():
    import os
    import importlib.util
    
    # Find the PEFT installation path
    spec = importlib.util.find_spec('peft')
    if spec is None:
        print("PEFT library not found")
        return
    
    peft_path = os.path.dirname(spec.origin)
    loftq_utils_path = os.path.join(peft_path, 'utils', 'loftq_utils.py')
    
    with open(loftq_utils_path, 'r') as f:
        content = f.read()
    
    # Replace the import statement with the proper function from newer Accelerate versions
    modified_content = content.replace(
        "from accelerate.utils.memory import clear_device_cache",
        """
try:
    from accelerate.utils.memory import clear_device_cache
except ImportError:
    # Use proper equivalent from newer Accelerate versions
    from accelerate.utils.memory import empty_cache as clear_device_cache
"""
    )
    
    # Write the modified file
    with open(loftq_utils_path, 'w') as f:
        f.write(modified_content)
    
    print(f"Successfully patched {loftq_utils_path}")

if __name__ == "__main__":
    apply_patch()