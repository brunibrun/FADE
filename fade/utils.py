import os
import time
from contextlib import contextmanager
import yaml
import importlib.resources
from typing import Dict, Optional, Any, TypeVar, Callable


def get_module_by_name(model: Any, module_name: str) -> Any:
    """
    Retrieve a module from a model using its string representation.

    Args:
        model (Any): The model containing the module.
        module_name (str): Dot-separated path to the module, 
                           can include indexing for nested modules.

    Returns:
        Any: The retrieved module or submodule.
    """
    parts = module_name.split('.')
    current = model
    for part in parts:
        if '[' in part and part.endswith(']'): # handle indexing
            component, idx = part.split('[')
            idx = int(idx.rstrip(']'))
            current = getattr(current, component)[idx]
        else:
            current = getattr(current, part)
    return current

def set_up_paths(
    output_folder: str, 
    neuron_module: str, 
    neuron_index: int, 
    store_data: bool = True
    ) -> str:
    """
    Set up directory paths for storing experimental results.

    Args:
        output_folder (str): Base output directory.
        neuron_module (str): Name of the neuron module.
        neuron_index (int): Index of the specific neuron.
        store_data (bool, optional): Whether to create data storage directory. Defaults to True.

    Returns:
        str: Path to the specific neuron's results directory.
    """
    if not os.path.exists(output_folder):
        raise ValueError("Output path does not exist")
    if not os.path.exists(os.path.join(output_folder, neuron_module)):
        os.makedirs(os.path.join(output_folder, neuron_module))
    result_path = os.path.join(output_folder, neuron_module, str(neuron_index))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(os.path.join(result_path, "plots")):
        os.makedirs(os.path.join(result_path, "plots"))
    if store_data:
        if not os.path.exists(os.path.join(result_path, "data")):
            os.makedirs(os.path.join(result_path, "data"))
    return result_path



@contextmanager
def timed_section(verbose: bool, message: str):
    """
    Context manager for timing code sections with optional verbose output.

    Args:
        verbose (bool): Whether to print timing information.
        message (str): Description of the timed section.
    """
    if verbose:
        start = time.time()
        print(f"{message}", end=" ... ", flush=True)  # Print the message without a newline
    yield
    if verbose:
        duration = time.time() - start
        print(f"completed in {duration:.2f} seconds")  # Add the elapsed time on the same line





def deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a nested dictionary without completely overwriting nested dictionaries.

    Args:
        original (Dict[str, Any]): The original dictionary to update.
        update (Dict[str, Any]): The dictionary with updates to apply.

    Returns:
        Dict[str, Any]: The updated dictionary.
    """
    for key, value in update.items():
        if key not in original:
            raise KeyError(f"Unexpected key in config: '{key}'")
        if isinstance(value, dict) and isinstance(original[key], dict):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original

def load_config(custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration with fallback to default settings.

    Args:
        custom_config (Optional[Dict[str, Any]], optional): 
            Dictionary of custom configuration values to override defaults. Defaults to None.

    Returns:
        Dict[str, Any]: Merged configuration dictionary with custom values overriding defaults.
    """
    with importlib.resources.open_text('fade', 'default_config.yaml') as f:
        default_config = yaml.safe_load(f)
    
    # If no custom config provided, return default
    if custom_config is None:
        return default_config
    
    # Create a copy of default config
    config = default_config.copy()
    
    # Update with custom values
    return deep_update(config, custom_config)