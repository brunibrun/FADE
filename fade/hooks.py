from typing import Any, Callable, Dict, Optional
import torch
import torch.nn as nn

class ActivationHooks:

    def __init__(self):
        """
        Initialize activation hooks storage and management.

        Attributes:
            activations (Dict[str, torch.Tensor]): Stores layer output activations.
            handles (Dict[str, Any]): Stores hook handles for registered hooks.
            activation_modification (Dict[str, Dict[int, float]]): Stores modifications for layer activations.
        """
        self.activations = {}
        self.handles = {}
        self.activation_modification = {}

    def register_hook(
        self, 
        module: nn.Module, 
        name: str, 
        hook_fn: Callable, 
        **kwargs
    ) -> None:
        """
        Register a forward hook to a module.

        Args:
            module (nn.Module): The neural network module to attach the hook.
            name (str): A unique identifier for the hook.
            hook_fn (Callable): The hook function to register.
            **kwargs: Additional keyword arguments to pass to the hook function.
        """
        handle = module.register_forward_hook(hook_fn(name, **kwargs))
        self.handles[name] = handle

    def remove_hook(self, name: str) -> None:
        """
        Remove a previously registered hook.

        Args:
            name (str): The unique identifier of the hook to remove.
        """
        if name in self.handles:
            handle = self.handles.pop(name)
            handle.remove()

    def set_activation_modification(self, name: str, mask: Dict[int, float]) -> None:
        """
        Set activation modification for a specific layer.

        Args:
            name (str): The layer name to modify.
            mask (Dict[int, float]): A dictionary mapping neuron indices to modification factors.
        """
        self.activation_modification[name] = mask

    def clear_activation_modification(self, name: str) -> None:
        """
        Clear activation modification for a specific layer.

        Args:
            name (str): The layer name to clear modifications for.
        """
        if name in self.activation_modification:
            self.activation_modification.pop(name)

    def save_layer_output_activation(self, name: str) -> Callable:
        """
        Create a hook function to save layer output activations.

        Args:
            name (str): The unique identifier for the layer activations.

        Returns:
            Callable: A hook function that saves the layer's output activation.
        """
        def hook(model: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            self.activations[name] = output.detach().clone()
        return hook
    
    def modify_layer_output_activation(self, name: str) -> Callable:
        """
        Create a hook function to modify layer output activations by multiplication.

        Args:
            name (str): The unique identifier for the layer activations.

        Returns:
            Callable: A hook function that modifies the layer's output activation.
        """
        def hook(model, input, output):
            if self.activation_modification.get(name) is not None:
                for neuron_index, modification_factor in self.activation_modification[name].items():
                    output[..., neuron_index] *= modification_factor
                return output
            else:
                raise ValueError(f"Multiplication mask for {name} not found")
        return hook
    
    def modify_sae_layer_output_activation(self, name: str, value: float) -> Callable:
        """
        Create a hook function to modify SAE layer output activations.

        Args:
            name (str): The unique identifier for the layer activations.
            value (float): The base value to use for modification.

        Returns:
            Callable: A hook function that modifies the SAE layer's output activation.
        """
        def hook(model, input, output):
            if self.activation_modification.get(name) is not None:
                for neuron_index, modification_factor in self.activation_modification[name].items():
                    output[..., neuron_index] = modification_factor * value
                return output
            else:
                raise ValueError(f"Multiplication mask for {name} not found")
        return hook