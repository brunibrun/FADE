
import copy
import torch
from torch import nn
from tqdm import tqdm

from fade.hooks import ActivationHooks
from fade.utils import get_module_by_name
from fade.data import DictionaryDataset



@torch.no_grad()
def generate_activations(model, tokenizer, dataset, named_module, device, batch_size):
    """
    Extract activations from a specific model module for all samples in a dataset.
    
    Args:
        model: The model to extract activations from
        tokenizer: Tokenizer for processing text data
        dataset: Dictionary of {id: text} pairs
        named_module: String path to the target module in the model
        device: Device to run inference on
        batch_size: Number of samples to process at once
        
    Returns:
        Dictionary mapping sample IDs to their corresponding activations
    """
    stored_activations = {}

    # create dataloader 
    def simple_collate(batch):
        keys, values = zip(*batch)
        return list(keys), list(values)
    dataloader = torch.utils.data.DataLoader(DictionaryDataset(dataset), batch_size=batch_size, shuffle=False, collate_fn=simple_collate)

    # hook module and create activation function
    hooks = ActivationHooks()
    hooks.register_hook(
        module=get_module_by_name(model, named_module),
        name=named_module,
        hook_fn=hooks.save_layer_output_activation)

    def activation_function(tokens):
        _ = model(tokens["input_ids"], attention_mask=tokens["attention_mask"], return_dict=True)
        activations = hooks.activations[named_module]
        return activations

    # iterate over dataloader
    for batch_ids, batch_sequences in tqdm(dataloader, total=len(dataloader), mininterval=0.5):

        tokens = tokenizer(batch_sequences, padding=True, return_tensors="pt").to(device)
        activations = activation_function(tokens).detach().cpu()

        for i, id in enumerate(batch_ids):
            sequence_length = tokens["attention_mask"][i].sum().item()
            sample_activation = activations[i, :sequence_length, :].squeeze()
            stored_activations[id] = sample_activation

    return stored_activations


class SAEModuleWrapper(nn.Module):
    """
    Wrapper module that embeds Sparse Autoencoder (SAE) modules into a model's computational graph.
    
    This wrapper preserves the original module's behavior while adding SAE encoding/decoding,
    allowing activations to be captured for interpretability analysis. It also allows for steering.
    
    Args:
        original_module: The model module to wrap
        additional_modules: List of SAE modules to embed (typically [encoder, decoder])
        active: Whether the SAE processing is active (if False, behaves like original module)
    """
    def __init__(self, original_module, additional_modules=None, active=True):
        super(SAEModuleWrapper, self).__init__()
        self.original_module = original_module
        self.additional_modules = torch.nn.ModuleList(additional_modules) if additional_modules else torch.nn.ModuleList()
        self.active = active
        for module in self.additional_modules:
            module.to(next(self.original_module.parameters()).device)
        self.encoder_copy = copy.deepcopy(self.additional_modules[0])


    def forward(self, *args, **kwargs):
        # Pass the inputs through the original module
        original_output = self.original_module(*args, **kwargs)

        if not self.active:
            return original_output

        # If the output is a tuple, handle it differently # Assumes that the first argument is the input to the additional modules
        if isinstance(original_output, tuple):
            sae_input = original_output[0]  # Apply additional module to the first element
        else:
            sae_input = original_output  # If it's not a tuple, just pass the original output

        # Apply the encoder and decoder in sequence
        sae_encoded = self.additional_modules[0](sae_input)
        sae_decoded = self.additional_modules[1](sae_encoded)

        # add first token of the original output to the additional modules output # Assumes shape (batch_size, seq_len, hidden_size) # This is done to avoid BOS outlier problems
        sae_decoded[:, 0, :] = sae_input[:, 0, :]

        # do error correction
        sae_decoded_copy = self.additional_modules[1](self.encoder_copy(sae_input))
        sae_decoded_copy[:, 0, :] = sae_input[:, 0, :]
        error  = sae_input - sae_decoded_copy
        sae_decoded += error.detach()

        # If the original output was a tuple, return a tuple with modified first element
        if isinstance(original_output, tuple):
            return (sae_decoded,) + original_output[1:]
        else:
            return sae_decoded  # Otherwise, return the modified output directly


class JumpReLUSAEEncodeNoBOS(nn.Module):
  def __init__(self, d_model, d_sae):
    super().__init__()
    self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
    self.threshold = nn.Parameter(torch.zeros(d_sae))
    self.b_enc = nn.Parameter(torch.zeros(d_sae))

  def forward(self, acts):
    pre_acts = acts @ self.W_enc + self.b_enc
    mask = (pre_acts > self.threshold)
    return mask * torch.nn.functional.relu(pre_acts)


class JumpReLUSAEDecode(nn.Module):
  def __init__(self, d_model, d_sae):
    super().__init__()
    self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
    self.b_dec = nn.Parameter(torch.zeros(d_model))

  def forward(self, acts):
    return acts @ self.W_dec + self.b_dec