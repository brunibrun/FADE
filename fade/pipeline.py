import sys
import itertools
import numpy
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from fade.hooks import ActivationHooks
from fade.llm import EvaluationLLM, SubjectLLM, PydanticGenerationModel, PydanticRatingModel, RateSamples
from fade.results import GenerateResults
from fade.data import DictionaryDataset, ActivationLoader
from fade.utils import timed_section, get_module_by_name, set_up_paths, load_config


class EvaluationPipeline:
    def __init__(
           self,
            subject_model: Any, 
            subject_tokenizer: Any,

            dataset: Union[Dict[int, str], DictionaryDataset], 
            device: torch.device,
            config: Dict[str, Any],
            activations: Optional[ActivationLoader] = None,
            verbose: bool = False,
            seed: Optional[int] = None,
            ):
        """        
        Args:
            subject_model: The model to be evaluated
            subject_tokenizer: Tokenizer for the subject model
            dataset: Dictionary or DictionaryDataset of text samples
            device: Device to run the model on
            config: Configuration dictionary
            activations: Optional activation loader for cached activations
            verbose: Whether to print progress information
            seed: Random seed for reproducibility
        """
        self.subject_model = subject_model
        self.subject_tokenizer = subject_tokenizer
        self.dataset = dataset if isinstance(dataset, DictionaryDataset) else DictionaryDataset(dataset)
        self.config = load_config(config)
        self.device = device
        self.cached_activations = activations
        self.verbose = verbose
        self.seed = seed

        self.set_seed()

        self.generation_llm = EvaluationLLM(
            model_kind=self.config["evaluationLLM"]["type"],
            model_name=self.config["evaluationLLM"]["name"],
            system_prompt=self.config["prompts"]["generation"],
            pydantic_model=PydanticGenerationModel,
            openai_key=self.config["evaluationLLM"]["api_key"],
            base_url=self.config["evaluationLLM"]["base_url"],
            api_version=self.config["evaluationLLM"]["api_version"]
            )

        self.rating_llm = EvaluationLLM(
            model_kind=self.config["evaluationLLM"]["type"],
            model_name=self.config["evaluationLLM"]["name"],
            system_prompt=self.config["prompts"]["rating"],
            pydantic_model=PydanticRatingModel,
            openai_key=self.config["evaluationLLM"]["api_key"],
            base_url=self.config["evaluationLLM"]["base_url"],
            api_version=self.config["evaluationLLM"]["api_version"]
            )
                
    def set_seed(self) -> None:
        """Set the random seed for reproducibility."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            numpy.random.seed(self.seed)

    def prepare_tokenizer(self, padding_side: str = "right") -> None:
        """Prepare the tokenizer with the appropriate padding settings."""
        if self.subject_tokenizer.pad_token is None:
            self.subject_tokenizer.pad_token = self.subject_tokenizer.eos_token
        self.subject_tokenizer.padding_side = padding_side

    @torch.no_grad()
    def run(self, neuron_module: str, neuron_index: int, concept: str) -> Dict[str, float]:
        """Run the evaluation pipeline for a specific neuron and concept.
        
        Args:
            neuron_module: The module name containing the neuron
            neuron_index: The index of the neuron within the module
            concept: The concept to evaluate for the neuron
            
        Returns:
            Dict with evaluation metrics (clarity, responsiveness, purity, faithfulness)
        """
        # set up paths and results if necessary
        store_data = self.config["experiments"]["store_data"]

        self.result_path = set_up_paths(
            output_folder=self.config["paths"]["output_path"],
            neuron_module=neuron_module,
            neuron_index=neuron_index,
            store_data=store_data
        ) if store_data else None

        self.generate_results = GenerateResults(
            output_path=self.result_path,
            store_data=store_data,
        )
    
        if self.verbose:
            print(f"\nRunning Evaluation for \n- Module: {neuron_module} and Neuron: {neuron_index} \n- Concept: {concept}")

        with timed_section(self.verbose, "Activation centric Pipeline"):
            activation_centric_pipeline = ActivationCentricPipeline(self)
            clarity, responsiveness, purity = activation_centric_pipeline.run_pipeline(neuron_module, neuron_index, concept)

        # Only run output-centric pipeline if activation-centric results are promising
        faithfulness = None
        threshold = self.config["experiments"]["gini_threshold"]
        if clarity is not None and responsiveness is not None and clarity >= threshold and responsiveness >= threshold:
            with timed_section(self.verbose, "Output centric Pipeline"):
                output_centric_pipeline = OutputCentricPipeline(self)
                faithfulness = output_centric_pipeline.run_pipeline(neuron_module, neuron_index, concept)

        # clean up state
        self.sae_max_value = None

        return self.generate_results.combine_results(clarity, responsiveness, purity, faithfulness, neuron_index, concept)


class ActivationCentricPipeline:
    def __init__(self, evaluation_pipeline: EvaluationPipeline):
        """Initialize the activation-centric pipeline.
        
        Args:
            evaluation_pipeline: The main evaluation pipeline
        """
        self.pipeline = evaluation_pipeline 
        self.rating_function = self._setup_rating_function()

    def _setup_rating_function(self) -> RateSamples:
        """Set up the rating function with configuration parameters."""
        config = self.pipeline.config["experiments"]["responsiveness_and_purity"]
        return RateSamples(
            evaluation_llm=self.pipeline.rating_llm,
            num_samples=config["num_samples"], 
            max_failed_retries=config["max_failed_retries"],
            max_sparse_retries=config["max_sparse_retries"],
            retry_sparse_threshold=config["retry_sparse_threshold"],
            repeat_non_zeros=config["repeat_non_zeros"],
            top_sampling=config["rating_top_sampling_percentage"],
            batch_size=self.pipeline.config["experiments"]["rating_batch_size"]
        )

    def generate_activations(self, subject_llm_function: Callable, dataset: DictionaryDataset) -> torch.Tensor:
        """Generate activations for a dataset using the subject model.
        
        Args:
            subject_llm_function: Function to extract activations
            dataset: Dataset of text samples
            
        Returns:
            Tensor of activations for each sample
        """
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.pipeline.config["subjectLLM"]["batch_size"], 
            shuffle=False
            )
        
        activations = []
        for keys, batch in dataloader:
            token_ids = self.pipeline.subject_tokenizer(batch, padding=True, return_tensors="pt").to(self.pipeline.device)
            activation = subject_llm_function(token_ids).cpu()
            sequence_lengths = token_ids["attention_mask"].sum(dim=1)
            for i in range(activation.shape[0]):
                argmax_index = torch.argmax(torch.abs(activation[i, :sequence_lengths[i]]))
                activations.append(activation[i, argmax_index].item())

        return torch.tensor(activations)

    def generate_synthetic_data(self, concept: str) -> DictionaryDataset:
        """Generate synthetic data for the given concept.
        
        Args:
            concept: The concept to generate data for
            
        Returns:
            Dataset of generated samples
        """
        # Generate prompts and get responses
        prompts = [concept for _ in range(self.pipeline.config["experiments"]["clarity"]["llm_calls"])] 
        synthetic_results = self.pipeline.generation_llm.run_chain(prompts)
        
        # Filter valid responses and flatten
        valid_responses = [d for d in synthetic_results if d is not None]
        if not valid_responses:
            print("No synthetic data generated!", file=sys.stderr)
            return DictionaryDataset({})
            
        unique_examples = list(set(itertools.chain(*valid_responses)))
        
        # Create dataset
        synthetic_dict = {i: text for i, text in enumerate(unique_examples)}
        return DictionaryDataset(synthetic_dict)
    
    def hook_lm(self, neuron_module: str, neuron_index: int) -> Tuple[ActivationHooks, SubjectLLM]:
        """Set up hooks for the activation-centric pipeline.
        
        Args:
            neuron_module: Module containing the neuron
            neuron_index: Index of the neuron to analyze
            
        Returns:
            Tuple of (hooks, subject_llm_function)
        """
        hooks = ActivationHooks()
        hooks.register_hook(
            module=get_module_by_name(self.pipeline.subject_model, neuron_module),
            name=neuron_module,
            hook_fn=hooks.save_layer_output_activation)
        def get_activations(tokens: Dict[str, torch.Tensor]) -> torch.Tensor:
            """Extract activations for the specific neuron."""
            input_ids=tokens["input_ids"]
            attention_mask=tokens["attention_mask"]
            _ = self.pipeline.subject_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            activations = hooks.activations[neuron_module][:, :, neuron_index]
            return activations
        subject_llm_function = SubjectLLM(forward_function=get_activations)
        return hooks, subject_llm_function
    

    @torch.no_grad()
    def run_pipeline(self, neuron_module: str, neuron_index: int, concept: str) -> Tuple[float, float, float]:
        """Run the activation-centric pipeline.
        
        Args:
            neuron_module: Module containing the neuron
            neuron_index: Index of the neuron to analyze
            concept: Concept to evaluate
            
        Returns:
            Tuple of (clarity, responsiveness, purity) metrics
        """
        # set up hooks and prepare tokenizer
        self.pipeline.subject_model.to(self.pipeline.device)
        hooks, subject_llm_function = self.hook_lm(neuron_module, neuron_index)
        self.pipeline.prepare_tokenizer(padding_side="right")

        # generate synthetic data
        with timed_section(self.pipeline.verbose, "\n- Creating Synthetic Data"): 
            synthetic_dataset = self.generate_synthetic_data(concept=concept) 

        # generate synthetic activations
        with timed_section(self.pipeline.verbose, "- Synthetic Activations"):
            synthetic_activations = self.generate_activations(subject_llm_function, synthetic_dataset)

        # generate or fetch natural activations
        with timed_section(self.pipeline.verbose, "- Natural Activations"):
            if self.pipeline.cached_activations:
                natural_activations = self.pipeline.cached_activations(neuron_index=neuron_index)
            else:
                natural_activations = self.generate_activations(subject_llm_function, self.pipeline.dataset)
        
        # handle SAE module if needed
        if self.pipeline.config["subjectLLM"]["sae_module"]:
            self.pipeline.sae_max_value = natural_activations.abs().max().item()

        hooks.remove_hook(name=neuron_module)

        # generate ratings
        with timed_section(self.pipeline.verbose, "- Collecting Ratings"):
            natural_ratings = self.rating_function.rate_samples(
                concept=concept, 
                dataset=self.pipeline.dataset, 
                activations=natural_activations
                )

        # generate results
        return self.pipeline.generate_results.activation_centric_results(
            natural_dataset=self.pipeline.dataset,
            synthetic_dataset=synthetic_dataset,
            natural_activations=natural_activations,
            synthetic_activations=synthetic_activations,
            natural_ratings=natural_ratings
            )


class OutputCentricPipeline:
    def __init__(self, evaluation_pipeline: EvaluationPipeline):
        """Initialize the output-centric pipeline.
        
        Args:
            evaluation_pipeline: The main evaluation pipeline
        """
        self.pipeline = evaluation_pipeline
        self.rating_function = self._setup_rating_function()

    def _setup_rating_function(self) -> RateSamples:
        """Set up the rating function with configuration parameters."""
        config = self.pipeline.config["experiments"]["faithfulness"]
        return RateSamples(
            evaluation_llm=self.pipeline.rating_llm,
            batch_size=self.pipeline.config["experiments"]["rating_batch_size"],
            num_samples=None, 
            max_failed_retries=config["max_failed_retries"],
            max_sparse_retries=config["max_sparse_retries"],
            retry_sparse_threshold=config["retry_sparse_threshold"],
            repeat_non_zeros=config["repeat_non_zeros"],
            top_sampling=config["rating_top_sampling_percentage"],
        )

    def batch_prompt_generator(self, concept: str, samples: Dict[int, str], batch_size: int) -> List[str]:
        """Generate batch prompts for rating.
        
        Args:
            concept: The concept to evaluate
            samples: Dictionary of samples to rate
            batch_size: Number of samples per batch
            
        Returns:
            List of prompts for the rating LLM
        """
        prompts = []
        for i in range(0, len(samples), batch_size):
            batch_indices = list(samples.keys())[i:i + batch_size]
            sequences = [f"Sequence ID {key}: {sample}" for key, sample in samples.items() if key in batch_indices]
            prompts.append("Concept: " + concept + "\n" + "\n".join(sequences))
        return prompts
    
    def batch_generate_sequences(self, subject_llm_function: Callable, dataloader: DataLoader) -> Dict[int, str]:
        """Generate sequence continuations for each item in the dataloader.
        
        Args:
            subject_llm_function: Function to generate sequences
            dataloader: DataLoader containing samples
            
        Returns:
            Dictionary mapping sample IDs to generated sequences
        """
        generated_sequences = {}
        for keys, batch in dataloader:
            keys = [int(key.item()) for key in keys]
            token_ids = self.pipeline.subject_tokenizer(batch, padding=True, return_tensors="pt").to(self.pipeline.device)
            generated_sequence = subject_llm_function(token_ids).cpu()[:, -self.pipeline.config["experiments"]["faithfulness"]["generation_length"]:].squeeze()
            generated_sequence = [self.pipeline.subject_tokenizer.decode(sequence, skip_special_tokens=True) for sequence in generated_sequence]
            for key, sequence in zip(keys, generated_sequence):
                generated_sequences[key] = sequence
        return generated_sequences

    def hook_lm(self, neuron_module: str) -> Tuple[ActivationHooks, SubjectLLM]:
        """Set up hooks for the output-centric pipeline.
        
        Args:
            neuron_module: Module containing the neuron
            neuron_index: Index of the neuron to analyze (optional)
            
        Returns:
            Tuple of (hooks, subject_llm_function)
        """
        hooks = ActivationHooks()
        if self.pipeline.config["subjectLLM"]["sae_module"]:
            if hasattr(self.pipeline, "sae_max_value") and self.pipeline.sae_max_value is not None:
                sae_value = self.pipeline.sae_max_value
            else:
                print("Warning: SAE steering value not set, using default value of 1.0", file=sys.stderr)
                sae_value = 1.0

            hooks.register_hook(
                module=get_module_by_name(self.pipeline.subject_model, neuron_module),
                name=neuron_module,
                hook_fn=hooks.modify_sae_layer_output_activation,
                value=sae_value
                )
        else:
            hooks.register_hook(
                module=get_module_by_name(self.pipeline.subject_model, neuron_module),
                name=neuron_module,
                hook_fn=hooks.modify_layer_output_activation
                )
        def generate_sequences(token_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
            """Generate sequences using the subject model."""
            output = self.pipeline.subject_model.generate(
                input_ids=token_ids["input_ids"], 
                attention_mask=token_ids["attention_mask"],
                max_new_tokens=self.pipeline.config["experiments"]["faithfulness"]["generation_length"],
                min_new_tokens=self.pipeline.config["experiments"]["faithfulness"]["generation_length"],
                )
            return output
        subject_llm_function = SubjectLLM(forward_function=generate_sequences)
        return hooks, subject_llm_function


    @torch.no_grad()
    def run_pipeline(self, neuron_module: str, neuron_index: int, concept: str) -> float: 
        """Run the output-centric pipeline.
        
        Args:
            neuron_module: Module containing the neuron
            neuron_index: Index of the neuron to analyze
            concept: Concept to evaluate
            
        Returns:
            Faithfulness metric
        """
        # set up hooks and prepare tokenizer
        self.pipeline.subject_model.to(self.pipeline.device)
        hooks, subject_llm_function = self.hook_lm(neuron_module)
        self.pipeline.prepare_tokenizer(padding_side="left")

        # sample starting sequences
        num_samples = self.pipeline.config["experiments"]["faithfulness"]["num_samples"]
        sample_indices = numpy.random.choice(len(self.pipeline.dataset), num_samples, replace=False)
        subset = Subset(self.pipeline.dataset, sample_indices)
        dataloader = DataLoader(subset, batch_size=self.pipeline.config["subjectLLM"]["batch_size"], shuffle=False)

        # generate continuations under modification
        sequences = {}
        modification_factors = self.pipeline.config["experiments"]["faithfulness"]["modification_factors"]
        if 0 not in modification_factors: # add baseline factor if not present already
            modification_factors = [0] + modification_factors
        for modification_factor in modification_factors:
            with timed_section(self.pipeline.verbose, f"- Generating Sequences for Modification Factor {modification_factor}"):
                hooks.set_activation_modification(neuron_module, {neuron_index: modification_factor})
                generated_sequences = self.batch_generate_sequences(subject_llm_function, dataloader)
                sequences[modification_factor] = generated_sequences
                hooks.clear_activation_modification(neuron_module)
        hooks.remove_hook(name=neuron_module)

        # rate generated sequences
        ratings_dict = {}
        with timed_section(self.pipeline.verbose, "- Collecting Ratings"):
            for modification_factor, generated_sequences in sequences.items():
                ratings = self.rating_function.rate_samples(concept=concept, dataset=generated_sequences, activations=None)
                ratings_dict[modification_factor] = ratings

        # generate results
        return self.pipeline.generate_results.output_centric_results(sequences, ratings_dict)