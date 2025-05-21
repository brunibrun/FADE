import sys
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, Set, Type
import numpy as np
import torch

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, Field, field_validator

from fade.data import DictionaryDataset


class SubjectLLM:
    def __init__(self, forward_function: Callable):
        """
        Initialize a Subject Language Model wrapper.

        Args:
            forward_function (Callable): The forward pass function for the LLM.
        """
        self.forward_function = forward_function

    def __call__(self, tokens: Any, **kwargs) -> Any:
        """
        Call the forward function with tokens and optional keyword arguments.

        Args:
            tokens (Any): Input tokens for the forward function.
            **kwargs: Additional keyword arguments to pass to the forward function.        

        Returns:
            Any: The output of the forward function.
        """
        return self.forward_function(tokens, **kwargs)


class EvaluationLLM:
    def __init__(
        self, 
        model_kind: str, 
        model_name: str, 
        system_prompt: str, 
        pydantic_model: Type[BaseModel], 
        openai_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        api_version: Optional[str] = None
    ):
        """
        Initialize an Evaluation Language Model with specific configuration.

        Args:
            model_kind (str): Type of language model.
            model_name (str): Name or identifier of the specific model.
            system_prompt (str): Initial system prompt for context setting.
            pydantic_model (type[BaseModel]): Pydantic model for parsing output.
            openai_key (Optional[str], optional): API key for OpenAI-like services. Defaults to None.
            base_url (Optional[str], optional): Base URL for the LLM service. Defaults to None.
            api_version (Optional[str], optional): API version for services like Azure. Defaults to None.
        """
        self.model_kind = model_kind
        self.model_name = model_name
        self.openai_key = openai_key
        self.base_url = base_url
        self.api_version = api_version

        # set up prompt and parser
        self.system_prompt = system_prompt
        self.prompt = ChatPromptTemplate([
            ("system", system_prompt),
            ("user", "{query}\n")
            ])
        
        self.pydantic_model = pydantic_model
        self.parser = PydanticOutputParser(pydantic_object=pydantic_model)

        # set up llm
        self.init_llm()

        # set up chain
        self.init_chain()

    def init_llm(self) -> None:
        """
        Initialize the language model based on the specified model kind.
        """
        if self.model_kind == 'openai': 
            self.llm = ChatOpenAI(openai_api_key=self.openai_key, model=self.model_name, openai_api_base=self.base_url)
        elif self.model_kind == 'azure':
            self.llm = AzureChatOpenAI(azure_endpoint=self.base_url, api_key=self.openai_key, model=self.model_name, api_version=self.api_version)
        elif self.model_kind == 'ollama':
            self.llm = ChatOllama(model=self.model_name, base_url=self.base_url)
        else:
            raise ValueError('Model kind not recognized')

    def init_chain(self) -> None:
        """
        Initialize the processing chain for the language model.

        The chain combines the prompt template, language model, and a safe parsing function.
        """
        def safe_parse(output: Union[str, Any]) -> Optional[Dict[str, Any]]:
            """
            Safely parse the LLM output using the specified Pydantic model.

            Args:
                output (Union[str, Any]): The output from the language model.

            Returns:
                Optional[Dict[str, Any]]: Parsed results or None if parsing fails.
            """
            try:
                # Extracting just the response string
                if hasattr(output, 'content'):
                    output = output.content

                # add prefix to turn into pydanctic object and parse
                prefixed_output = f'{{"results": {output.strip()}}}'
                return self.parser.parse(prefixed_output).model_dump()["results"]
            
            except OutputParserException as e:
                print(f"Parsing error: {str(e)[:50]}", file=sys.stderr)
                      
        self.chain = self.prompt | self.llm | (safe_parse)

    def run_chain(self, query: Union[str, List[str]]) -> Any:
        """
        Run the processing chain on a query or list of queries.

        Args:
            query (Union[str, List[str]]): A single query string or a list of query strings.

        Returns:
            Any: The processed result(s) from the chain.
               - For a single string: Returns the parsed result
               - For a list: Returns a list of results (may include exceptions)
        """
        if type(query) == str:
            return self.chain.invoke({"query": query})
        elif type(query) == list:
            return self.chain.batch(query, return_exceptions=True)


class PydanticGenerationModel(BaseModel): 
    results : List[str] = Field(description="List of sequences that represent the concept")
    @field_validator("results")
    def check_results(cls, v: List[str]) -> Optional[List[str]]:
        """
        Validate the results list to ensure it contains only strings.

        Args:
            v (List[str]): The list of results to validate.

        Returns:
            Optional[List[str]]: The validated list of strings, or None if invalid.
        """
        if isinstance(v, List) and all(isinstance(sequence, str) for sequence in v):
            return v
        else:
            print(f"Invalid results detected: {str(v)[:50]}. Results must be a list of strings.", file=sys.stderr)
            return None
        

class PydanticRatingModel(BaseModel):
    results : Dict[str, int] = Field(description="Dict of ratings. Each value is the rating of a sequence")
    @field_validator("results")
    def check_results(cls, v: Dict[str, int]) -> Optional[Dict[str, int]]:
        """
        Validate the ratings dictionary to ensure values are between 0 and 3.

        Args:
            v (Dict[str, int]): The dictionary of ratings to validate.

        Returns:
            Optional[Dict[str, int]]: The validated ratings dictionary, or None if invalid.
        """
        if isinstance(v, dict) and all(rating in [0,1,2,3] for rating in v.values()):
            return v
        else:
            print(f"Invalid ratings detected: {str(v)[:50]}. Ratings must be int between 0 and 4", file=sys.stderr)
            return None
        


class RateSamples: # TODO simplify this class and make more efficient!
    def __init__(
        self, 
        evaluation_llm: 'EvaluationLLM', 
        batch_size: int,
        num_samples: Optional[int] = None, 
        max_failed_retries: int = 1, 
        max_sparse_retries: int = 1,
        retry_sparse_threshold: int = 30, 
        repeat_non_zeros: int = 0,
        top_sampling: float = 0.1,
    ):
        """
        Initialize a sample rating process with configurable parameters.

        Args:
            evaluation_llm (EvaluationLLM): The language model used to explain and rate concepts.
            batch_size (int): Number of samples to process in each batch.
            num_samples (Optional[int], optional): Total number of samples to rate. Defaults to None.
            max_failed_retries (int, optional): Maximum number of retries for failed sample ratings. Defaults to 1.
            max_sparse_retries (int, optional): Maximum number of retries for generating additional samples 
                                                when concept samples are sparse. Defaults to 1.
            retry_sparse_threshold (int, optional): Minimum number of non-zero ratings to consider 
                                                    the concept sufficiently sampled. Defaults to 30.
            repeat_non_zeros (int, optional): Number of times to repeat ratings for non-zero samples. Defaults to 0.
            top_sampling (float, optional): Fraction of samples to select from top activations. Defaults to 0.1.
        """
        self.evaluation_llm = evaluation_llm
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.max_failed_retries = max_failed_retries
        self.max_sparse_retries = max_sparse_retries
        self.retry_sparse_threshold = retry_sparse_threshold
        self.repeat_non_zeros = repeat_non_zeros
        self.top_sampling = top_sampling

    def rate_samples(
        self, 
        concept: str,
        dataset: DictionaryDataset,
        activations: Optional[torch.Tensor] = None
    ) -> Dict[str, int]:
        """
        Rate samples based on a given concept using the evaluation language model.

        Args:
            concept (str): The concept to evaluate samples against.
            dataset (DictionaryDataset): The dataset of samples to rate. Each sample is a tuple of (id, text).
            activations (Optional[torch.Tensor], optional): Activation values for the samples. Defaults to None.

        Returns:
            Dict[str, int]: A dictionary of ratings for the samples, 
                            where keys are sample identifiers and values are rating scores.
        """
        rating_set = self.select_samples(dataset, activations, self.num_samples, self.top_sampling)
        all_ratings, failed_samples = self.generate_ratings(concept, rating_set, self.batch_size)

        if self.max_failed_retries > 0:
            all_ratings = self.retry_failed_samples(concept, dataset, failed_samples, self.max_failed_retries, all_ratings)

        if self.max_sparse_retries > 0: 
            if activations is not None and self.num_samples is not None:
                all_ratings = self.handle_sparse_concepts(concept, dataset, all_ratings, self.retry_sparse_threshold, activations, self.max_sparse_retries, self.num_samples)

        if self.repeat_non_zeros > 0:
            all_ratings = self.repeat_ratings(concept, dataset, all_ratings, self.repeat_non_zeros)

        return all_ratings


    def select_samples(
        self, 
        dataset: DictionaryDataset,
        activations: Optional[torch.Tensor], 
        num_samples: Optional[int], 
        top_sampling: float = 0.1, 
    ) -> Dict[int, str]:
        """
        Select a subset of samples for rating, potentially based on activation values.

        Args:
            dataset (DictionaryDataset): The dataset of samples to select from.
            activations (Optional[torch.Tensor]): Activation values for the samples.
            num_samples (Optional[int]): Number of samples to select.
            top_sampling (float, optional): Fraction of samples to select from top activations. Defaults to 0.1.

        Returns:
            Dict[int, str]: A subset of samples to rate.
        """
        if num_samples is not None and activations is not None:
            indices = self.get_activation_based_subset(activations, num_samples, top_sampling=top_sampling)
            return {index: dataset[index][1] for index in indices}
        return dataset


    def get_activation_based_subset(
        self, 
        activations: torch.Tensor, 
        num_samples: int, 
        top_sampling: float = 0.1, 
    ) -> List[int]:
        """
        Sample a subset of the natural dataset based on their activation distribution.

        Args:
            activations (torch.Tensor): Tensor of activation values.
            num_samples (int): Number of samples to select.
            top_sampling (float, optional): Fraction of samples to select from top activations. Defaults to 0.1.

        Returns:
            List[int]: Indices of the sampled subset.
        """
        if len(activations) <= num_samples:
            return list(range(len(activations)))

        sorted_indices = torch.argsort(activations).tolist()

        percentiles = [0, 50, 75, 95, 100]
        percentile_indices = {}
        for i in range(len(percentiles) - 1):
            start_idx = int(percentiles[i] / 100 * len(sorted_indices))
            stop_idx = int(percentiles[i + 1] / 100 * len(sorted_indices))
            percentile_indices[percentiles[i]] = sorted_indices[start_idx:stop_idx]

        samples_per_percentile = int(num_samples * (1 - top_sampling) // (len(percentiles) - 1))

        sampled_indices = []
        for samples in percentile_indices.values():
            if len(samples) > samples_per_percentile:
                sampled_indices.extend(np.random.choice(samples, samples_per_percentile, replace=False))
            else:
                sampled_indices.extend(samples)

        remaining_samples = num_samples - len(sampled_indices)
        sampled_indices.extend(sorted_indices[-remaining_samples:])

        return list(set(sampled_indices))


    def generate_ratings(
        self, 
        concept: str, 
        rating_set: Dict[int, str], 
        batch_size: int
    ) -> Tuple[Dict[str, int], Set[int]]:
        """
        Generate ratings for a set of samples using the evaluation language model.

        Args:
            concept (str): The concept to evaluate samples against.
            rating_set (Dict[int, str]): The set of samples to rate. Each sample is a tuple of (id, text).
            batch_size (int): Number of samples to process in each batch.

        Returns:
            Tuple[Dict[str, int], Set[int]]: 
            - A dictionary of ratings for the samples. Keys are sample identifiers and values are rating scores.
            - A set of sample IDs that failed to generate ratings
        """
        prompts = []
        all_prompts = [f"Sequence ID {key}: '{sample}'" for key, sample in rating_set.items()]
        for batch_id in range(0, len(rating_set), batch_size):
            prompts.append("Concept: " + concept + "\n" + "\n".join(all_prompts[batch_id:batch_id + batch_size]))
        ratings = self.evaluation_llm.run_chain(prompts)
        ratings = [d for d in ratings if isinstance(d, dict) and all(isinstance(k, str) and isinstance(v, int) for k, v in d.items())]        
        ratings = {k: v for d in ratings if d is not None for k, v in d.items()}
        failed_samples = {id for id in rating_set if str(id) not in ratings}
        return ratings, failed_samples

    def retry_failed_samples(
        self, 
        concept: str, 
        dataset: DictionaryDataset,
        failed_samples: Set[int], 
        max_retries: int, 
        all_ratings: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Retry generating ratings for samples that failed in the initial pass.

        Args:
            concept (str): The concept to evaluate samples against.
            dataset (DictionaryDataset): The dataset of samples to rate.
            failed_samples (Set[int]): Set of sample IDs that failed to generate ratings.
            max_retries (int): Maximum number of retry attempts.
            all_ratings (Dict[str, int]): Current ratings dictionary.

        Returns:
            Dict[str, int]: Updated ratings dictionary after retrying failed samples.
        """
        retries = 0
        while retries < max_retries and len(failed_samples) > 0:
            retries += 1
            retry_set = {id: dataset[id][1] for id in failed_samples}
            retry_ratings, retry_failed = self.generate_ratings(concept, retry_set, self.batch_size)
            all_ratings.update(retry_ratings)
            failed_samples = retry_failed
        return all_ratings

    def handle_sparse_concepts(
        self, 
        concept: str, 
        dataset: DictionaryDataset,
        all_ratings: Dict[str, int], 
        retry_sparse_threshold: int, 
        activations: Optional[torch.Tensor], 
        max_sparse_retries: int, 
        num_samples: Optional[int], 
    ) -> Dict[str, int]:
        """
        Handle sparse concepts by generating additional samples when concept ratings are low.

        Args:
            concept (str): The concept to evaluate samples against.
            dataset (DictionaryDataset): The dataset of samples to rate.
            all_ratings (Dict[str, int]): Current ratings dictionary.
            retry_sparse_threshold (int): Minimum number of concept ratings to consider sufficient.
            activations (Optional[torch.Tensor]): Activation values for the samples.
            max_sparse_retries (int): Maximum number of attempts to generate additional samples.
            num_samples (Optional[int]): Number of samples to select.

        Returns:
            Dict[str, int]: Updated ratings dictionary after handling sparse concepts.
        """
        sparse_retries = 0
        while sparse_retries < max_sparse_retries:
            concept_count = sum(1 for rating in all_ratings.values() if rating == 2)
            if concept_count >= retry_sparse_threshold:
                break
            sparse_retries += 1
            additional_samples = self.get_additional_samples_for_sparse_concepts(dataset=dataset, existing_ratings=all_ratings, activations=activations, num_samples=num_samples)
            sparse_ratings, _ = self.generate_ratings(concept, additional_samples, self.batch_size)
            all_ratings.update(sparse_ratings)
        return all_ratings

    def get_additional_samples_for_sparse_concepts(
        self, 
        dataset: DictionaryDataset,
        existing_ratings: Dict[str, int], 
        activations: Optional[torch.Tensor], 
        num_samples: Optional[int], 
    ) -> Dict[int, str]:
        """
        Get additional samples for sparse concepts, avoiding previously rated samples.

        Args:
            dataset (DictionaryDataset): The dataset of samples to select from.
            existing_ratings (Dict[str, int]): Current ratings to avoid re-selecting samples.
            activations (Optional[torch.Tensor]): Activation values for the samples.
            num_samples (Optional[int]): Number of additional samples to select.

        Returns:
            Dict[int, str]: A dictionary of additional samples to rate. Keys are sample identifiers and values are sample texts.
        """
        
        # Identify unrated samples
        rated_indices = list(set(existing_ratings.keys()))
        # convert to int to avoid issues with json serialization
        rated_indices = [int(index) for index in rated_indices]
        unrated_indices = list(set(range(len(dataset))) - set(rated_indices))

        # return all samples if if there are not enough unrated samples
        if len(unrated_indices) <= num_samples:
            return {index: dataset[index] for index in unrated_indices}

        # Select additional samples based on activations, if provided
        if activations is not None:
            mask = torch.ones(activations.size(0), dtype=torch.bool)
            mask[rated_indices] = False
            unrated_activations = activations[mask]
            relative_indices = self.get_activation_based_subset(unrated_activations, num_samples)
            additional_indices = [unrated_indices[index] for index in relative_indices]
        else:
            additional_indices = np.random.choice(unrated_indices, num_samples, replace=False)
        
        additional_samples = {index: dataset[index] for index in additional_indices}

        return additional_samples

    def repeat_ratings(
        self, 
        concept: str, 
        dataset: DictionaryDataset,
        all_ratings: Dict[str, int], 
        max_repetitions: int
    ) -> Dict[str, int]:
        """
        Repeat ratings for samples with non-zero scores to improve reliability.

        Args:
            concept (str): The concept to evaluate samples against.
            dataset (DictionaryDataset): The dataset of samples to rate.
            all_ratings (Dict[str, int]): Current ratings dictionary.
            max_repetitions (int): The maximum number of repetitions.
        
        Returns:
            Dict[str, int]: The updated ratings dictionary.
        """
        repetition = 0
        while repetition < max_repetitions:
            non_zero_ids = [id for id, rating in all_ratings.items() if rating > 0]
            if len(non_zero_ids) == 0:
                break
            else:
                repetition += 1
                repeat_set = {id: dataset[int(id)][1] for id in non_zero_ids}
                repeated_ratings, _ = self.generate_ratings(concept, repeat_set, self.batch_size)
                all_ratings.update(repeated_ratings)
        return all_ratings