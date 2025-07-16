import os
import torch
import json
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, List, Tuple, Optional

from fade.data import DictionaryDataset
from fade.visualization import PlotResults


def gini_ap_metric(non_concept_activations: torch.Tensor, concept_activations: torch.Tensor) -> Tuple[float, float]:
    """
    Calculate Gini coefficient and Average Precision (AP) for activation distributions.

    Args:
        non_concept_activations (torch.Tensor): Activations for non-concept samples.
        concept_activations (torch.Tensor): Activations for concept samples.

    Returns:
        Tuple[float, float]: A tuple containing (Gini coefficient, Average Precision).
    """
    all_activations = torch.cat([non_concept_activations, concept_activations])
    labels = torch.cat([torch.zeros_like(non_concept_activations), torch.ones_like(concept_activations)])
    gini = 2 * roc_auc_score(labels.numpy(), all_activations.numpy()) - 1
    if gini < 0: # ensure AP is calculated for correct parts of activations
        all_activations = -all_activations
    ap = average_precision_score(labels.numpy(), all_activations.numpy()) 
    return float(abs(gini)), float(ap)


def faithfulness_metric(ratings_dict) -> Tuple[float, int, pd.DataFrame]:
    """
    Calculate faithfulness metric based on the ratings of the generated sequences.

    Args:
        ratings_dict (Dict[float, Dict[str, int]]): Ratings for generated sequences.

    Returns:
        float: Faithfulness metric.
        int: maximal modification factor.
        pd.DataFrame: Value counts of ratings for each modification factor.
    """
    ratings = pd.DataFrame(ratings_dict)
    value_counts = ratings.apply(pd.Series.value_counts)
    value_counts = value_counts.fillna(0).astype(int)
    value_counts = value_counts.div(value_counts.sum(axis=0), axis=1)
    if 2 in value_counts.T:
        concept_vector = value_counts.T[2]
        faithfulness = float(max(concept_vector.max() - concept_vector[0], 0) / (1-concept_vector[0]))
        max_modification_factor = concept_vector.index.max() if faithfulness > 0 else None
    else:
        faithfulness = 0
        max_modification_factor = None

    return faithfulness, max_modification_factor, value_counts

class GenerateResults:
    def __init__(self, output_path: str, store_data: bool = True):
        """
        Initialize results generation with output configuration.

        Args:
            output_path (str): Directory path to store generated results.
            store_data (bool, optional): Whether to store data files. Defaults to True.
        """        
        self.output_path = output_path
        self.store_data = store_data

    def activation_centric_results(
        self, 
        natural_dataset: DictionaryDataset,
        synthetic_dataset: DictionaryDataset, 
        natural_activations: torch.Tensor, 
        synthetic_activations: torch.Tensor, 
        natural_ratings: Dict[str, int]
    ) -> Tuple[float, float, float]:
        """
        Compute activation-centric metrics and optionally store results.

        Args:
            natural_dataset (DictionaryDataset): Dataset of natural samples. 
            synthetic_dataset (DictionaryDataset): Dataset of synthetic samples.
            natural_activations (torch.Tensor): Activations for natural samples.
            synthetic_activations (torch.Tensor): Activations for synthetic samples.
            natural_ratings (Dict[str, int]): Ratings for natural samples.

        Returns:
            Tuple[float, float, float]: Clarity, Responsiveness, and Purity metrics.
        """
        clarity, _ = gini_ap_metric(natural_activations, synthetic_activations)
        
        aggregated_ratings = {key: [] for key in natural_ratings.values()}
        for key, value in natural_ratings.items():
            aggregated_ratings[value].append(natural_activations[int(key)])
        aggregated_ratings = {key: torch.tensor(value) for key, value in aggregated_ratings.items()}
        if 2 in aggregated_ratings and 0 in aggregated_ratings:
            responsiveness, purity = gini_ap_metric(aggregated_ratings[0], aggregated_ratings[2])
        
        if self.store_data:
            # data
            natural_rating_full = {natural_dataset[int(index)][1]: rating for index, rating in natural_ratings.items()}
            with open(os.path.join(self.output_path, "data/activation_centric_ratings.json"), "w") as file:
                json.dump(natural_rating_full, file, indent=2)

            synthetic_dataset = [sample[1] for sample in synthetic_dataset]
            with open(os.path.join(self.output_path, "data/activation_centric_synthetic_data.json"), "w") as file:
                json.dump(synthetic_dataset, file, indent=2)

            activation_centric_metadata = {
                "Clarity": clarity,
                "Responsiveness": responsiveness,
                "Purity": purity,
                "Natural Samples": len(natural_activations),
                "Synthetic Samples": len(synthetic_activations),
                "Natural Concept Samples": len(aggregated_ratings[2]) if 2 in aggregated_ratings else 0,
                "Natural Non-Concept Samples": len(aggregated_ratings[0]) if 0 in aggregated_ratings else 0
            }
            with open(os.path.join(self.output_path, "data/activation_centric_metrics.json"), "w") as file:
                json.dump(activation_centric_metadata, file, indent=2)

            # plots
            PlotResults.activation_density_comparison_plot(natural_activations, synthetic_activations, save_path=os.path.join(self.output_path, "plots/clarity_plot.png"))
            PlotResults.stacked_density_plot(aggregated_ratings, save_path=os.path.join(self.output_path, "plots/responsiveness_purity_plot.png"))

        return clarity, responsiveness, purity


    def output_centric_results(
        self, 
        generated_sequences: Dict[float, Dict[int, str]], 
        ratings_dict: Dict[float, Dict[str, int]]
    ) -> float:
        """
        Compute output-centric metrics, particularly faithfulness.

        Args:
            generated_sequences (Dict[float, Dict[int, str]]): Sequences for different modification factors.
            ratings_dict (Dict[float, Dict[str, int]]): Ratings for generated sequences.

        Returns:
            float: Faithfulness metric.
        """
        faithfulness, max_modification_factor, value_counts = faithfulness_metric(ratings_dict)

        if self.store_data:
            # data
            combined_dict = {modification_factor: {id: (generated_sequences[modification_factor][id], ratings_dict[modification_factor][str(id)]) for id in generated_sequences[modification_factor].keys()} for modification_factor in ratings_dict.keys()}
            with open(os.path.join(self.output_path, "data/output_centric_ratings.json"), "w") as file:
                json.dump(combined_dict, file, indent=2)

            output_centric_metadata = {
                "Faithfulness": faithfulness,
                "Number of Modification Factors": len(ratings_dict),
                "Number of Sequences per Modification Factor": len(generated_sequences),
                "Value Counts": value_counts.to_dict()
            }
            with open(os.path.join(self.output_path, "data/output_centric_metrics.json"), "w") as file:
                json.dump(output_centric_metadata, file, indent=2)
            
            # plots
            PlotResults.barplot_visualization(value_counts, save_path=os.path.join(self.output_path, "plots/faithfulness_plot.png"))

        return faithfulness

    def combine_results(
        self, 
        clarity: float, 
        responsiveness: float, 
        purity: float, 
        faithfulness: float, 
        neuron_index: int, 
        concept: str
    ) -> Tuple[float, float, float, float]:
        if self.store_data:
            PlotResults.plot_polygonal_radar(
                {f"Neuron {neuron_index}": (responsiveness, faithfulness, clarity, purity)},
                save_path=os.path.join(self.output_path, "plots/diamond_plot.png"))
            PlotResults.combine_plots(neuron_index, concept, self.output_path)

        return clarity, responsiveness, purity, faithfulness