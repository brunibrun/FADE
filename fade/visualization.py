import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import textwrap
import numpy as np
import pandas as pd

sns.set_theme()


class PlotResults:

    @staticmethod
    def plot_polygonal_radar(
        data: Dict[str, Tuple], 
        colors: Optional[List[str]] = None, 
        save_path: Optional[str] = None, 
        figsize: Tuple[int, int] = (6, 6), 
        fontsize: int = 10
    ) -> None:
        """
        Plots a polygonal radar chart without circular gridlines.

        Args:
            data: Dictionary mapping labels to metric tuples, where each tuple contains 
                  values for metrics in order: responsiveness, faithfulness, clarity, purity.
            colors: List of colors for each entry in the data dictionary.
            save_path: Path to save the plot. If None, the plot is displayed.
            figsize: Size of the figure as (width, height).
            fontsize: Font size for labels.
        """
        sns.set_theme(style='whitegrid')
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        # set polygonal gridlines
        angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()
        angles += angles[:1]
        grid_levels = [ 0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
        for grid_value in grid_levels:
            grid_points = [grid_value] * 4
            grid_points += grid_points[:1]  # Close the polygon
            ax.plot(angles, grid_points, color='gray', linewidth=0.7, linestyle='solid')
            ax.fill(angles, grid_points, color='gray', alpha=0.01)  # Optional grid fill
        
        # Plot each neuron
        for idx, (key, values) in enumerate(data.items()):
            values += values[:1]  # Close the loop
            color = colors[idx] if colors and idx < len(colors) else 'blue'
            ax.fill(angles, values, color=color, alpha=0.5, label=key)
            ax.plot(angles, values, color=color, linewidth=2, alpha=0.9)

        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
        ax.text(0.5, 1.01, 'F', transform=ax.transAxes, fontsize=fontsize, va='bottom', ha='center')
        ax.text(1.01, 0.5, 'R', transform=ax.transAxes, fontsize=fontsize, va='center', ha='left')
        ax.text(0.5, -0.01, 'P', transform=ax.transAxes, fontsize=fontsize, va='top', ha='center')
        ax.text(-0.01, 0.5, 'C', transform=ax.transAxes, fontsize=fontsize, va='center', ha='right')

        # Remove circular gridlines
        ax.yaxis.grid(False)
        # Remove the circular outline
        ax.spines['polar'].set_visible(False)
        # remove numbers from the axes
        ax.set_yticklabels([])
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01))

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


    @staticmethod
    def activation_density_comparison_plot(
        non_concept_activations: np.ndarray, 
        concept_activations: np.ndarray, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Creates a density plot comparing activations for concept vs non-concept.
        
        Args:
            non_concept_activations: Array of activation values for non-concept samples.
            concept_activations: Array of activation values for concept samples.
            save_path: Path to save the plot. If None, the plot is displayed.
        """
        plt.figure(figsize=(10, 6))

        non_concept = non_concept_activations
        concept = concept_activations
        
        sns.histplot(non_concept, bins=100, alpha=0.6, label="non-concept", color="#482677ff", stat="density", edgecolor='b')
        sns.histplot(concept, bins=100,alpha=0.8, label="concept", color="#fde725ff", stat="density",  edgecolor='b')

        plt.xlabel("Activation", fontsize=10)
        plt.ylabel("Density", fontsize=10)
        plt.gca().set_yticklabels([])
        plt.legend()
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


    @staticmethod
    def stacked_density_plot(
        aggregated_ratings: Dict[int, np.ndarray], 
        save_path: Optional[str] = None
    ) -> None:
        """
        Creates a stacked density plot of activations for each class.
        
        Args:
            aggregated_ratings: Dictionary with class labels as integer keys and 
                               arrays of activations as values.
            save_path: Path to save the plot. If None, the plot is displayed.
        """
        # convert to dataframe
        dataframe = []
        for class_name, values in aggregated_ratings.items():
            for value in values:
                dataframe.append({"Activation": value.item(), "Class": class_name})
        dataframe = sorted(dataframe, key=lambda x: x["Class"])
        dataframe = pd.DataFrame(dataframe)

        # normalize activations
        dataframe["Activation"] = (dataframe["Activation"]-dataframe["Activation"].min())/(dataframe["Activation"].max()-dataframe["Activation"].min())

        # Bin the data
        dataframe["Bin"], binned_centers = pd.cut(dataframe["Activation"], bins=50, labels=False, retbins=True)
        dataframe["Bin"] = dataframe["Bin"].apply(lambda x: binned_centers[x])

        # Create contingency table
        binned_data = dataframe.groupby(["Bin", "Class"]).size().unstack(fill_value=0)

        colors = {0: "#482677ff", 1: "#1f968b", 2: "#fde725ff"}
        colors = [colors[cls] for cls in dataframe["Class"].unique()]

        # Set up the figure and axes
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
        fig.subplots_adjust(hspace=0.1)

        ax1.stackplot(binned_data.index, binned_data.values.T, colors=colors, alpha=1,edgecolor='w', linewidth=0.2)

        ax1.legend(title="Classes", labels=dataframe["Class"].unique())
        ax1.set_ylabel("Density", fontsize=10)
        ax1.set_yticklabels([])

        # Add the conditional density plot (normalized by class count)
        conditional_proportions = binned_data.div(binned_data.sum(axis=1), axis=0)
        ax2.stackplot(conditional_proportions.index, conditional_proportions.values.T, colors=colors, alpha=1,edgecolor='w', linewidth=0.2)

        ax2.set_ylabel("Cond. Density", fontsize=10)
        ax2.set_xlabel("Relative Activation", fontsize=10)
        ax2.tick_params(axis='x', labelsize=8)
        ax2.set_yticklabels([])
        ax1.set_xlim(-0.01, 1)

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


    @staticmethod
    def barplot_visualization(
        ratings: pd.DataFrame, 
        save_path: Optional[str] = None
    ) -> None:
        """
        Creates a bar plot showing differences in ratings relative to a baseline.
        
        Args:
            ratings: DataFrame containing ratings data with modification factors.
            save_path: Path to save the plot. If None, the plot is displayed.
        """
        # Exclude modification factor 1 and calculate differences relative to it
        base = ratings[0]  # Base case with modification_factor = 0
        differences = ratings.subtract(base, axis=0).drop(columns=[0])

        # Transform the differences data into long format
        differences_long = differences.reset_index().melt(
            id_vars='index', var_name='Species_', value_name='Difference'
        )
        differences_long.rename(columns={'index': 'Class'}, inplace=True)

        # Define colors for the classes
        colors = {0: "#482677ff", 1: "#1f968b", 2: "#fde725ff"}
        class_colors = {cls: colors[cls] for cls in differences_long["Class"].unique()}

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=differences_long, 
            x='Species_', 
            y='Difference', 
            hue='Class', 
            dodge=True, 
            palette=class_colors
        )
    
        # Customize axes and legend
        plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add baseline for zero difference

        plt.xlabel('Neuron Modification Factor', fontsize=10)
        plt.ylabel('Difference in Frequency', fontsize=10)
        plt.xticks(rotation=0, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(title='Class', fontsize=10, title_fontsize='10', loc='upper right', frameon=True)

        # Adjust layout
        plt.tight_layout()

        # Save or show the plot
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


    @staticmethod
    def combine_plots(
        neuron: str, 
        concept: str, 
        path: str
    ) -> None:
        """
        Combines multiple plots into a single figure for comprehensive visualization.
        
        Args:
            neuron: Identifier for the neuron being analyzed.
            concept: Description of the concept being visualized.
            path: Directory path containing the plots to combine.
        """
        plot_files = ["clarity_plot.png", "responsiveness_purity_plot.png", "faithfulness_plot.png", "diamond_plot.png"]
        plot_files = [f for f in os.listdir(os.path.join(path, "plots")) if f in plot_files]
        images = {f: mpimg.imread(os.path.join(path, "plots", f)) for f in plot_files}
        
        fig, axs = plt.subplots(2, 2, figsize=(20, 12))
        
        if "diamond_plot.png" in images:
            axs[0, 0].imshow(images["diamond_plot.png"])
            axs[0, 0].set_title("Metrics", fontsize=10, loc='center', weight='bold')
        axs[0, 0].axis('off')

        if "responsiveness_purity_plot.png" in images:
            axs[0, 1].imshow(images["responsiveness_purity_plot.png"])
            axs[0, 1].set_title("Responsiveness & Purity", fontsize=10, loc='center', weight='bold')
        axs[0, 1].axis('off')

        if "clarity_plot.png" in images:
            axs[1, 0].imshow(images["clarity_plot.png"])
            axs[1, 0].set_title("Clarity", fontsize=10, loc='center', weight='bold')
        axs[1, 0].axis('off')

        if "faithfulness_plot.png" in images:
            axs[1, 1].imshow(images["faithfulness_plot.png"])
            axs[1, 1].set_title("Faithfulness", fontsize=10, loc='center', weight='bold')
        axs[1, 1].axis('off')

        max_line_length = 120
        concept_wrapped = "\n".join(textwrap.wrap(concept, max_line_length))
        fig.text(0.5, 0.98, f"Neuron: {neuron}\nConcept: '{concept_wrapped}'" , ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(path, "plots", "combined_plot.png"), bbox_inches='tight')
        plt.close()