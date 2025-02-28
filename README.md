# <img src="assets/images/fade_logo_lightmode.svg" alt="Logo" width="25" style="vertical-align: middle;"> FADE: Why Bad Descriptions Happen to Good Features


**FADE** is a framework for evaluating the alignment between LLM features and their descriptions across four key metrics: Clarity, Responsiveness, Purity, and Faithfulness.


### 🚧 This repository is still in development! More examples and documentation will follow soon! 🚧


## Getting Started
### Installation

Currently we only support installation from source:
```bash
git clone https://github.com/brunibrun/FADE
pip install ./FADE
```

### Example Usage

In the following example, we show how to use the `EvaluationPipeline` to evaluate the alignment between a feature and a feature description.

```python
from fade import EvaluationPipeline

# Initialize custom configuration with OpenAI LLM
custom_config = {
    'paths': {
        'output_path': '/Path/To/Output/Folder/'
    },
    'subjectLLM': {
        'sae_module': False,
        'batch_size': 4,
    },
    'evaluationLLM': {
        'type': 'openai',
        'name': 'gpt-4o-mini-2024-07-18',
        'api_key': 'YOUR-KEY-HERE',
    }
}

# Initialize eval pipeline
eval_pipeline = EvaluationPipeline(
    subject_model=model,  # e.g. huggingface model
    subject_tokenizer=tokenizer,  # e.g. huggingface tokenizer
    dataset=dataset,  # python dict with int keys and str values
    config=custom_config,
    device=device,  # e.g. torch.device
)

# Example neuron specification
neuron_module = 'named.module.of.the.feature'  # str of the module name
neuron_index = 42  # int of the neuron index
concept = "The feature description you want to evaluate."  # str of the feature description

# Run evaluation
(clarity, responsiveness, purity, faithfulness) = eval_pipeline.run(
    neuron_module=neuron_module,
    neuron_index=neuron_index,
    concept=concept
)
```

## Citation

```
@misc{puri2025fadebaddescriptionshappen,
    title={FADE: Why Bad Descriptions Happen to Good Features}, 
    author={Bruno Puri and Aakriti Jain and Elena Golimblevskaia and Patrick Kahardipraja and Thomas Wiegand and Wojciech Samek and Sebastian Lapuschkin},
    year={2025},
    eprint={2502.16994},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2502.16994}, 
}
```