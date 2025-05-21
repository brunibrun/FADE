<br>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/images/fade_title_logo_darkmode.svg">
    <img src="assets/images/fade_title_logo_lightmode.svg" alt="FADE Logo" width="500">
  </picture>
</p>
<br>

[![arXiv](https://img.shields.io/badge/arXiv-2502.16994-b31b1b.svg)](https://arxiv.org/abs/2502.16994)


# <img src="assets/images/fade_logo_lightmode.svg" alt="Logo" width="25" style="vertical-align: middle;"> FADE: Why Bad Descriptions Happen to Good Features

<img src="assets/images/fade_logo_lightmode.svg" alt="Logo" width="20" style="vertical-align: middle;"> **FADE** helps you evaluate the alignment between LLM features and their natural language descriptions across four key metrics: Clarity, Responsiveness, Purity, and Faithfulness.

## 🔍 Features

- Model-agnostic evaluation of feature-to-description alignment
- Works with standard tramsformer neurons and SAE features
- Support for OpenAI, Azure, Ollama, vLLM, and other evaluation models


## Installation

```bash
pip install fade-language
```

## Tutorial
Check out our [**Tutorial Notebook**](examples/fade_tutorial.ipynb) that walks you through:

- A basic evaluation setup
- Using cached activations for improved performance
- Working with SAE features
- Using different evaluation models
- Advanced configuration options


## Quickstart

```python
from fade import EvaluationPipeline

# custom evaluation-model configuration with OpenAI LLM
config = {
    'evaluationLLM': {
        'type': 'openai', # type of evaluation model
        'name': 'gpt-4o-mini-2024-07-18', # the model variant
        'api_key': 'YOUR-KEY-HERE',
    }
}

# initialize evaluation pipeline
eval_pipeline = EvaluationPipeline(
    subject_model=model,  # e.g. huggingface model
    subject_tokenizer=tokenizer,  # e.g. huggingface tokenizer
    dataset=dataset,  # dict with int keys and str values
    config=config, # the custom config
    device=device,  # torch device
)

# example neuron specification
neuron_module = 'named.module.of.the.feature'  # str of the module name
neuron_index = 42  # int of the neuron index
concept = "The feature description you want to evaluate."  # str of the feature description

# run evaluation
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


<br>

🚧 This repository is still in active development! More examples and detailed documentation will follow soon! 🚧
