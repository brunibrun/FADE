<br>
<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/images/fade_title_logo_darkmode.svg">
    <img src="assets/images/fade_title_logo_lightmode.svg" alt="FADE Logo" width="400">
  </picture>
</p>
<br>



[![ACL Findings](https://img.shields.io/badge/ACL-Findings-green?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgo8c3ZnCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgdmVyc2lvbj0iMS4wIgogICB3aWR0aD0iNjgiCiAgIGhlaWdodD0iNjgiCiAgIGlkPSJzdmcyIj4KICA8ZGVmcwogICAgIGlkPSJkZWZzNCIgLz4KICA8cGF0aAogICAgIGQ9Ik0gNDEuOTc3NTUzLC0yLjg0MjE3MDllLTAxNCBDIDQxLjk3NzU1MywxLjc2MTc4IDQxLjk3NzU1MywxLjQ0MjExIDQxLjk3NzU1MywzLjAxNTggTCA3LjQ4NjkwNTQsMy4wMTU4IEwgMCwzLjAxNTggTCAwLDEwLjUwMDc5IEwgMCwzOC40Nzg2NyBMIDAsNDYgTCA3LjQ4NjkwNTQsNDYgTCA0OS41MDA4MDIsNDYgTCA1Ni45ODc3MDgsNDYgTCA2OCw0NiBMIDY4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDEwLjUwMDc5IEwgNTYuOTg3NzA4LDMuMDE1OCBDIDU2Ljk4NzcwOCwxLjQ0MjExIDU2Ljk4NzcwOCwxLjc2MTc4IDU2Ljk4NzcwOCwtMi44NDIxNzA5ZS0wMTQgTCA0MS45Nzc1NTMsLTIuODQyMTcwOWUtMDE0IHogTSAxNS4wMTAxNTUsMTcuOTg1NzggTCA0MS45Nzc1NTMsMTcuOTg1NzggTCA0MS45Nzc1NTMsMzAuOTkzNjggTCAxNS4wMTAxNTUsMzAuOTkzNjggTCAxNS4wMTAxNTUsMTcuOTg1NzggeiAiCiAgICAgc3R5bGU9ImZpbGw6I2VkMWMyNDtmaWxsLW9wYWNpdHk6MTtmaWxsLXJ1bGU6ZXZlbm9kZDtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MTIuODk1NDExNDk7c3Ryb2tlLWxpbmVjYXA6YnV0dDtzdHJva2UtbGluZWpvaW46bWl0ZXI7c3Ryb2tlLW1pdGVybGltaXQ6NDtzdHJva2UtZGFzaGFycmF5Om5vbmU7c3Ryb2tlLWRhc2hvZmZzZXQ6MDtzdHJva2Utb3BhY2l0eToxIgogICAgIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsIDExKSIKICAgICBpZD0icmVjdDIxNzgiIC8+Cjwvc3ZnPgo=)](https://aclanthology.org/2025.findings-acl.881/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.16994-b31b1b.svg)](https://arxiv.org/abs/2502.16994)
[![PyPI version](https://img.shields.io/pypi/v/fade-language.svg)](https://pypi.org/project/fade-language/)


# <img src="assets/images/fade_logo_lightmode.svg" alt="Logo" width="25" style="vertical-align: middle;"> FADE: Why Bad Descriptions Happen to Good Features

**FADE** helps you evaluate the alignment between LLM features and their natural language descriptions across four key metrics: Clarity, Responsiveness, Purity, and Faithfulness.

## 🔍 Features

- Model-agnostic evaluation of feature-to-description alignment
- Works with standard transformer neurons and SAE features
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

If you use FADE in your research, please cite:

```
@inproceedings{DBLP:conf/acl/PuriJGKWSL25,
  author       = {Bruno Puri and
                  Aakriti Jain and
                  Elena Golimblevskaia and
                  Patrick Kahardipraja and
                  Thomas Wiegand and
                  Wojciech Samek and
                  Sebastian Lapuschkin},
  editor       = {Wanxiang Che and
                  Joyce Nabende and
                  Ekaterina Shutova and
                  Mohammad Taher Pilehvar},
  title        = {{FADE:} Why Bad Descriptions Happen to Good Features},
  booktitle    = {Findings of the Association for Computational Linguistics, {ACL} 2025,
                  Vienna, Austria, July 27 - August 1, 2025},
  series       = {Findings of {ACL}},
  volume       = {{ACL} 2025},
  pages        = {17138--17160},
  publisher    = {Association for Computational Linguistics},
  year         = {2025},
  url          = {https://aclanthology.org/2025.findings-acl.881/},
  timestamp    = {Tue, 27 Jan 2026 20:27:02 +0100},
}
```
<br>