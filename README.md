# MaLA-500: Massive Language Adaptation of Large Language Models

[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/collections/MaLA-LM/mala-500-660e57f8e53e3cc2ccd31cb9)
[![Data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Data-green)](https://huggingface.co/datasets/cis-lmu/Glot500)
[![arXiv](https://img.shields.io/badge/arXiv-2305.12182-b31b1b.svg)](https://arxiv.org/abs/2401.13303)

MaLA-500 is a novel large language model designed to cover an extensive range of 534 languages. This model builds upon LLaMA 2 7B and integrates continued pretraining with vocabulary extension, with an expanded vocabulary size of 260,164, and LoRA low-rank adaptation.


- **Continued Pretraining:** Enhances the model's ability to adapt to a wide range of languages.
- **LoRA Low-Rank Adaptation:** LoRA low-rank adaptation refines the model's adaptation capabilities.
- **Vocabulary Extension:** MaLA-500 boasts an extended vocabulary size of 260,164.
- **Multilingual Proficiency:** Trained on Glot500-c, covering 534 languages.

Please refer to [our paper](https://arxiv.org/pdf/2401.13303.pdf) for more details.

## How to Get Started with the Model

Requirements:
```
transformers>=4.36.1
peft>=0.6.2
```

Use the code below to get started with the model.

``` python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
base_model.resize_token_embeddings(260164)
tokenizer = AutoTokenizer.from_pretrained('MaLA-LM/mala-500-10b-v2')
model = PeftModel.from_pretrained(base_model, 'MaLA-LM/mala-500-10b-v2')
```

## Codebase

### Vocabulary Extension

Code under the directory of `./tokenization`. 

### Continued Pretraining

Code under the directory of `./continued_pretraining`. 

Customize the `run.sh` script for your own clusters or workstation. The script is provided for SLURM-based systems.
You might want to use DeepSpeed. See config examples under `./continued_pretraining/config`. 

### Evaluation

Code under the directory of `./evaluation`.

## Citation

```
@misc{lin2024mala500,
      title={MaLA-500: Massive Language Adaptation of Large Language Models}, 
      author={Peiqin Lin and Shaoxiong Ji and Jörg Tiedemann and André F. T. Martins and Hinrich Schütze},
      year={2024},
      eprint={2401.13303},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
