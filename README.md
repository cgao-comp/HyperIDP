# HyperIDP: Customizing Temporal Hypergraph Neural Networks for Multi-Scale Information Diffusion Prediction

## Overview

This is the code for our paper **HyperIDP: Customizing Temporal Hypergraph Neural Networks for Multi-Scale Information Diffusion Prediction**.
It is a differentiable architecture search framework based on temporal hypergraph for multi-scale information diffusion prediciton.

## Requirements:

To execute this project, it's essential to install the required dependencies. To do so, navigate to the directory containing the `requirements.txt` file and execute the following command:

```
pip install -r requirements.txt
```

## Instructions to run the experiment

To execute the project, run the following command in your terminal:

**Step 1.** Run the search process, given different random seeds.
(The _Twitter16_ dataset is used as an example)

```bash
python Libs/Exp/search.py
```

The results are saved in the directory `Output/Logs/`, e.g., `Search-christianity`.

**Step 2.** Fine tune the searched architectures. You need specify the arch_filename with the resulting filename from Step 1.

```bash
python Libs/Exp/finetune.py
```

Step 2 is a coarse-graind tuning process, and the results are saved in a picklefile in the directory `Finetune`.

### Setting

``` python
parser.add_argument('-dataset_name', default='christianity')
parser.add_argument('-epoch', default=50)
parser.add_argument('-batch_size', default=64)
parser.add_argument('-emb_dim', default=64)
parser.add_argument('-train_rate', default=0.8)
parser.add_argument('-valid_rate', default=0.1)
parser.add_argument('-lambda_loss', default=0.3)
parser.add_argument('-gamma_loss', default=0.05)
parser.add_argument('-max_seq_length', default=200)
parser.add_argument('-step_split', default=8)
parser.add_argument('-lr', default=0.001)
```

## Acknowledgement

The code is built on [SANE](https://github.com/LARS-research/SANE) and [MINDS](https://github.com/cspjiao/MINDS).
