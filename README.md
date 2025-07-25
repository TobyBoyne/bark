# BARK

This repository contains the code for the experiments for the paper *BARK: A 
Fully Bayesian Tree Kernel for Black-box Optimization*.

## Installation

To set up a Python environment, we recommend the simplest approach below 
(for Windows, different platforms may vary). After cloning this repository:
```bash
python -m venv barkvenv
barkvenv\Scripts\activate
python -m pip install .
```
You can also use `uv`, `poetry`, or `conda` if you'd prefer.

Since we use Gurobi to optimize the acquisition function, you will need a 
license (you can get a free [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/), if applicable).

## Running experiments

To run the experiments, you will use the configuration files in `configs/`,
and the scripts in `examples/`. For example, to reproduce the Bayesian 
optimization for the `BARK` model on the `TreeFunction` benchmark:

```bash
python examples/bayes_opt/bark_study_strategy.py
    -s 42 # random seed for initial data
    -c configs/benchmark_configs/treefunction_config.yaml
    -m configs/model_configs/bark_config.yaml
    -o results/
```
