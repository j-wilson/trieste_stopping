# Overview
Stopping rules for Bayesian optimization within the [Trieste](https://github.com/secondmind-labs/trieste/tree/develop/trieste) framework, a Bayesian optimization package based on [GPflow](https://github.com/GPflow/GPflow/tree/develop/gpflow) and [TensorFlow](https://github.com/tensorflow/tensorflow). 

This package serves as companion code for [Stopping Bayesian Optimization with Probabilistic Regret Bounds](http://arxiv.org/abs/2402.16811). This paper introduces a `ProbabilisticRegreBoundRule` stopping rule which says: stop when a point has been found whose value is within $\epsilon>0$ of the best possible outcome with probability at least $1 - \delta$ under the model.


# Installation
This package can installed from the command line using
```
pip install git+https://github.com/j-wilson/trieste_stopping.git
```
Additional requirements for experiments and tutorials can instead be installed as
```
git clone https://github.com/j-wilson/trieste_stopping.git
cd trieste_stopping
pip install ".[experiments,tutorials]"
```


# Tutorials
Tutorials for parts of this package are included in the `tutorials` directory. These notebooks explain various methods and APIs:


| Notebook                                                                                                        | Content                                                   |
|-----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| [level_tests](https://github.com/j-wilson/trieste_stopping/blob/main/tutorials/level_tests.ipynb)               | Review of statistical tests                               |
| [factories](https://github.com/j-wilson/trieste_stopping/blob/main/tutorials/factories.ipynb)                   | Demo of helper methods used for experiment book keeping   |
| [knowledge_gradient](https://github.com/j-wilson/trieste_stopping/blob/main/tutorials/knowledge_gradient.ipynb) | Comparison of EI and (in-sample) KG acquisition functions | 
| [stopping_rules](https://github.com/j-wilson/trieste_stopping/blob/main/tutorials/stopping_rules.ipynb)         | Overview of stopping rules                                |


# Experiments
Experiments from the companion paper can be run using the `experiments/run_local_experiments.py` and `experiments/run_wandb_experiments.py` scripts. For example, you can run BO on Branin with the `ProbabilisticRegretBound` stopping rule by calling
```commandline
python -m experiments.run_local_experiment --objective Branin --step_limit 64 --seed 0
```
- Stopping rules can be chosen using the `--stopping_rule` argument and may be configured by passing a string-encoded dictionary via the `--stopping_kwargs` flag. 

- Objectives can be configured via the `--objective` and `--objective_kwargs` flags. 

- We recommend enabling [wandb](https://wandb.ai) to help track experiments. Some runs have been uploaded [here](https://wandb.ai/jtwilson/trieste_stopping/workspace?workspace=user-jtwilson) for demonstration purposes and a worked example is given below:
```commandline
python -m experiments.run_wandb_experiment \
--seed 0 \
--project trieste_stopping \
--step_limit 64 \
--objective Matern52Trajectory \
--objective_kwargs '{"dim": 2, "lengthscales": 0.33, "noise_variance": 1e-6}' \
--stopping_rule AcquisitionThreshold \
--stopping_kwargs '{"threshold": 1e-5}' 
```


# Citing Us
```
@article{wilson2024stopping,
    title={{Stopping Bayesian Optimization with Probabilistic Regret Bounds}}, 
    author={James T. Wilson},
    journal={Advances in Neural Information Processing Systems},
    year={2024},
    url={https://arxiv.org/abs/2402.16811},
}
```

