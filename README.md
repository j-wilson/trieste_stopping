# Overview
Stopping rules for Bayesian optimization within the [Trieste](https://github.com/secondmind-labs/trieste/tree/develop/trieste) framework, a Bayesian optimization package based on [GPflow](https://github.com/GPflow/GPflow/tree/develop/gpflow) and [TensorFlow](https://github.com/tensorflow/tensorflow). 

This package serves as companion code for [Stopping Bayesian Optimization with Probabilistic Regret Bounds](???). This paper introduces a `ProbabilisticRegreBoundRule` stopping rule which says: stop when a point has been found whose value is within $\delta>0$ of the best possible outcome with probability at least $1 - \epsilon$ under the model.


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
Tutorials for parts of this package are included in the `tutorials` directory. These notebooks explain various methods ad APIs:


| Notebook           | Content                                                      |
|--------------------|--------------------------------------------------------------|
| [adaptive_estimator](https://github.com/j-wilson/trieste_stopping/blob/icml2024/tutorials/adaptive_estimator.ipynb) | Review of adaptive empirical Bernstein estimator algorithm |
| [factories](https://github.com/j-wilson/trieste_stopping/blob/icml2024/tutorials/factories.ipynb)          | Demo of helper methods used for experiment book keeping    |
| [knowledge_gradient](https://github.com/j-wilson/trieste_stopping/blob/icml2024/tutorials/knowledge_gradient.ipynb) | Comparison of EI and (in-sample) KG acquisition functions  | 
| [stopping_rules](https://github.com/j-wilson/trieste_stopping/blob/icml2024/tutorials/stopping_rules.ipynb)     | Overview of stopping rules                                 |


# Experiments
Experiments from the companion paper can be run using the `experiments/run_experiments.py` script. 

As a starter example, you can run BO on Branin using the `ProbabilisticRegretBound` as described in the paper by calling
```commandline
python -m experiments.run_experiment --problem Branin --step_limit 64 --seed 0
```
- Stopping rules can be chosen using the `--stopping_rule` argument and may be configured by passing a string-encoded dictionary via the `--stopping_kwargs` flag. 

- Problems can be selected via the `--problem` flag. If you wish to minimize a draw from a GP prior, you should instead pass `--problem Matern52Objective` along with a dictionary of string-encoded arguments for `--problem_kwargs` (see below).

- We recommend enabling [wandb](https://wandb.ai) for quality of life purposes. Assuming W&B is already setup on your machine, this functionality can be enabled by passing a project name via the `--wandb` flag. Some examples are available [here](???) for demonstration purposes.

 - A worked example is given below:
```commandline
python -m experiments.run_experiment \
--seed 0 \
--wandb trieste_stopping \
--step_limit 64 \
--problem Matern52Objective \
--problem_kwargs '{"dim": 2, "lengthscales": 0.33, "noise_variance": 1e-6}' \
--stopping_rule AcquisitionThreshold \
--stopping_kwargs '{"threshold": 1e-5}' 
```


# Citing Us
```
@misc{wilson2024stopping,
    title={{Stopping Bayesian Optimization with Probabilistic Regret Bounds}}, 
    author={James T. Wilson},
    year={2024},
    eprint={???},
    url={???}
}
```

