from setuptools import setup, find_packages

setup(
    name="trieste_stopping",
    version="1.0",
    packages=find_packages(exclude=["experiments", "tutorials"]),
    python_requires=">=3.10",
    install_requires=[
        "trieste>=2.0.0",
        "tensorflow",
        "tensorflow-probability",
        "gpflow",
        "cmaes",
        "multipledispatch",
        "numpy",
    ],
    extras_require={
        "experiments": ["datasets",  "pandas", "scikit-learn", "wandb" "xgboost"],
        "tutorials": ["jupyter", "matplotlib", "seaborn"],
    }

)
