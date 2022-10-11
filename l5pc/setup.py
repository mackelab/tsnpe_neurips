from setuptools import find_packages, setup

REQUIRED = [
    "neuron",
    "hydra-core",
    "datajoint",
    "pandas",
    "dill",
    "ruamel.yaml",
    "bluepyopt",
    "hydra-submitit-launcher",
]

setup(
    name="l5pc",
    python_requires=">=3.6.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)
