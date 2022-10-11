from setuptools import find_packages, setup

REQUIRED = [
    "hydra-core",
    "pandas",
    "dill",
    "hydra-submitit-launcher",
    "sbibm"
]

setup(
    name="benchmark",
    python_requires=">=3.6.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)
