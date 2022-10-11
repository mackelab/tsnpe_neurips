from setuptools import find_packages, setup

REQUIRED = [
    "matplotlib",
]

setup(
    name="paper",
    python_requires=">=3.6.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)
