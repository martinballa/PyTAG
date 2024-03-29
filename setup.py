from setuptools import setup, find_packages

setup(
    name="pytag",
    version="0.1",
    packages=find_packages(),
    package_data={'': ['jars/*.jar']},
    include_package_data=True,
    install_requires=["gymnasium", "Jpype1", "numpy", "gdown"],
    extras_require={
        "examples": ["torch", "tensorboard", "wandb"]
    }
)