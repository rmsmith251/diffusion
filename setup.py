import os
from distutils.core import setup
from subprocess import getoutput

import setuptools

with open("README.md") as f:
    long_description = f.read()

extras_include = {
    "stable": [
        "ftfy",
        "scipy",
        "diffusers",
        "transformers",
        "huggingface-hub",
    ]
}

setup(
    name="diffusion",
    version="0.0.1",
    author="Ryan Smith",
    author_email="rsmith@plainsight.ai",
    url="https://github.com/rmsmith251/diffusion",
    packages=setuptools.find_packages(),
    package_data={"diffusion": ["py.typed"]},
    include_package_data=True,
    description="Playing around with diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "click==8.0.4",
        "fsspec[gs]",
        "torchvision",
        "numpy",
        "opencv-python",
        "pillow",
        "pydantic",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
