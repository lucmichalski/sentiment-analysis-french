#!/usr/bin/env python
from pathlib import Path
from setuptools import setup, find_packages

version_file = Path(__file__).parent.joinpath("adaptnlp", "VERSION.txt")
version = version_file.read_text(encoding="UTF-8").strip()

with open("requirements.txt") as reqs_file:
    install_requires = reqs_file.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="adaptnlp",
    version=version,
    author="Andrew Chang",
    author_email="achang@novetta.com",
    packages=find_packages(),
    keywords=[
        "NLP",
        "flair",
        "Natural Language Processing",
        "Machine Learning",
        "ML",
        "torch",
        "pytorch",
        "NER",
    ],
    install_requires=install_requires,
    license="Apache 2.0",
    description="AdaptNLP: A Natural Language Processing Library and Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=True,
)
