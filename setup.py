#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

VERSION_FILE = os.path.join(os.path.dirname(__file__), "VERSION.txt")

setup(
    name="mb_nlp",
    description="NLP text classification with BERT",
    author=["Malav Bateriwala"],
    packages=find_packages(),
    scripts=[],
    install_requires=[
        "transformers>=4.38.0",
        "torch>=2.0.0",
        "pandas",
        "scikit-learn"
    ],
    setup_requires=["setuptools-git-versioning<2"],
    python_requires='>=3.8',
    setuptools_git_versioning={
        "enabled": True,
        "version_file": VERSION_FILE,
        "count_commits_from_version_file": True,
        "template": "{tag}",
        "dev_template": "{tag}.dev{ccount}+{branch}",
        "dirty_template": "{tag}.post{ccount}",
    },
)
