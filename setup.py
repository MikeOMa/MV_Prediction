#!/usr/bin/env python
import setuptools

packages = [
    "probdrift",
    "probdrift.models",
    "probdrift.summary",
    "probdrift.oceanfns",
    "probdrift.plot"
]
setuptools.setup(
    name="probdrift",
    version="0.1",
    description="python code for distribution drifter prediction",
    author="Michael O'Malley",
    author_email="michael.omalley1011@gmail.com",
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
