Repository for Serving a NLP Model using Ray Serve

## Setup Guide

Install requirements

```
pip install -r requirements.txt
```

Setup Ray Cluster

```
ray start --head
```

Deploy API Endpoint

```
python src/app_composite.py
```

## Environment

The project contains a setup.py file and a requirement file, you can use pip
to install the dependencies with either of teh files, but the setup.py is the
preferred choice. The project has been developed with python 3.8, and we would
suggest you to do the same, although the project can work with python >=3.6

We would also suggest you to create a separate environment for this project,
you could use conda or venv for this, there is no difference.

## Tensorflow model

This project expects you to deploy a machine learning model trained with
Tensorflow 2. The model is provided as h5 and accepts as input an array of
arrays of floats of any size. For example:

[
[0.2,4.3,0.5]
]

Is an input containing one array for inferencem where the array has dimension 3.

## Huggingface transformers

The trabsformers library is included within the requirements.txt or setup/py files

## NLTK

Nltk installation is provided and without version, as any version is OK. You might need, however, to install external data
for the sentence tokenization. If you need to do so, the logs will help you through the process
