# AI Studio / AML samples
This repository contains some notebooks to demonstrate some common tasks in Azure AI Studio and Azure Machine Learning Worskspace.

## Pre-requisits
This repository is intended to be used inside a Compute Instance in Azure AI Studio or Azure Machine Learning.

- For AI Studio, go to https://ai.azure.com 

- For AML, got to https://ml.azure.com

## Tasks

### Generic tasks in AML / AI Studio

Examples of tasks using notebooks in Azure AI Studio / AML:

- Using a Hugging Face model in foreground in a Compute Instance or Local computer: [./generic-notebooks/using-hf-locally.ipynb](./generic-notebooks/using-hf-locally.ipynb)

- Using Open AI locally in a Compute Instance or Local computer: [./generic-notebooks/using-openai.ipynb](./generic-notebooks/using-openai.ipynb)

- using MLFlow to track experiments from a local computer or compute instance [./generic-notebooks/using-mlflow.ipynb)](./generic-notebooks/using-mlflow.ipynb)

### Deploying Public models in Azure Machine Learning / AI Studio 

This is to be done when:

- a model is not available inside *Azure Model Catalog*

- a model is gated and you need a Hugging Face token to access it

> Note: This process is similar to the one that Model Catalog follows to deploy models as managed endpoint

The model is downloaded from its public location and made available as an **Azure Managed Endpoint**

There are two examples:

- A Gated Llama Meta Guard B2

    The repo contains two methods:

    - Downloading the model dynamically, when endpoints starts: [deploy_llama/dynamic_model_load/1.deploy.ipynb](./deploy_llama/dynamic_model_load/1.deploy.ipynb)
    - Registering the model in Azure Machine Learning, then using it in the endpoint: [deploy_llama/static_model_load/1.deploy.ipynb](./deploy_llama/static_model_load/1.deploy.ipynb)

- A Hugging Face model (not available in model catalog):

    The repo contains two methods:

    - Downloading the model dynamically, when endpoints starts: [deploy_HuggingFace/dynamic_model/1.deploy.ipynb](./deploy_HuggingFace/dynamic_model/1.deploy.ipynb)
    - Registering the model in Azure Machine Learning, then using it in the endpoint: [deploy_HuggingFace/static_model/1.deploy.ipynb](./deploy_HuggingFace/static_model/1.deploy.ipynb)
