# ai-studio-101
This repository contains some notebooks to demonstrate certain tasks in Azure AI Studio

## Pre-requisits
This repository is intended to be used inside a Compute Instance in Azure AI Studio


## Tasks

### Generic tasks in AML / AI Studio

Examples of tasks using notebooks in Azure AI Studio / AML:

- Using a Hugging Face model in foreground in a Compute Instance or Local computer: [./generic-notebooks/using-hf-locally.ipynb](./generic-notebooks/using-hf-locally.ipynb)

- Using Open AI locally in a Compute Instance or Local computer: [./generic-notebooks/using-openai.ipynb](./generic-notebooks/using-openai.ipynb)

- using MLFlow to track experiments from a local computer or compute instance [./generic-notebooks/using-mlflow.ipynb)](./generic-notebooks/using-mlflow.ipynb)

### Deploying Hugging Face models as a Managed Endpoing

This is to be done when a model is not available inside *Azure Model Catalog*

Follow this link: [./deploy_model/1.deploy.ipynb](./deploy_model/1.deploy.ipynb)

An alternative method to deploy the same model can be found here: [./deploy_model_v2/custom_model_deployment.ipynb](./deploy_model_v2/custom_model_deployment.ipynb)

### Deploy a gated Hugging Face model as a managed endpoing

This example shows the deployment of a gated meta-llama model (from Hugging Face).

Follow this link: [./deploy_llama/1.deploy.ipynb](./deploy_llama/1.deploy.ipynb)