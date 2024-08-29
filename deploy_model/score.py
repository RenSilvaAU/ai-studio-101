import os
import logging
import json
from sentence_transformers import SentenceTransformer

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    sentences = json.loads(raw_data)["sentences"]
    embeddings = model.encode(sentences)
    logging.info("Request processed")
    return embeddings.tolist()
