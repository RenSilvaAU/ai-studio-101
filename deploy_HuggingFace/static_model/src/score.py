import os
import logging
import json
from sentence_transformers import SentenceTransformer

def init():

    global model
    # Load the model from the path specified by the environment variable
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model"
    )
    model = SentenceTransformer.load(model_path)
    logging.info("Init complete")

def run(data):
    try:
        # Parse the input data
        inputs = json.loads(data)
        sentences = inputs['sentences']
        
        # Perform inference
        embeddings = model.encode(sentences)
        
        # Return the embeddings as a JSON response
        return json.dumps({"embeddings": embeddings.tolist()})
    except Exception as e:
        # Return any errors that occur during inference
        return json.dumps({"error": str(e)})
