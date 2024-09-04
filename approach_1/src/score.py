import os
import logging
import json
import joblib

def init():
    global model
    # Load the model from the path specified by the environment variable
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model/all-mpnet-base-v2"
    )
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)
    logging.info("Init complete")

def run(data):
    try:
        # Parse the input data
        inputs = json.loads(data)
        sentences = inputs['data']
        
        # Perform inference
        embeddings = model.encode(sentences)
        
        # Return the embeddings as a JSON response
        return json.dumps({"embeddings": embeddings.tolist()})
    except Exception as e:
        # Return any errors that occur during inference
        return json.dumps({"error": str(e)})
