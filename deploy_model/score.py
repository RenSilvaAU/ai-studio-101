import os
import logging
import json
from sentence_transformers import SentenceTransformer
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

def init():
    global model
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    logging.info("Init complete")

# Define the expected input and output schema
input_sample = StandardPythonParameterType({"sentences": "I feel great this morning"})
output_sample = StandardPythonParameterType([[]])  # Expecting a list of embeddings (lists of floats)

@input_schema(param_name="data", param_type=input_sample)
@output_schema(output_type=output_sample)
def run(data):
    logging.info("Request received")
    sentences = data.get("sentences")
    embeddings = model.encode(sentences)
    logging.info("Request processed")
    return embeddings.tolist()
