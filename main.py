import os
import sys
from flask import Flask

# Import the libraries to test their initialization
import vertexai
from google.cloud import storage
from vertexai.generative_models import GenerativeModel
from vertexai.vision_models import ImageGenerationModel

# --- Configuration and Initialization ---

print("--- Starting Full Application Initialization ---")
sys.stdout.flush()

app = Flask(__name__)

try:
    # Get environment variables
    PROJECT_ID = os.environ["PROJECT_ID"]
    LOCATION = "europe-west1" # Hard-coded to the correct region

    print(f"STEP 1: Initializing Vertex AI SDK for project {PROJECT_ID} in {LOCATION}...")
    sys.stdout.flush()
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    print("STEP 1: Vertex AI SDK Initialized SUCCESSFULLY.")
    sys.stdout.flush()

    print("STEP 2: Initializing Cloud Storage client...")
    sys.stdout.flush()
    storage_client = storage.Client()
    print("STEP 2: Cloud Storage client Initialized SUCCESSFULLY.")
    sys.stdout.flush()

    print("STEP 3: Loading Gemini Model...")
    sys.stdout.flush()
    gemini_model = GenerativeModel("gemini-1.0-pro-001")
    print("STEP 3: Gemini Model Loaded SUCCESSFULLY.")
    sys.stdout.flush()

    print("STEP 4: Loading Imagen Model...")
    sys.stdout.flush()
    imagen_model = ImageGenerationModel.from_pretrained("imagegeneration@006")
    print("STEP 4: Imagen Model Loaded SUCCESSFULLY.")
    sys.stdout.flush()

    print("--- Application Initialized Successfully ---")
    sys.stdout.flush()

except Exception as e:
    print(f"FATAL: An error occurred during initialization: {e}")
    sys.stdout.flush()
    # We will not crash the app, just log the error
    pass

# --- Simple Test Route ---
@app.route('/')
def initialization_test():
    return 'Initialization test complete. Check the logs for success messages.'

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
