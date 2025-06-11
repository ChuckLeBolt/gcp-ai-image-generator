import os
import io
import uuid
import requests

from flask import Flask, request, jsonify
from PIL import Image

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.vision_models import ImageGenerationModel
from google.cloud import storage

# --- Configuration and Initialization ---

# Initialize Flask app
app = Flask(__name__)

# Get environment variables
# These are set during deployment in a later step
try:
    PROJECT_ID = os.environ["PROJECT_ID"]
    GCS_OUTPUT_BUCKET_NAME = os.environ["GCS_OUTPUT_BUCKET"]
except KeyError:
    # This check helps prevent deployment errors if variables aren't set.
    # We will provide default values here, but they will be overridden during deployment.
    PROJECT_ID = "your-gcp-project-id"
    GCS_OUTPUT_BUCKET_NAME = "your-gcs-output-bucket"


LOCATION = "us-central1"

# Initialize Vertex AI SDK
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Initialize Google Cloud Storage client
storage_client = storage.Client()
output_bucket = storage_client.bucket(GCS_OUTPUT_BUCKET_NAME)

# Load AI Models
# Using a specific version is recommended for production
gemini_model = GenerativeModel("gemini-1.0-pro-001")
imagen_model = ImageGenerationModel.from_pretrained("imagegeneration@006")

# --- Helper Functions ---

def generate_gemini_prompt(general_desc, background_desc, copy_text):
    """Uses Gemini to create an optimized prompt for Imagen."""
    
    meta_prompt = f"""
    You are an expert prompt engineer for a text-to-image AI model. Your task is to take the following details and combine them into a single, highly descriptive, and photorealistic prompt.

    CRITICAL INSTRUCTIONS:
    1. The final image must contain a clear, empty space in the center foreground, suitable for placing a product packshot onto it later. Do not describe or generate the product itself in the scene.
    2. The prompt must also include a request to render the following text clearly and legibly within the scene: '{copy_text}'. The text should be well-integrated but not obscure the central empty space.
    3. The overall style should be: {general_desc}.

    BACKGROUND DETAILS:
    - {background_desc}

    Generate only the final, combined prompt. Do not add any conversational text or explanations.
    """
    
    print(f"Generating Gemini prompt for: {general_desc}")
    response = gemini_model.generate_content(meta_prompt)
    
    clean_prompt = response.text.strip().replace("\n", " ")
    print(f"Generated prompt: {clean_prompt}")
    return clean_prompt

def generate_imagen_background(prompt):
    """Generates a background image using Imagen."""
    
    print("Generating background with Imagen...")
    images = imagen_model.generate_images(
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="1:1" # Enforces a square image
    )
    
    # Load the generated image data into a Pillow Image object
    image_bytes = images[0]._image_bytes
    background_image = Image.open(io.BytesIO(image_bytes))
    
    print("Imagen background generated successfully.")
    return background_image

def download_and_prepare_packshot(url):
    """Downloads the packshot from a URL and returns a Pillow Image."""
    
    print(f"Downloading packshot from {url}")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        packshot_image = Image.open(response.raw).convert("RGBA")
        print("Packshot downloaded and opened.")
        return packshot_image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading packshot: {e}")
        raise

def composite_images(background_img, packshot_img):
    """Pastes the packshot onto the center of the background."""
    
    print("Compositing images...")
    # Resize packshot to be 45% of the background's width, maintaining aspect ratio
    bg_width, bg_height = background_img.size
    packshot_aspect_ratio = packshot_img.height / packshot_img.width
    new_width = int(bg_width * 0.45)
    new_height = int(new_width * packshot_aspect_ratio)
    
    resized_packshot = packshot_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate position to paste in the center
    paste_x = (bg_width - new_width) // 2
    paste_y = (bg_height - new_height) // 2
    
    # Paste using the packshot's alpha channel as a mask for transparency
    background_img.paste(resized_packshot, (paste_x, paste_y), resized_packshot)
    print("Compositing complete.")
    return background_img

def upload_to_gcs(image_pil):
    """Uploads a Pillow image object to GCS and returns its public URL."""
    
    print("Uploading final image to GCS...")
    filename = f"generated-image-{uuid.uuid4()}.png"
    blob = output_bucket.blob(filename)
    
    # Save the image to an in-memory buffer
    buffer = io.BytesIO()
    image_pil.save(buffer, 'PNG')
    buffer.seek(0)
    
    # Upload the buffer to the GCS blob
    blob.upload_from_file(buffer, content_type='image/png')
    
    print(f"Image uploaded. Public URL: {blob.public_url}")
    return blob.public_url


# --- Main Flask Route ---

@app.route("/", methods=["POST"])
def process_image_request():
    """Main endpoint to handle the entire image generation workflow."""
    
    # 1. --- Input Validation ---
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400
        
    required_fields = ["general_description", "background_description", "copy", "packshot_url"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400

    try:
        # 2. --- AI Orchestration ---
        gemini_prompt = generate_gemini_prompt(
            data["general_description"],
            data["background_description"],
            data["copy"]
        )
        
        background_image = generate_imagen_background(gemini_prompt)
        packshot_image = download_and_prepare_packshot(data["packshot_url"])
        final_image = composite_images(background_image, packshot_image)
        image_url = upload_to_gcs(final_image)

        # 3. --- Success Response ---
        return jsonify({
            "success": True,
            "imageUrl": image_url,
            "geminiGeneratedPrompt": gemini_prompt
        }), 200

    except Exception as e:
        # 4. --- Error Handling ---
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal error occurred.", "details": str(e)}), 500

if __name__ == "__main__":
    # This is used for local development, which we are not doing.
    # Gunicorn runs the app in a production environment as specified in the Dockerfile.
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
