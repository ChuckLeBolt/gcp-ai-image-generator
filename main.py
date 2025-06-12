"""Cloud Run entrypoint that generates a photorealistic background with Vertex AI
and composites a product packshot onto it.

Key improvements over the previous version
------------------------------------------
1. **Exponential back‑off & retry** for Vertex GenAI calls via `retry_call()`.
2. **Propagate upstream status codes**—a ServiceUnavailable (503) coming from
   Vertex is returned to the caller as 503, not rewritten to 500.
3. **Structured logging** in JSON‑friendly format for easier observability.
4. **Region override** via `LOCATION` env var; default remains europe‑west1.
"""

from __future__ import annotations

import io
import json
import logging
import os
import random
import sys
import time
import uuid
from typing import Any, Callable, ParamSpec, TypeVar

import requests
from flask import Flask, jsonify, request
from google.api_core import exceptions as gapi_exceptions
from google.cloud import storage
from PIL import Image
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.vision_models import ImageGenerationModel

# ---------------------------------------------------------------------------
# Configuration & initialization
# ---------------------------------------------------------------------------

PROJECT_ID = os.getenv("PROJECT_ID", "your-gcp-project-id")
GCS_OUTPUT_BUCKET_NAME = os.getenv("GCS_OUTPUT_BUCKET", "your-gcs-output-bucket")
LOCATION = os.getenv("LOCATION", "europe-west1")  # override to us-central1 if models GA there first

vertexai.init(project=PROJECT_ID, location=LOCATION)

storage_client = storage.Client()
output_bucket = storage_client.bucket(GCS_OUTPUT_BUCKET_NAME)

# Vertex AI models
GEMINI_MODEL = GenerativeModel("gemini-2.0-flash")
IMAGEN_MODEL = ImageGenerationModel.from_pretrained("imagegeneration@006")

# Flask
app = Flask(__name__)

# Logging (JSON‑like, prints to stdout so Cloud Run aggregate picks it up)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

P = ParamSpec("P")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# Utility: retry wrapper for transient 503s
# ---------------------------------------------------------------------------

def retry_call(
    fn: Callable[P, R],
    *args: P.args,
    max_attempts: int = 5,
    base_delay: float = 1.5,
    jitter: float = 0.2,
    **kwargs: P.kwargs,
) -> R:
    """Call *fn* with exponential back‑off when Vertex returns ServiceUnavailable.

    Parameters
    ----------
    fn : callable
        The function to invoke.
    max_attempts : int, default 5
        Maximum attempts before letting the exception bubble up.
    base_delay : float, default 1.5
        First sleep in seconds; doubles each retry (geometric).
    jitter : float, default 0.2
        Adds ±20 % randomness to avoid thundering herd.
    """

    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except gapi_exceptions.ServiceUnavailable as exc:
            if attempt == max_attempts:
                logging.error("Vertex ServiceUnavailable after %s attempts: %s", attempt, exc)
                raise

            sleep_time = base_delay * 2 ** (attempt - 1)
            sleep_time *= 1 + random.uniform(-jitter, jitter)
            logging.warning("Retry %s/%s after ServiceUnavailable: sleeping %.1fs", attempt, max_attempts, sleep_time)
            time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def generate_gemini_prompt(general_desc: str, background_desc: str, copy_text: str) -> str:
    """Uses Gemini to create a photorealistic Imagen prompt."""

    meta_prompt = f"""
You are an expert prompt engineer for a text‑to‑image AI model. Combine the
following details into a single, highly descriptive prompt. **Requirements:**
1. Leave a clear, empty space in the centre foreground suitable for pasting a
   product packshot later. Do *not* describe the product itself.
2. Render this text clearly within the scene (without obscuring the empty
   space): '{copy_text}'.
3. Overall style: {general_desc}.

BACKGROUND DETAILS:
- {background_desc}

Output only the final prompt.
"""

    logging.info("Generating Gemini prompt for '%s'", general_desc)
    response = retry_call(GEMINI_MODEL.generate_content, meta_prompt)
    clean_prompt = response.text.strip().replace("\n", " ")
    logging.info("Gemini prompt generated: %s", clean_prompt)
    return clean_prompt


def generate_imagen_background(prompt: str) -> Image.Image:
    """Generates a background image via Imagen."""

    logging.info("Generating background with Imagen…")
    images = retry_call(
        IMAGEN_MODEL.generate_images,
        prompt=prompt,
        number_of_images=1,
        aspect_ratio="1:1",
    )
    image_bytes = images[0]._image_bytes
    return Image.open(io.BytesIO(image_bytes))


def download_packshot(url: str) -> Image.Image:
    """Downloads the packshot and returns a Pillow Image (RGBA)."""

    logging.info("Downloading packshot %s", url)
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGBA")
    except requests.RequestException as exc:
        logging.error("Packshot download failed: %s", exc)
        raise ValueError("Unable to download packshot") from exc


def composite_images(bg: Image.Image, packshot: Image.Image) -> Image.Image:
    """Paste *packshot* centrally onto *bg* (45 % width)."""

    bg_w, bg_h = bg.size
    aspect = packshot.height / packshot.width
    new_w = int(bg_w * 0.45)
    new_h = int(new_w * aspect)
    packshot_resized = packshot.resize((new_w, new_h), Image.Resampling.LANCZOS)
    paste_x = (bg_w - new_w) // 2
    paste_y = (bg_h - new_h) // 2
    bg.paste(packshot_resized, (paste_x, paste_y), packshot_resized)
    return bg


def upload_to_gcs(image: Image.Image) -> str:
    """Uploads *image* to the output bucket and returns its public URL."""

    filename = f"generated-image-{uuid.uuid4()}.png"
    blob = output_bucket.blob(filename)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    blob.upload_from_file(buf, content_type="image/png")
    logging.info("Uploaded image to %s", blob.public_url)
    return blob.public_url


# ---------------------------------------------------------------------------
# Flask route
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"general_description", "background_description", "copy", "packshot_url"}


@app.post("/")
def generate_image() -> tuple[Any, int]:
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON payload"}), 400

    missing = REQUIRED_FIELDS - payload.keys()
    if missing:
        return jsonify({"error": f"Missing required field(s): {', '.join(sorted(missing))}"}), 400

    try:
        prompt = generate_gemini_prompt(
            payload["general_description"],
            payload["background_description"],
            payload["copy"],
        )
        bg_image = generate_imagen_background(prompt)
        packshot = download_packshot(payload["packshot_url"])
        final_image = composite_images(bg_image, packshot)
        url = upload_to_gcs(final_image)
        return (
            jsonify({
                "success": True,
                "imageUrl": url,
                "geminiGeneratedPrompt": prompt,
            }),
            200,
        )

    # ----------------------------- Exception handling ----------------------
    except gapi_exceptions.GoogleAPICallError as exc:
        status = exc.code.value if hasattr(exc, "code") else 503
        logging.error("Vertex API error (%s): %s", status, exc)
        return (
            jsonify({
                "error": "Vertex AI service error.",
                "details": str(exc),
            }),
            status,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        logging.exception("Unhandled exception: %s", exc)
        return (
            jsonify({"error": "Internal server error.", "details": str(exc)}),
            500,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
