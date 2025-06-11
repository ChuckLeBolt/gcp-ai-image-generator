# AI Product Image Generator - Cloud Run Service

This Google Cloud Run function orchestrates a multi-step AI process to generate professional product marketing images. It is deployed automatically from this GitHub repository.

## Workflow

1.  Receives a JSON request with product details.
2.  Uses **Vertex AI Gemini** to intelligently craft a prompt for an image generation model.
3.  Uses **Vertex AI Imagen** to generate a background image based on the prompt, including rendered text.
4.  Downloads a product packshot from a given URL.
5.  Composites the packshot onto the generated background.
6.  Uploads the final image to Google Cloud Storage and returns its public URL.

## How to Use

After the service is deployed, you can send `POST` requests to its public URL.

**Example Request using `curl`:**

```bash
# Replace YOUR_SERVICE_URL with the URL provided by Google Cloud Run
export SERVICE_URL="YOUR_SERVICE_URL"

# Replace the packshot_url with a direct link to a real image (PNG with transparent background is best)
curl -X POST $SERVICE_URL \
-H "Content-Type: application/json" \
-d '{
    "general_description": "A vibrant, colourful, eye-catching studio shot for a soft drink",
    "background_description": "A background of splashing orange and lemon slices, with dynamic water droplets frozen mid-air. Use dramatic, bright lighting.",
    "copy": "Zesty Fresh!",
    "packshot_url": "https://storage.googleapis.com/gweb-uniblog-publish-prod/images/Google_IO_2023_wordmark.width-1200.format-png.png"
}'
```

### Example Success Response:

```json
{
  "success": true,
  "imageUrl": "https://storage.googleapis.com/your-output-bucket/generated-image-....png",
  "geminiGeneratedPrompt": "A vibrant, colourful, eye-catching photorealistic studio shot of a background of splashing orange and lemon slices, with dynamic water droplets frozen mid-air..."
}
```
