from flask import Flask, request, jsonify, render_template
import torch
import os
from PIL import Image
import base64
from io import BytesIO
import numpy as np

# Import refactored prediction and gradcam functions
from predict import predict_image
from single_image_gradcam import generate_gradcam

torch.backends.cudnn.benchmark = True

# Default settings for model selection
default_model_type = "transformer"
default_model_variant = "vit"
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Map each (model_type, model_variant) combination to its fixed checkpoint path.
checkpoint_paths = {
    ("densenet", "base"): "checkpoints/densenet_base_best.pth",
    ("densenet", "imagenet"): "checkpoints/densenet_imagenet_best.pth",
    ("resnet", "base"): "checkpoints/resnet_base_best.pth",
    ("resnet", "imagenet"): "checkpoints/resnet_imagenet-freeze_best.pth",
    ("transformer", "vit"): "checkpoints/vit_base_patch16_224_best_auc.pth",
    ("transformer", "swin"): "checkpoints/vit_swin_tiny_patch4_window7_224_best.pth"
}

# Specify the template folder (ensure index.html is inside a folder named "templates")
app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/results")
def results():
    return render_template("results.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieve form data and normalize to lowercase as needed.
        model_type = request.form.get("model_type", default_model_type).lower()
        model_variant = request.form.get("model_variant", default_model_variant).lower()
        # Do not set a default model_name so that get_model() can choose based on model_variant.
        model_name = request.form.get("model_name")

        # Determine checkpoint path based on selected combination.
        checkpoint_path = checkpoint_paths.get((model_type, model_variant))
        if checkpoint_path is None:
            return jsonify({"error": "Invalid model type and variant combination."}), 400

        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files["image"]
        image_path = "temp.jpg"
        image_file.save(image_path)

        # Use the refactored predict_image function
        result, _ = predict_image(
            model_type,
            model_variant,
            model_name,
            checkpoint_path,
            image_path,
            default_device
        )

        # Clean up temporary file
        os.remove(image_path)

        # Return results as JSON
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def image_to_base64(image):
    """
    Convert a PIL Image or numpy array to base64 string.

    Parameters:
        image: PIL Image or numpy array

    Returns:
        str: Base64 encoded string
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            # Normalize to 0-255 if not already
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_img = Image.fromarray(image)
    else:
        pil_img = image

    # Convert to base64
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def resize_to_original(original_img, target_img):
    """
    Resize target image to match the original image dimensions without using antialiasing.
    Just makes pixels bigger or smaller.

    Parameters:
        original_img: PIL Image or numpy array - the reference image
        target_img: numpy array - the image to resize

    Returns:
        numpy array: Resized target image
    """
    if isinstance(original_img, Image.Image):
        width, height = original_img.size
    else:
        height, width = original_img.shape[:2]

    # Convert target_img to PIL for resizing
    if not isinstance(target_img, Image.Image):
        if target_img.dtype != np.uint8:
            if target_img.max() <= 1.0:
                target_img = (target_img * 255).astype(np.uint8)
            else:
                target_img = target_img.astype(np.uint8)

        if len(target_img.shape) == 2:  # Grayscale
            pil_target = Image.fromarray(target_img, mode='L')
        else:  # RGB or RGBA
            pil_target = Image.fromarray(target_img)
    else:
        pil_target = target_img

    # Use nearest neighbor for simple pixel expansion/reduction
    resized = pil_target.resize((width, height), Image.NEAREST)

    # Return as numpy array
    return np.array(resized)


@app.route("/gradcam", methods=["POST"])
def gradcam_endpoint():
    """
    Endpoint to generate GradCAM visualization.
    Expects the following form-data:
      - "model_type": Architecture to use; only "resnet" and "densenet" are supported.
      - "model_variant": Model variant; for example, "base", "imagenet", or "attention".
      - "image": The image file.
    Returns a JSON with the target class, class probabilities, and three base64-encoded images:
      - original_image: The input image
      - gradcam_heatmap: The GradCAM heatmap visualization
      - gradcam_overlay: The heatmap overlaid on the original image
    """
    try:
        # Get model type and variant (for GradCAM, only resnet and densenet are supported)
        model_type = request.form.get("model_type", "resnet").lower()
        model_variant = request.form.get("model_variant", "base").lower()

        if model_type not in ["resnet", "densenet"]:
            return jsonify({"error": "GradCAM only supports 'resnet' and 'densenet' architectures."}), 400

        # Get checkpoint path based on provided model_type and variant.
        checkpoint_path = checkpoint_paths.get((model_type, model_variant))
        if checkpoint_path is None:
            return jsonify({"error": "Invalid architecture/variant combination."}), 400

        # Optional target class parameter.
        target_class = request.form.get("target_class", None)
        if target_class is not None:
            target_class = int(target_class)

        # Ensure an image file was provided.
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files["image"]
        image_path = "temp_gradcam.jpg"
        image_file.save(image_path)

        # Load the original image for later resizing reference
        original_img = Image.open(image_path)

        # Use the generate_gradcam function
        results = generate_gradcam(
            model_type,
            model_variant,
            checkpoint_path,
            image_path,
            target_class,
            default_device
        )

        # Convert heatmap to a visual representation (using jet colormap)
        from matplotlib import cm
        heatmap = results["cam"]
        colormap = cm.get_cmap('jet')
        heatmap_colored = (colormap(heatmap) * 255).astype(np.uint8)

        # Resize heatmap and overlay to match original image dimensions
        heatmap_resized = resize_to_original(original_img, heatmap_colored)
        overlay_resized = resize_to_original(original_img, results["overlay_img"])

        # Prepare response with all three images
        response = {
            "target_class": results["target_class"],
            "probabilities": results["probabilities"],
            "original_image": image_to_base64(original_img),
            "gradcam_heatmap": image_to_base64(heatmap_resized),
            "gradcam_overlay": image_to_base64(overlay_resized)
        }

        # Clean up temporary file
        os.remove(image_path)

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
