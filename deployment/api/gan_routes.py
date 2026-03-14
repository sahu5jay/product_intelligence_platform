from flask import Blueprint, request, render_template
from src.gan_module.pipeline.inference_pipeline import GANInferencePipeline

gan_bp = Blueprint("gan_bp", __name__)

@gan_bp.route("/generate", methods=["POST"])
def generate_image_route():

    try:

        label = request.form.get("label")
        num_images = int(request.form.get("num_images"))

        pipeline = InferencePipeline()

        images = pipeline.generate_images(label, num_images)

        labels = pipeline.get_labels()

        return render_template(
            "generate.html",
            labels=labels,
            images=images
        )

    except Exception as e:

        return render_template(
            "generate.html",
            labels=[],
            images=[],
            error=str(e)
        )