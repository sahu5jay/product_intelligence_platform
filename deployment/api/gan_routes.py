from flask import Blueprint, jsonify
from src.gan_module.pipeline.inference_pipeline import InferencePipeline

gan_bp = Blueprint("gan_bp", __name__)

@gan_bp.route("/generate-image", methods=["POST"])
def generate_image_route():

    try:

        pipeline = InferencePipeline()

        pipeline.run_pipeline()

        return jsonify({
            "success": True,
            "message": "Images generated successfully"
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        })