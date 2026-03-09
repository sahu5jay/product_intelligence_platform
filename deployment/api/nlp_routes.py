from flask import Blueprint, request, jsonify
import logging
from src.nlp_module.pipeline.prediction_pipeline import PredictPipeline

nlp_bp = Blueprint("nlp_bp", __name__)

@nlp_bp.route("/analyze-text", methods=["POST"])
def analyze_text_route():
    try:
        data = request.json
        text = data.get("text", "")

        logging.info(f"Input text: {text}")

        sentiment = PredictPipeline()
        result = sentiment.predict(text)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })