from flask import Blueprint, request, jsonify
from src.nlp_module.pipeline.prediction_pipeline import analyze_text

nlp_bp = Blueprint("nlp_bp", __name__)

@nlp_bp.route("/analyze-text", methods=["POST"])
def analyze_text_route():
    try:
        data = request.json
        text = data.get("text", "")

        result = analyze_text(text)

        return jsonify({
            "success": True,
            "result": result
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })